"""
DataNormalizer — fan-out trio task.

Consumes raw WS events from `raw_recv` and routes them to typed downstream
channels based on the OKX channel name:

    opt-summary       → ob_snapshot_send   (one OBSnapshot per expiry)
    candle5m          → candle_send
    mark-price        → updates internal futures price (no dedicated channel)
    orders (fills)    → fill_send

OKX opt-summary data format (one entry per live option instrument):
  {
    "instId":  "BTC-USD-250328-90000-C",
    "uly":     "BTC-USD",
    "fwdPx":   "95000",
    "bidVol":  "0.48",
    "askVol":  "0.52",
    "markVol": "0.50",
    "ts":      "1740000000000"
    ... (delta, gamma, etc. — not used here, we recompute from B76)
  }

The normalizer is stateful: it maintains the most recent IV per instrument
so that a partial opt-summary update (only changed instruments) does not
lose the last known IV for unchanged ones.  The full merged state is used
to produce per-expiry OBSnapshot objects for the SVI fitter.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

import trio

from scalp.channels import Channels
from scalp.config import Settings
from scalp.debug_log import debug_log
from scalp.schema import Candle, FillEvent, OBSnapshot, RawWSEvent

logger = logging.getLogger(__name__)
_NORM_CH_COUNT: dict[str, int] = {}


# ── Option instrument parser ───────────────────────────────────────────────────

def _parse_expiry_str(expiry_str: str) -> datetime:
    """
    Convert OKX expiry format (yymmdd) to a UTC datetime at 08:00 UTC.
    OKX options expire at 08:00 UTC on the expiry date.
    """
    year = 2000 + int(expiry_str[:2])
    month = int(expiry_str[2:4])
    day = int(expiry_str[4:6])
    return datetime(year, month, day, 8, 0, 0, tzinfo=timezone.utc)


def _time_to_expiry(expiry_str: str) -> float:
    """Return years-to-expiry; clipped at 1/365 to avoid division by zero."""
    expiry_dt = _parse_expiry_str(expiry_str)
    now = datetime.now(timezone.utc)
    seconds = (expiry_dt - now).total_seconds()
    return max(seconds / (365.25 * 24 * 3600), 1.0 / 365.25)


def _parse_inst_id(inst_id: str) -> tuple[str, float, str, str] | None:
    """
    Parse OKX option instrument ID.

    Returns (underlying, strike, expiry_str, opt_type) or None on failure.
    Example: "BTC-USD-250328-90000-C" → ("BTC-USD", 90000.0, "250328", "C")
    """
    parts = inst_id.split("-")
    if len(parts) != 5:
        return None
    underlying = f"{parts[0]}-{parts[1]}"
    expiry_str = parts[2]
    try:
        strike = float(parts[3])
    except ValueError:
        return None
    opt_type = parts[4]
    return underlying, strike, expiry_str, opt_type


# ── Normalizer state ──────────────────────────────────────────────────────────

class _OptionState:
    """Latest IV snapshot for a single option instrument."""
    __slots__ = ("inst_id", "strike", "expiry_str", "opt_type", "fwd_px", "mid_iv")

    def __init__(
        self,
        inst_id: str,
        strike: float,
        expiry_str: str,
        opt_type: str,
        fwd_px: float,
        mid_iv: float,
    ) -> None:
        self.inst_id = inst_id
        self.strike = strike
        self.expiry_str = expiry_str
        self.opt_type = opt_type
        self.fwd_px = fwd_px
        self.mid_iv = mid_iv


class DataNormalizer:
    """
    Stateful normalizer that merges incremental opt-summary updates and emits
    per-expiry OBSnapshot objects.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # instId → _OptionState
        self._option_state: dict[str, _OptionState] = {}
        self._futures_price: float = 0.0
        self._last_mark_ts: float = 0.0

    # ── opt-summary ────────────────────────────────────────────────────────────

    def process_opt_summary(self, data: list[dict[str, Any]]) -> list[OBSnapshot]:
        """
        Merge opt-summary entries and produce one OBSnapshot per expiry that
        has ≥ 5 valid strikes with a defined mid IV.
        """
        for entry in data:
            inst_id: str = entry.get("instId", "")
            parsed = _parse_inst_id(inst_id)
            if parsed is None:
                continue

            underlying, strike, expiry_str, opt_type = parsed

            try:
                bid_vol = float(entry.get("bidVol") or 0)
                ask_vol = float(entry.get("askVol") or 0)
                fwd_px = float(entry.get("fwdPx") or 0)
            except (TypeError, ValueError):
                continue

            if bid_vol <= 0 or ask_vol <= 0 or fwd_px <= 0:
                continue

            mid_iv = (bid_vol + ask_vol) / 2.0
            self._option_state[inst_id] = _OptionState(
                inst_id=inst_id,
                strike=strike,
                expiry_str=expiry_str,
                opt_type=opt_type,
                fwd_px=fwd_px,
                mid_iv=mid_iv,
            )

        return self._build_snapshots()

    def _build_snapshots(self) -> list[OBSnapshot]:
        """
        Group current option state by expiry and build OBSnapshot list.

        We use only call options to avoid put-call parity ambiguity in the
        SVI fit.  Log-strikes are sorted ascending.
        """
        # Group calls by expiry
        expiry_groups: dict[str, list[_OptionState]] = {}
        for state in self._option_state.values():
            if state.opt_type != "C":
                continue
            expiry_groups.setdefault(state.expiry_str, []).append(state)

        snapshots: list[OBSnapshot] = []
        now_ts = time.time()

        for expiry_str, states in expiry_groups.items():
            if len(states) < 5:
                continue

            # Sort by strike
            states.sort(key=lambda s: s.strike)
            fwd_px = states[0].fwd_px
            T = _time_to_expiry(expiry_str)

            if T <= 0:
                continue  # expired

            log_strikes = np.array([np.log(s.strike / fwd_px) for s in states])
            total_variances = np.array([s.mid_iv**2 * T for s in states])

            # Drop non-positive total variances
            valid = total_variances > 0
            if valid.sum() < 5:
                continue

            snapshots.append(OBSnapshot(
                underlying=self._settings.uly,
                expiry_str=expiry_str,
                forward_price=fwd_px,
                log_strikes=log_strikes[valid],
                total_variances=total_variances[valid],
                time_to_expiry=T,
                timestamp=now_ts,
            ))

        return snapshots

    # ── candle5m ───────────────────────────────────────────────────────────────

    @staticmethod
    def process_candle(
        arg: dict[str, str],
        data: list[dict[str, Any]],
    ) -> list[Candle]:
        """Parse candle5m push entries."""
        inst_id = arg.get("instId", "")
        candles: list[Candle] = []
        for entry in data:
            # OKX index candle: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
            if len(entry) < 6:
                continue
            try:
                candles.append(Candle(
                    instrument=inst_id,
                    open_time=float(entry[0]),
                    open=float(entry[1]),
                    high=float(entry[2]),
                    low=float(entry[3]),
                    close=float(entry[4]),
                    volume=float(entry[5]),
                ))
            except (IndexError, TypeError, ValueError):
                continue
        return candles

    # ── orders (fills) ─────────────────────────────────────────────────────────

    @staticmethod
    def process_order(data: list[dict[str, Any]]) -> list[FillEvent]:
        """Convert OKX order push data to FillEvent list (filled orders only)."""
        fills: list[FillEvent] = []
        for entry in data:
            state = entry.get("state", "")
            if state not in ("filled", "partially_filled"):
                continue
            try:
                fills.append(FillEvent(
                    order_id=entry.get("ordId", ""),
                    client_order_id=entry.get("clOrdId", ""),
                    instrument=entry.get("instId", ""),
                    side=entry.get("side", ""),
                    quantity=float(entry.get("fillSz") or entry.get("sz") or 0),
                    price=float(entry.get("fillPx") or entry.get("avgPx") or 0),
                    fee=float(entry.get("fee") or 0),
                    order_type=entry.get("tag", "TRADE"),  # tag set by OrderManager
                    timestamp=float(entry.get("uTime") or time.time() * 1000) / 1000,
                ))
            except (KeyError, TypeError, ValueError) as exc:
                logger.debug("Order parse error: %s — %s", exc, entry)
        return fills

    # ── mark-price ─────────────────────────────────────────────────────────────

    def process_mark_price(self, data: list[dict[str, Any]]) -> None:
        """Update internal futures price cache from mark-price push."""
        for entry in data:
            try:
                px = float(entry.get("markPx") or 0)
                if px > 0:
                    self._futures_price = px
                    self._last_mark_ts = time.time()
            except (TypeError, ValueError):
                pass

    @property
    def futures_price(self) -> float:
        return self._futures_price

    @property
    def last_mark_ts(self) -> float:
        return self._last_mark_ts


# ── Trio task ─────────────────────────────────────────────────────────────────

async def normalizer_task(channels: Channels, settings: Settings) -> None:
    """
    Fan-out: consume `raw_recv`, route to typed downstream channels.
    """
    norm = DataNormalizer(settings)
    logger.info("Normalizer task started")
    last_activity_log = time.time()
    channel_counts: dict[str, int] = {"opt-summary": 0, "mark-price": 0, "orders": 0, "candle": 0}
    last_activity_counts: dict[str, int] = channel_counts.copy()

    async for event in channels.raw_recv:
        ch = event.channel
        n = _NORM_CH_COUNT.get(ch, 0) + 1
        _NORM_CH_COUNT[ch] = n
        if n <= 3 or n % 50 == 0:
            # region agent log
            debug_log(
                hypothesis_id="H3",
                location="scalp/exchange/normalizer.py:normalizer_task",
                message="normalizer_received_raw_event",
                data={"channel": ch, "count": n},
            )
            # endregion

        if ch == "opt-summary":
            channel_counts["opt-summary"] += 1
            snapshots = norm.process_opt_summary(event.data)
            if snapshots:
                # region agent log
                debug_log(
                    hypothesis_id="H4",
                    location="scalp/exchange/normalizer.py:normalizer_task",
                    message="normalizer_built_snapshots",
                    data={"snapshots_count": len(snapshots)},
                )
                # endregion
            for snap in snapshots:
                try:
                    channels.ob_snapshot_send.send_nowait(snap)
                except trio.WouldBlock:
                    # Fitter is busy; overwrite the oldest snapshot for this expiry
                    # by dropping the channel item and replacing it.
                    logger.debug("ob_snapshot_ch full, dropping oldest for %s", snap.expiry_str)

        elif ch in ("candle5m", "index-candle5m", "index-candles5m"):
            channel_counts["candle"] += 1
            parsed = DataNormalizer.process_candle(event.arg, event.data)
            if parsed:
                # region agent log
                debug_log(
                    hypothesis_id="H5",
                    location="scalp/exchange/normalizer.py:normalizer_task",
                    message="normalizer_parsed_candles",
                    data={"candles_count": len(parsed), "instId": event.arg.get("instId", "")},
                )
                # endregion
            for candle in parsed:
                try:
                    channels.candle_send.send_nowait(candle)
                except trio.WouldBlock:
                    logger.debug("candle_ch full, dropping candle")

        elif ch == "mark-price":
            channel_counts["mark-price"] += 1
            norm.process_mark_price(event.data)

        elif ch == "orders":
            channel_counts["orders"] += 1
            for fill in DataNormalizer.process_order(event.data):
                try:
                    channels.fill_send.send_nowait(fill)
                except trio.WouldBlock:
                    logger.warning("fill_ch full — fill event dropped! %s", fill)

        now = time.time()
        if now - last_activity_log >= 30.0:
            mark_age = -1.0
            if norm.last_mark_ts > 0:
                mark_age = now - norm.last_mark_ts
            d_opt = channel_counts["opt-summary"] - last_activity_counts["opt-summary"]
            d_mark = channel_counts["mark-price"] - last_activity_counts["mark-price"]
            d_candle = channel_counts["candle"] - last_activity_counts["candle"]
            d_orders = channel_counts["orders"] - last_activity_counts["orders"]
            logger.info(
                "Activity(30s): opt=+%d mark=+%d candle=+%d orders=+%d | total opt=%d mark=%d candle=%d orders=%d | cached_opts=%d mark_age=%.1fs",
                d_opt,
                d_mark,
                d_candle,
                d_orders,
                channel_counts["opt-summary"],
                channel_counts["mark-price"],
                channel_counts["candle"],
                channel_counts["orders"],
                len(norm._option_state),
                mark_age,
            )
            last_activity_counts = channel_counts.copy()
            last_activity_log = now
