"""
OrderManager — unified TRADE and HEDGE order lifecycle.

Consumes from two input channels:
  • signal_recv  — RegimeEvent from SignalEngine → translates to TRADE orders
  • order_recv   — Order from HedgeEngine → HEDGE market orders

Order state machine:
    PENDING → (REST place_order) → OPEN → (WS fill) → FILLED
                                        ↘ (WS cancel) → CANCELLED
                                        ↘ (error)     → FAILED

Strategy execution (TRADE)
--------------------------
On a regime transition the OrderManager executes a simple ATM straddle
(buy call + buy put for LONG_VOL, sell call + sell put for SHORT_VOL) on
the nearest expiry using the currently known ATM strike.  Position sizing
is a fixed notional of `trade_notional_contracts` (configurable).

On return to NEUTRAL the open option legs are closed by sending offsetting
market orders for each leg in the live portfolio.

Note: this is deliberately minimal for a demo.  In production, instrument
selection, position sizing, and risk checks would be more sophisticated.

REST calls run in a thread worker via trio.to_thread.run_sync, protected
by the RateLimiter.
"""

from __future__ import annotations

import math
import logging
import uuid
from typing import Any

import trio

from scalp.channels import Channels
from scalp.config import Settings
from scalp.debug_log import debug_log
from scalp.exchange.okx_rest import OKXRestClient
from scalp.execution.rate_limiter import RateLimiter
from scalp.schema import (
    FillEvent,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Regime,
    RegimeEvent,
)

logger = logging.getLogger(__name__)

# Notional size per option leg in contracts (base currency units on OKX)
_TRADE_SIZE = "1"


def _delta_to_contracts(
    delta_abs: float,
    futures_price: float,
    ct_type: str,
    ct_val: float,
    ct_mult: float,
) -> float:
    """Convert base-asset delta (e.g. BTC) to swap contract quantity."""
    notional_per_contract = ct_val * ct_mult
    if notional_per_contract <= 0:
        return 0.0
    if futures_price <= 0:
        return 0.0
    if ct_type == "inverse":
        # inverse swap: one contract delta in base ~= (ct_val * ct_mult) / F
        return delta_abs * futures_price / notional_per_contract
    # fallback for linear instruments
    return delta_abs / notional_per_contract


def _okx_safe_clordid(raw: str) -> str:
    """
    OKX rejects some clOrdId formats. Keep only [A-Za-z0-9] and cap to 32 chars.
    """
    cleaned = "".join(ch for ch in raw if ch.isalnum())[:32]
    if cleaned:
        return cleaned
    return f"scalp{uuid.uuid4().hex[:24]}"


class OrderManager:
    def __init__(self, rest: OKXRestClient, rate_limiter: RateLimiter) -> None:
        self._rest = rest
        self._rl = rate_limiter
        # live orders: client_order_id → Order
        self._orders: dict[str, Order] = {}
        # lightweight option instrument cache: uly -> list[inst dict]
        self._opt_cache: dict[str, list[dict[str, Any]]] = {}
        self._futures_inst_cache: dict[str, dict[str, Any]] = {}

    async def _resolve_option_pair(
        self,
        uly: str,
        target_expiry: str,
        target_strike: float,
    ) -> tuple[str, str] | None:
        """
        Pick the nearest listed (call, put) pair for target expiry/strike.
        """
        await self._rl.acquire("trade")
        instruments = await trio.to_thread.run_sync(
            lambda: self._rest.get_option_instruments(uly),
            abandon_on_cancel=True,
        )
        self._opt_cache[uly] = instruments

        pairs: dict[float, dict[str, str]] = {}
        for inst in instruments:
            inst_id = str(inst.get("instId", ""))
            parts = inst_id.split("-")
            if len(parts) != 5:
                continue
            expiry = parts[2]
            if expiry != target_expiry:
                continue
            try:
                strike = float(parts[3])
            except ValueError:
                continue
            cp = parts[4]
            if cp not in ("C", "P"):
                continue
            pairs.setdefault(strike, {})[cp] = inst_id

        valid = [(k, v) for k, v in pairs.items() if "C" in v and "P" in v]
        if not valid:
            return None

        strike, pair = min(valid, key=lambda kv: abs(kv[0] - target_strike))
        # region agent log
        debug_log(
            hypothesis_id="H8",
            location="scalp/execution/order_manager.py:_resolve_option_pair",
            message="order_manager_resolved_option_pair",
            data={
                "target_expiry": target_expiry,
                "target_strike": target_strike,
                "resolved_strike": strike,
            },
        )
        # endregion
        return pair["C"], pair["P"]

    # ── Order placement ────────────────────────────────────────────────────────

    async def place(self, order: Order) -> str | None:
        """
        Send a market order to OKX.  Returns the exchange order_id or None on
        failure.  The REST call runs in a thread and is rate-limited.
        """
        await self._rl.acquire("trade")

        raw_coid = order.client_order_id or f"scalp{uuid.uuid4().hex[:24]}"
        coid = _okx_safe_clordid(raw_coid)
        order.client_order_id = coid
        self._orders[coid] = order
        order.status = OrderStatus.PENDING

        rest = self._rest

        def _do_place() -> dict[str, Any]:
            is_option = order.instrument.endswith("-C") or order.instrument.endswith("-P")
            ord_type = "market"
            price = ""
            size_value = order.quantity
            if is_option:
                ticker = rest.get_ticker(order.instrument) or {}
                if order.side == OrderSide.BUY:
                    price = str(ticker.get("askPx") or ticker.get("last") or "")
                else:
                    price = str(ticker.get("bidPx") or ticker.get("last") or "")
                ord_type = "limit"
            else:
                inst = self._futures_inst_cache.get(order.instrument)
                if inst is None:
                    inst = rest.get_futures_instrument(order.instrument) or {}
                    self._futures_inst_cache[order.instrument] = inst
                try:
                    lot = float(inst.get("lotSz") or 1.0)
                except (TypeError, ValueError):
                    lot = 1.0
                try:
                    min_sz = float(inst.get("minSz") or lot)
                except (TypeError, ValueError):
                    min_sz = lot

                if order.order_type == OrderType.HEDGE:
                    ct_type = str(inst.get("ctType") or "inverse").lower()
                    try:
                        ct_val = float(inst.get("ctVal") or 100.0)
                    except (TypeError, ValueError):
                        ct_val = 100.0
                    try:
                        ct_mult = float(inst.get("ctMult") or 1.0)
                    except (TypeError, ValueError):
                        ct_mult = 1.0
                    ticker = rest.get_ticker(order.instrument) or {}
                    try:
                        futures_price = float(
                            ticker.get("last")
                            or ticker.get("markPx")
                            or ticker.get("idxPx")
                            or 0.0
                        )
                    except (TypeError, ValueError):
                        futures_price = 0.0
                    size_value = _delta_to_contracts(
                        delta_abs=order.quantity,
                        futures_price=futures_price,
                        ct_type=ct_type,
                        ct_val=ct_val,
                        ct_mult=ct_mult,
                    )

                if lot > 0:
                    # Round down to nearest lot multiple to avoid reject spam.
                    size_value = math.floor(size_value / lot) * lot
                if size_value < min_sz:
                    return {
                        "skip": True,
                        "reason": (
                            f"size below min size after normalization "
                            f"(size={size_value:.8f}, min={min_sz:.8f})"
                        ),
                    }
            return rest.place_order(
                inst_id=order.instrument,
                side=order.side.value,
                size=f"{size_value:.8f}",
                client_order_id=coid,
                ord_type=ord_type,
                price=price,
            )

        try:
            result = await trio.to_thread.run_sync(_do_place, abandon_on_cancel=True)
        except Exception as exc:
            logger.error("place_order exception: %s", exc)
            order.status = OrderStatus.FAILED
            return None

        if result.get("skip"):
            logger.warning(
                "Order placement skipped: %s %s — %s",
                order.side.value,
                order.instrument,
                result.get("reason"),
            )
            order.status = OrderStatus.FAILED
            return None

        data = result.get("data", [{}])
        if not data:
            order.status = OrderStatus.FAILED
            return None

        item = data[0]
        if item.get("sCode") != "0":
            logger.warning(
                "Order placement failed: %s %s — %s",
                order.side.value, order.instrument, item.get("sMsg"),
            )
            order.status = OrderStatus.FAILED
            return None

        order_id = item.get("ordId", "")
        order.order_id = order_id
        order.status = OrderStatus.OPEN
        logger.info(
            "Order placed [%s]: %s %s %.6f %s  ordId=%s",
            order.order_type.value, order.side.value,
            order.instrument, order.quantity,
            "", order_id,
        )
        return order_id

    def apply_fill(self, fill: FillEvent) -> None:
        """Update order status when a fill event arrives."""
        order = self._orders.get(fill.client_order_id)
        if order is None:
            return
        order.status = OrderStatus.FILLED
        logger.info(
            "Order filled [%s]: %s %s qty=%.6f @ %.2f",
            order.order_type.value, fill.side, fill.instrument,
            fill.quantity, fill.price,
        )

    # ── Regime-driven TRADE orders ─────────────────────────────────────────────

    async def handle_regime(
        self,
        event: RegimeEvent,
        settings: Settings,
        nearest_expiry: str,
        atm_strike: float,
    ) -> None:
        """
        Translate a RegimeEvent into ATM straddle orders.

        LONG_VOL  → buy  call + buy  put  (long straddle)
        SHORT_VOL → sell call + sell put  (short straddle)
        NEUTRAL   → log only (leg closure is handled separately)
        """
        regime = event.regime

        if regime == Regime.NEUTRAL:
            logger.info("Regime → NEUTRAL (κ=%.4f); no new trades", event.kappa)
            return

        side = OrderSide.BUY if regime == Regime.LONG_VOL else OrderSide.SELL
        uly = settings.uly
        resolved = await self._resolve_option_pair(uly, nearest_expiry, atm_strike)
        if resolved is None:
            logger.warning(
                "No listed call/put pair found for %s expiry=%s near strike=%.2f",
                uly, nearest_expiry, atm_strike,
            )
            return
        call_id, put_id = resolved

        logger.info(
            "Regime → %s (κ=%.4f): %s straddle on %s/%s",
            regime.value, event.kappa, side.value, call_id, put_id,
        )

        for inst_id in (call_id, put_id):
            order = Order(
                instrument=inst_id,
                side=side,
                quantity=float(_TRADE_SIZE),
                order_type=OrderType.TRADE,
                client_order_id=f"trd{uuid.uuid4().hex[:16]}",
            )
            await self.place(order)


# ── Trio task ─────────────────────────────────────────────────────────────────

async def order_manager_task(channels: Channels, settings: Settings) -> None:
    """
    Consume regime events and hedge orders, dispatch to OKX, route fills.

    Maintains a shared RateLimiter and REST client for all order operations.
    Two sub-tasks run concurrently: one for signal/trade orders, one for
    hedge orders.  Fills from the WS (via fill_recv) are echoed back so
    PortfolioTracker gets them — the normalizer already puts them in fill_ch,
    so here we only track internal order state.
    """
    rest = OKXRestClient(settings)
    rate_limiter = RateLimiter()
    manager = OrderManager(rest, rate_limiter)

    logger.info("OrderManager task started")

    async def _handle_signals() -> None:
        signal_count = 0
        async for event in channels.signal_recv:
            signal_count += 1
            if signal_count <= 3 or signal_count % 20 == 0:
                # region agent log
                debug_log(
                    hypothesis_id="H8",
                    location="scalp/execution/order_manager.py:_handle_signals",
                    message="order_manager_received_signal",
                    data={
                        "count": signal_count,
                        "regime": event.regime.value,
                        "kappa": event.kappa,
                    },
                )
                # endregion
            if not event.nearest_expiry or event.atm_strike <= 0:
                logger.warning(
                    "RegimeEvent missing ATM surface info; skipping trade order"
                )
                continue
            await manager.handle_regime(
                event, settings, event.nearest_expiry, event.atm_strike
            )

    async def _handle_hedge_orders() -> None:
        async for order in channels.order_recv:
            await manager.place(order)

    async with trio.open_nursery() as nursery:
        nursery.start_soon(_handle_signals)
        nursery.start_soon(_handle_hedge_orders)
