"""
PortfolioTracker — maintains live position state and net Greeks.

State is updated on two event streams:
  1. FillEvent (from fill_recv)  — adds/removes legs
  2. VolSurface (from surface_portfolio_recv) — reprices existing legs

On every update, a fresh PortfolioState is emitted into portfolio_state_send
for HedgeEngine consumption.

Instrument classification
-------------------------
  • Options  — greeks computed via Black-76 using SVI ATM IV (or per-strike IV)
  • Futures / Swap — Δ = ±1 per unit, Γ = Θ = ν = 0
"""

from __future__ import annotations

import logging
import math
import time
from copy import copy
from typing import Any

import trio

from scalp.channels import Channels
from scalp.config import Settings
from scalp.exchange.okx_rest import OKXRestClient
from scalp.models.black76 import greeks
from scalp.schema import FillEvent, Leg, PortfolioState, VolSurface

logger = logging.getLogger(__name__)

_FUTURES_SUFFIXES = ("-SWAP", "-PERP", "-FUT")


def _is_option(inst_id: str) -> bool:
    """Heuristic: OKX option IDs end with -C or -P after the strike."""
    return inst_id.endswith("-C") or inst_id.endswith("-P")


def _parse_option_fields(
    inst_id: str,
) -> tuple[float, str, float] | None:
    """
    Extract (strike, expiry_str, opt_type_flag) from an OKX option instId.

    Returns None if parsing fails.
    """
    parts = inst_id.split("-")
    if len(parts) != 5:
        return None
    try:
        strike = float(parts[3])
    except ValueError:
        return None
    expiry_str = parts[2]
    is_call = parts[4] == "C"
    return strike, expiry_str, float(is_call)


def _compute_option_greeks(
    inst_id: str,
    quantity: float,
    surface: VolSurface,
) -> dict[str, float]:
    """
    Compute Black-76 Greeks for an option leg using the fitted SVI surface.

    Returns zero Greeks on any failure (e.g. expiry not in surface).
    """
    parsed = _parse_option_fields(inst_id)
    if parsed is None:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    strike, expiry_str, is_call_f = parsed
    is_call = bool(is_call_f)

    T = surface.expiry_times.get(expiry_str)
    if T is None or T <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    log_k = math.log(strike / surface.futures_price) if surface.futures_price > 0 else 0.0

    try:
        sigma = surface.iv_at(expiry_str, log_k)
    except (KeyError, Exception):
        sigma = surface.atm_iv

    if sigma <= 0 or surface.futures_price <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    raw = greeks(surface.futures_price, strike, sigma, T, is_call)
    # Scale by signed quantity
    return {k: v * quantity for k, v in raw.items()}


class PortfolioTracker:
    """
    Manages a dict of Leg objects keyed by instrument ID.

    Thread-note: this class is only ever mutated from within the trio task;
    no external locking is needed.
    """

    def __init__(self, futures_specs: dict[str, dict[str, Any]] | None = None) -> None:
        self._legs: dict[str, Leg] = {}
        self._surface: VolSurface | None = None
        self._futures_price: float = 0.0
        self._futures_specs = futures_specs or {}

    # ── Mutation ───────────────────────────────────────────────────────────────

    def apply_fill(self, fill: FillEvent) -> None:
        """Update position from a fill event."""
        signed_qty = fill.quantity if fill.side == "buy" else -fill.quantity

        leg = self._legs.get(fill.instrument)
        if leg is None:
            leg = Leg(instrument=fill.instrument, quantity=0.0)
            self._legs[fill.instrument] = leg

        leg.quantity += signed_qty

        if abs(leg.quantity) < 1e-9:
            del self._legs[fill.instrument]
            logger.debug("Leg closed: %s", fill.instrument)
        else:
            logger.debug(
                "Leg updated: %s qty=%.6f (fill side=%s qty=%.6f)",
                fill.instrument, leg.quantity, fill.side, fill.quantity,
            )

    def seed_position(self, instrument: str, signed_quantity: float) -> None:
        """Seed tracker position directly from bootstrap account snapshot."""
        if abs(signed_quantity) < 1e-9:
            return
        self._legs[instrument] = Leg(instrument=instrument, quantity=signed_quantity)

    def _futures_delta(self, inst_id: str, contracts: float) -> float:
        """
        Convert futures/swap position size to base-asset delta.

        For inverse swaps, quantity is in USD contracts and must be divided by price.
        For linear products, quantity is already in base contracts.
        """
        spec = self._futures_specs.get(inst_id, {})
        ct_type = str(spec.get("ctType") or "inverse").lower()
        try:
            ct_val = float(spec.get("ctVal") or 100.0)
        except (TypeError, ValueError):
            ct_val = 100.0
        try:
            ct_mult = float(spec.get("ctMult") or 1.0)
        except (TypeError, ValueError):
            ct_mult = 1.0

        notional_per_contract = ct_val * ct_mult
        if notional_per_contract <= 0:
            return contracts
        if ct_type == "inverse":
            if self._futures_price <= 0:
                return 0.0
            return contracts * notional_per_contract / self._futures_price
        return contracts * notional_per_contract

    def reprice(self, surface: VolSurface) -> None:
        """Recompute all Greeks using the latest VolSurface."""
        self._surface = surface
        self._futures_price = surface.futures_price

        for inst_id, leg in self._legs.items():
            if _is_option(inst_id):
                g = _compute_option_greeks(inst_id, leg.quantity, surface)
                leg.delta = g["delta"]
                leg.gamma = g["gamma"]
                leg.theta = g["theta"]
                leg.vega = g["vega"]
            else:
                # Futures/swap delta must be converted from contracts to base units.
                leg.delta = self._futures_delta(inst_id, leg.quantity)
                leg.gamma = 0.0
                leg.theta = 0.0
                leg.vega = 0.0

    def snapshot(self) -> PortfolioState:
        """Return an immutable snapshot of current aggregate Greeks."""
        legs_copy = {k: copy(v) for k, v in self._legs.items()}
        delta_net = sum(l.delta for l in legs_copy.values())
        gamma_net = sum(l.gamma for l in legs_copy.values())
        theta_net = sum(l.theta for l in legs_copy.values())
        vega_net = sum(l.vega for l in legs_copy.values())

        return PortfolioState(
            legs=legs_copy,
            delta_net=delta_net,
            gamma_net=gamma_net,
            theta_net=theta_net,
            vega_net=vega_net,
            futures_price=self._futures_price,
        )


# ── Trio task ─────────────────────────────────────────────────────────────────

async def portfolio_tracker_task(
    channels: Channels,
    settings: Settings,
    bootstrap_positions: list[dict[str, Any]] | None = None,
) -> None:
    """
    Merge fill and surface events, push PortfolioState updates downstream.

    Uses trio.open_nursery to listen on two channels concurrently:
    whenever either fires, we recompute and push a new state snapshot.
    """
    futures_specs: dict[str, dict[str, Any]] = {}
    rest = OKXRestClient(settings)
    inst = await trio.to_thread.run_sync(
        lambda: rest.get_futures_instrument(settings.futures_inst_id),
        abandon_on_cancel=True,
    )
    if inst:
        futures_specs[settings.futures_inst_id] = inst
    tracker = PortfolioTracker(futures_specs=futures_specs)
    logger.info("PortfolioTracker task started")

    seeded_count = 0
    for pos in bootstrap_positions or []:
        inst_id = str(pos.get("instId") or "")
        if not inst_id:
            continue
        try:
            signed_qty = float(pos.get("pos") or 0.0)
        except (TypeError, ValueError):
            continue
        if abs(signed_qty) < 1e-9:
            continue
        pos_side = str(pos.get("posSide") or "").lower()
        if pos_side == "short" and signed_qty > 0:
            signed_qty = -signed_qty
        if pos_side == "long" and signed_qty < 0:
            signed_qty = abs(signed_qty)
        tracker.seed_position(inst_id, signed_qty)
        seeded_count += 1
    if seeded_count > 0:
        logger.info("PortfolioTracker seeded with %d bootstrap position(s)", seeded_count)

    ignore_historical_fills = seeded_count > 0
    fill_cutoff_ts = time.time()

    def _publish_state(state: PortfolioState) -> None:
        try:
            channels.portfolio_state_send.send_nowait(state)
        except trio.WouldBlock:
            pass
        try:
            channels.portfolio_state_order_send.send_nowait(state)
        except trio.WouldBlock:
            pass

    if seeded_count > 0:
        _publish_state(tracker.snapshot())

    async def _handle_fills() -> None:
        async for fill in channels.fill_recv:
            if ignore_historical_fills and fill.timestamp < fill_cutoff_ts:
                continue
            tracker.apply_fill(fill)
            _publish_state(tracker.snapshot())

    async def _handle_surface() -> None:
        async for surface in channels.surface_portfolio_recv:
            tracker.reprice(surface)
            _publish_state(tracker.snapshot())

    async with trio.open_nursery() as nursery:
        nursery.start_soon(_handle_fills)
        nursery.start_soon(_handle_surface)
