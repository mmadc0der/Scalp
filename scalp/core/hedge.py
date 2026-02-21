"""
HedgeEngine — Whalley-Wilmott optimal delta-hedging bandwidth.

Reference: Whalley & Wilmott (1997) "An asymptotic analysis of an optimal
hedging model for option pricing with transaction costs".

The engine polls the latest PortfolioState at a configurable interval
(settings.hedge_poll_interval).  A hedge order is triggered whenever the
absolute net delta exceeds the Whalley-Wilmott half-width w:

    w = (3 φ Γ²_Net F / 2 λ)^{1/3}

where:
    φ   — proportional transaction cost (fees + slippage)
    Γ_Net — total portfolio gamma
    F   — current futures price
    λ   — risk-aversion parameter

The hedge size is the full net delta (we delta-zero the position rather
than just touching the boundary, which is the standard implementation).

Futures quantity sign convention: positive = long, negative = short.
A hedge BUY offsets a negative Δ_Net; a hedge SELL offsets positive Δ_Net.
"""

from __future__ import annotations

import logging
import uuid

import trio

from scalp.channels import Channels
from scalp.config import Settings
from scalp.schema import Order, OrderSide, OrderType, PortfolioState

logger = logging.getLogger(__name__)

_MIN_BANDWIDTH = 1e-6   # floor to avoid degenerate w=0 with zero gamma


def whalley_wilmott_bandwidth(
    gamma_net: float,
    futures_price: float,
    phi: float,
    lambda_risk: float,
) -> float:
    """
    Compute the Whalley-Wilmott optimal no-trade half-bandwidth w.

    w = ( 3 φ Γ² F / 2 λ )^{1/3}
    """
    gamma2 = gamma_net**2
    if gamma2 < 1e-30 or futures_price <= 0:
        return _MIN_BANDWIDTH
    numerator = 3.0 * phi * gamma2 * futures_price
    denominator = 2.0 * lambda_risk
    return (numerator / denominator) ** (1.0 / 3.0)


def _hedge_qty_and_side(delta_net: float) -> tuple[float, OrderSide]:
    """
    Return the hedge quantity and direction to zero Δ_Net via futures.

    We short futures to reduce a positive delta, long to increase a negative.
    """
    if delta_net > 0:
        return abs(delta_net), OrderSide.SELL
    else:
        return abs(delta_net), OrderSide.BUY


# ── Trio task ─────────────────────────────────────────────────────────────────

async def hedge_engine_task(channels: Channels, settings: Settings) -> None:
    """
    Poll PortfolioState at `hedge_poll_interval` Hz.  When |Δ_Net| exceeds
    the WW bandwidth, emit a HEDGE Order into order_send.

    The task drains the portfolio_state channel on each poll tick to ensure
    it always acts on the freshest state, not a stale queued one.
    """
    current_state: PortfolioState | None = None
    last_sent = float("-inf")

    logger.info(
        "HedgeEngine task started — φ=%.4f  λ=%.2f  poll=%.2fs  cooldown=%.1fs",
        settings.phi,
        settings.lambda_risk,
        settings.hedge_poll_interval,
        settings.hedge_cooldown,
    )

    while True:
        await trio.sleep(settings.hedge_poll_interval)

        # Drain the channel, keeping the latest state
        try:
            while True:
                current_state = channels.portfolio_state_recv.receive_nowait()
        except trio.WouldBlock:
            pass

        if current_state is None:
            continue

        state = current_state
        delta_net = state.delta_net
        gamma_net = state.gamma_net
        futures_price = state.futures_price

        if futures_price <= 0:
            continue

        w = whalley_wilmott_bandwidth(
            gamma_net=gamma_net,
            futures_price=futures_price,
            phi=settings.phi,
            lambda_risk=settings.lambda_risk,
        )

        abs_delta = abs(delta_net)

        logger.debug(
            "HedgeCheck: Δ=%.6f  Γ=%.8f  F=%.2f  w=%.6f  trigger=%s",
            delta_net, gamma_net, futures_price, w, abs_delta > w,
        )

        if abs_delta <= w:
            continue
        if abs_delta < settings.hedge_min_qty:
            logger.debug("HedgeCheck: |Δ|=%.6f below min_qty=%.4f, skipping",
                         abs_delta, settings.hedge_min_qty)
            continue
        now = trio.current_time()
        if now - last_sent < settings.hedge_cooldown:
            continue

        _, side = _hedge_qty_and_side(delta_net)
        order = Order(
            instrument=settings.futures_inst_id,
            side=side,
            quantity=abs_delta,
            order_type=OrderType.HEDGE,
            client_order_id=f"hdg-{uuid.uuid4().hex[:16]}",
        )

        logger.info(
            "Hedge trigger: |Δ|=%.6f > w=%.6f → %s %.6f base %s",
            abs_delta, w, side.value, abs_delta, settings.futures_inst_id,
        )

        try:
            channels.order_send.send_nowait(order)
            last_sent = now
        except trio.WouldBlock:
            logger.warning("order_ch full — hedge order dropped: %s", order)
