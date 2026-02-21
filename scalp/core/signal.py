"""
SignalEngine — κ computation and hysteresis regime FSM.

κ = σ_SVI(K_ATM, nearest expiry) / RV̂_{t+1}

Hysteresis transitions (per MATH_SPEC §4.3):

    NEUTRAL  → SHORT_VOL  if κ > 1 + δ_entry
    NEUTRAL  → LONG_VOL   if κ < 1 - δ_entry
    SHORT_VOL → NEUTRAL   if |κ - 1| < δ_exit
    LONG_VOL  → NEUTRAL   if |κ - 1| < δ_exit
    (maintenance band: δ_exit ≤ |κ - 1| ≤ δ_entry → stay in current regime)

A RegimeEvent is emitted ONLY on a state transition, not on every surface
update.  This suppresses churning orders at the boundary.
"""

from __future__ import annotations

import logging
import math

import trio

from scalp.channels import Channels
from scalp.config import Settings
from scalp.debug_log import debug_log
from scalp.schema import Regime, RegimeEvent, RVForecast, VolSurface

logger = logging.getLogger(__name__)


def _compute_kappa(surface: VolSurface, rv_forecast: RVForecast) -> float:
    """κ = ATM implied vol / forecasted realised vol."""
    if rv_forecast.sigma_hat <= 0:
        return 1.0
    return surface.atm_iv / rv_forecast.sigma_hat


def _nearest_expiry_and_atm(surface: VolSurface) -> tuple[str, float]:
    """
    Return (expiry_str, atm_strike) for the nearest expiry.

    atm_strike is the rounded futures price, which is the centre of the
    ATM straddle.  Option strike granularity differs per exchange listing;
    we round to the nearest 100 USD for BTC (adjust for other underlyings).
    """
    if not surface.expiry_times:
        return "", 0.0
    nearest = min(surface.expiry_times, key=lambda e: surface.expiry_times[e])
    # Round to nearest 100 for BTC-style underlyings
    strike = round(surface.futures_price / 100) * 100
    return nearest, float(strike)


def _transition(
    kappa: float,
    current: Regime,
    delta_entry: float,
    delta_exit: float,
) -> Regime:
    """
    Apply hysteresis logic and return the next regime.
    """
    spread = abs(kappa - 1.0)

    if current == Regime.NEUTRAL:
        if kappa > 1.0 + delta_entry:
            return Regime.SHORT_VOL
        if kappa < 1.0 - delta_entry:
            return Regime.LONG_VOL
        return Regime.NEUTRAL

    # Active regime: only exit on convergence
    if spread < delta_exit:
        return Regime.NEUTRAL

    # Hysteresis band or still beyond entry threshold — hold
    return current


# ── Trio task ─────────────────────────────────────────────────────────────────

async def signal_engine_task(channels: Channels, settings: Settings) -> None:
    """
    Consume VolSurface and RVForecast events, maintain the regime FSM, and
    emit RegimeEvent on transitions.

    Both inputs are optional; the signal is only computed when both are
    available.  The HAR-RV forecast is updated rarely (once per day), while
    the vol surface updates frequently — so we cache the last RV forecast
    and recompute κ on every surface update.
    """
    current_regime = Regime.NEUTRAL
    last_rv: RVForecast | None = None
    last_kappa: float = 1.0
    surface_count = 0

    logger.info(
        "SignalEngine task started — δ_entry=%.3f  δ_exit=%.3f",
        settings.delta_entry, settings.delta_exit,
    )

    async def _consume_rv() -> None:
        """Background task that caches the latest RV forecast."""
        nonlocal last_rv
        async for forecast in channels.rv_forecast_recv:
            last_rv = forecast
            logger.info(
                "RV forecast updated: σ̂=%.4f  n_obs=%d",
                forecast.sigma_hat, forecast.n_obs,
            )

    async def _consume_surface() -> None:
        """Main loop: recompute κ and fire FSM on every surface update."""
        nonlocal current_regime, last_kappa, surface_count

        async for surface in channels.surface_signal_recv:
            surface_count += 1
            if last_rv is None:
                logger.debug("SignalEngine: waiting for first RV forecast")
                if surface_count <= 3 or surface_count % 200 == 0:
                    # region agent log
                    debug_log(
                        hypothesis_id="H6",
                        location="scalp/core/signal.py:_consume_surface",
                        message="signal_waiting_for_rv_forecast",
                        data={"surface_count": surface_count, "atm_iv": surface.atm_iv},
                    )
                    # endregion
                continue

            kappa = _compute_kappa(surface, last_rv)
            last_kappa = kappa

            next_regime = _transition(
                kappa, current_regime,
                settings.delta_entry, settings.delta_exit,
            )

            if next_regime != current_regime:
                nearest_exp, atm_strike = _nearest_expiry_and_atm(surface)
                ev = RegimeEvent(
                    regime=next_regime,
                    kappa=kappa,
                    prev_regime=current_regime,
                    nearest_expiry=nearest_exp,
                    atm_strike=atm_strike,
                )
                logger.info(
                    "Regime transition: %s → %s  (κ=%.4f  atm_iv=%.4f  rv̂=%.4f)",
                    current_regime.value, next_regime.value,
                    kappa, surface.atm_iv, last_rv.sigma_hat,
                )
                current_regime = next_regime
                # region agent log
                debug_log(
                    hypothesis_id="H8",
                    location="scalp/core/signal.py:_consume_surface",
                    message="signal_regime_transition",
                    data={
                        "from": ev.prev_regime.value,
                        "to": ev.regime.value,
                        "kappa": ev.kappa,
                        "nearest_expiry": ev.nearest_expiry,
                        "atm_strike": ev.atm_strike,
                    },
                )
                # endregion
                await channels.signal_send.send(ev)
            else:
                logger.debug(
                    "Regime held: %s  κ=%.4f  atm_iv=%.4f",
                    current_regime.value, kappa, surface.atm_iv,
                )

    async with trio.open_nursery() as nursery:
        nursery.start_soon(_consume_rv)
        nursery.start_soon(_consume_surface)
