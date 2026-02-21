"""
Raw SVI (Stochastic Volatility Inspired) volatility surface fitting.

Reference: Gatheral & Jacquier (2014) "Arbitrage-free SVI volatility surfaces".

The fitting minimises squared errors on total variance w(k) = σ²_BS × T,
where k = ln(K/F) is the log-moneyness, subject to the Gatheral-Jacquier
butterfly no-arbitrage constraint at the vertex:

    g(k) = a + b × σ_svi × √(1 - ρ²) ≥ 0

An optional warm-start cache keyed by expiry_str ensures the optimiser
converges quickly on successive updates of the same slice.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.optimize import Bounds, minimize

import trio

from scalp.channels import Channels
from scalp.config import Settings
from scalp.debug_log import debug_log
from scalp.schema import OBSnapshot, SVIParams, VolSurface

logger = logging.getLogger(__name__)


# ── SVI math ──────────────────────────────────────────────────────────────────

def svi_total_variance(k: np.ndarray, p: SVIParams) -> np.ndarray:
    """w(k; a, b, ρ, m, σ) = a + b { ρ(k−m) + √((k−m)² + σ²) }"""
    x = k - p.m
    return p.a + p.b * (p.rho * x + np.sqrt(x**2 + p.sigma_svi**2))


def svi_implied_vol(log_strike: float, p: SVIParams, T: float) -> float:
    """Convert SVI total variance to annualised implied volatility."""
    w = float(svi_total_variance(np.array([log_strike]), p)[0])
    w = max(w, 1e-10)
    return np.sqrt(w / T)


# ── Fitting ───────────────────────────────────────────────────────────────────

class _FitResult(NamedTuple):
    params: SVIParams
    residual: float
    success: bool


def _fit_slice(
    log_strikes: np.ndarray,
    total_variances: np.ndarray,
    x0: np.ndarray | None = None,
) -> _FitResult:
    """
    Fit raw SVI to one expiry slice via SLSQP.

    Parameters
    ----------
    log_strikes:      k = ln(K/F), shape (N,)
    total_variances:  w_i = mid_iv² × T, shape (N,)
    x0:               warm-start [a, b, ρ, m, σ_svi]; uses ATM-level seed if None
    """
    if len(log_strikes) < 5:
        raise ValueError("Need ≥ 5 strikes per expiry for SVI fitting")

    w_atm = float(np.interp(0.0, log_strikes, total_variances))

    if x0 is None:
        x0 = np.array([w_atm * 0.9, 0.1, -0.3, 0.0, 0.2])

    def objective(x: np.ndarray) -> float:
        p = SVIParams(*x)
        return float(np.sum((svi_total_variance(log_strikes, p) - total_variances) ** 2))

    # Gatheral-Jacquier vertex constraint: a + b σ √(1-ρ²) ≥ 0
    def vertex_nonneg(x: np.ndarray) -> float:
        a, b, rho, _, sigma_svi = x
        return a + b * sigma_svi * np.sqrt(max(1.0 - rho**2, 0.0))

    bounds = Bounds(
        lb=[-1.0, 0.0, -0.999, -3.0, 1e-4],
        ub=[2.0, 5.0,  0.999,  3.0, 2.0],
    )
    constraints = [{"type": "ineq", "fun": vertex_nonneg}]

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-10},
    )

    params = SVIParams(*result.x)
    return _FitResult(params=params, residual=float(result.fun), success=result.success)


# ── Warm-start cache ──────────────────────────────────────────────────────────

@dataclass
class _SliceCache:
    params: SVIParams
    last_updated: float = 0.0


class SVISurfaceFitter:
    """
    Stateful SVI fitter that maintains a warm-start cache per expiry.

    Call `fit(snapshots)` with a list of OBSnapshot objects (one per expiry)
    to get back a VolSurface.  The synchronous `fit` method is safe to run in
    a trio thread worker — it releases the GIL during the scipy optimisation.
    """

    def __init__(self) -> None:
        self._cache: dict[str, _SliceCache] = {}

    def fit(
        self,
        snapshots: list[OBSnapshot],
        underlying: str,
        futures_price: float,
    ) -> VolSurface | None:
        """
        Fit SVI to all provided expiry snapshots.

        Returns a VolSurface, or None if no expiry could be successfully fit.
        """
        fitted: dict[str, SVIParams] = {}
        times: dict[str, float] = {}

        for snap in snapshots:
            if len(snap.log_strikes) < 5:
                continue

            x0 = None
            cached = self._cache.get(snap.expiry_str)
            if cached is not None:
                x0 = np.array([
                    cached.params.a, cached.params.b, cached.params.rho,
                    cached.params.m, cached.params.sigma_svi,
                ])

            try:
                result = _fit_slice(snap.log_strikes, snap.total_variances, x0)
            except Exception as exc:
                logger.warning("SVI fit failed for expiry %s: %s", snap.expiry_str, exc)
                continue

            if not result.success:
                logger.debug(
                    "SVI optimiser did not converge for %s (residual=%.2e)",
                    snap.expiry_str, result.residual,
                )

            self._cache[snap.expiry_str] = _SliceCache(
                params=result.params, last_updated=time.time()
            )
            fitted[snap.expiry_str] = result.params
            times[snap.expiry_str] = snap.time_to_expiry

        if not fitted:
            return None

        # ATM IV from the nearest expiry
        nearest_exp = min(times, key=lambda e: times[e])
        atm_iv = svi_implied_vol(0.0, fitted[nearest_exp], times[nearest_exp])

        return VolSurface(
            underlying=underlying,
            futures_price=futures_price,
            expiries=fitted,
            expiry_times=times,
            atm_iv=atm_iv,
        )


# ── Trio task ─────────────────────────────────────────────────────────────────

async def svi_fitter_task(channels: Channels, settings: Settings) -> None:
    """
    Trio task: consumes OBSnapshot batches, fits SVI in a thread worker,
    and publishes VolSurface to both SignalEngine and PortfolioTracker.

    Snapshots are grouped by (underlying, futures_price) — the normalizer
    is expected to send one OBSnapshot per expiry per opt-summary update.
    We collect them within a short aggregation window then fit all at once.
    """
    fitter = SVISurfaceFitter()
    pending: dict[str, OBSnapshot] = {}  # expiry_str → latest snapshot
    underlying = settings.uly
    futures_price = 0.0

    # Aggregation window: collect snapshots for up to WINDOW_S seconds,
    # then trigger a fit with whatever has arrived.
    WINDOW_S = 0.5

    async def _collect_window() -> None:
        nonlocal futures_price
        with trio.move_on_after(WINDOW_S):
            async for snap in channels.ob_snapshot_recv:
                pending[snap.expiry_str] = snap
                futures_price = snap.forward_price  # take latest

    logger.info("SVIFitter task started, underlying=%s", underlying)

    while True:
        await _collect_window()

        if not pending:
            await trio.sleep(0.1)
            continue

        snapshots = list(pending.values())
        pending.clear()
        # region agent log
        debug_log(
            hypothesis_id="H4",
            location="scalp/models/svi.py:svi_fitter_task",
            message="svi_fit_cycle_start",
            data={"snapshots_count": len(snapshots)},
        )
        # endregion

        fp = futures_price  # capture for closure

        surface = await trio.to_thread.run_sync(
            lambda: fitter.fit(snapshots, underlying, fp),
            abandon_on_cancel=True,
        )

        if surface is None:
            logger.warning("SVIFitter: no expiries fit in this cycle")
            # region agent log
            debug_log(
                hypothesis_id="H4",
                location="scalp/models/svi.py:svi_fitter_task",
                message="svi_fit_cycle_no_surface",
                data={"snapshots_count": len(snapshots)},
            )
            # endregion
            continue

        logger.debug(
            "VolSurface fitted: expiries=%s  atm_iv=%.4f",
            list(surface.expiries.keys()), surface.atm_iv,
        )
        # region agent log
        debug_log(
            hypothesis_id="H4",
            location="scalp/models/svi.py:svi_fitter_task",
            message="svi_surface_emitted",
            data={"expiries_count": len(surface.expiries), "atm_iv": surface.atm_iv},
        )
        # endregion

        await channels.surface_signal_send.send(surface)
        await channels.surface_portfolio_send.send(surface)
