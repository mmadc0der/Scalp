"""
HAR-RV (Heterogeneous Autoregressive model of Realized Volatility).

Reference: Corsi (2009) "A simple approximate long-memory model of realized
volatility", Journal of Financial Econometrics.

Architecture
------------
* 5-min candles arrive from the normalizer via `candle_recv`.
* `HARRVModel.push_candle()` accumulates intra-day log-returns.
* When a full trading day's worth of candles has arrived (candles_per_day),
  the daily RV is computed and appended to the rolling window.
* The HAR-RV OLS fit and forecast run in a thread worker (numpy releases GIL).
* The forecast is emitted into `rv_forecast_send`.

Cold-start
----------
On first run the rolling window is empty. We bootstrap with a 30-day window
once we have 23+ days of RV data (minimum required: 1 + 22 days for the
monthly component).  `bootstrap(daily_rv_series)` can be called externally
to pre-populate from REST historical data before the live loop starts.
"""

from __future__ import annotations

import logging
import time
from collections import deque

import numpy as np
import trio

from scalp.channels import Channels
from scalp.config import Settings
from scalp.debug_log import debug_log
from scalp.schema import Candle, RVForecast

logger = logging.getLogger(__name__)
_HAR_CANDLE_COUNT = 0

# Number of intra-day observations required for a valid daily RV
_MIN_INTRADAY = 50   # tolerate gaps; don't require a full 288 candles


class HARRVModel:
    """
    Stateful HAR-RV model.

    Thread-safe read of `last_forecast` from the trio event loop while the
    thread worker updates it is safe because CPython (and the no-GIL 3.14t
    build) guarantees atomic assignment to Python object references.
    """

    def __init__(
        self,
        lookback_days: int = 365,
        candles_per_day: int = 288,
    ) -> None:
        self.lookback_days = lookback_days
        self.candles_per_day = candles_per_day

        # Intra-day state
        self._intraday_returns: list[float] = []
        self._last_close: float | None = None

        # Rolling daily RV window (annualised volatility, not variance)
        # Extra 22-day buffer so the monthly component is always available
        self._daily_rv: deque[float] = deque(maxlen=lookback_days + 23)

        self.last_forecast: RVForecast | None = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def bootstrap(self, daily_rv: list[float] | np.ndarray) -> None:
        """Pre-populate the rolling window with historical daily RV values."""
        self._daily_rv.extend(daily_rv)
        logger.info("HAR-RV bootstrapped with %d daily RV observations", len(daily_rv))

    def push_candle(self, close_price: float) -> RVForecast | None:
        """
        Record a 5-min candle close price.

        Returns an RVForecast when a new day completes and the OLS fit
        succeeds, otherwise None.
        """
        if self._last_close is not None:
            r = np.log(close_price / self._last_close)
            self._intraday_returns.append(r)
        self._last_close = close_price

        if len(self._intraday_returns) >= self.candles_per_day:
            rv_daily = self._flush_day()
            self._daily_rv.append(rv_daily)
            logger.debug("New daily RV: %.4f (window=%d)", rv_daily, len(self._daily_rv))

            if len(self._daily_rv) >= 23:
                forecast = self._fit_and_forecast()
                self.last_forecast = forecast
                return forecast

        return None

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _flush_day(self) -> float:
        """Compute annualised daily RV from accumulated intra-day returns."""
        ret = np.array(self._intraday_returns[:self.candles_per_day])
        self._intraday_returns = self._intraday_returns[self.candles_per_day:]

        # Realised variance: Σ r²_j, then annualise by ×252
        rv_var = float(np.sum(ret**2))
        rv_vol = np.sqrt(rv_var * 252.0)  # annualised volatility
        return float(rv_vol)

    def _fit_and_forecast(self) -> RVForecast:
        """
        Fit HAR-RV via OLS on the rolling lookback window and forecast t+1.

        Model (in log space):
            ln(RV_{t+1}) = β₀ + β_d ln(RV_t) + β_w ln(RV̄_{t-5,t})
                                + β_m ln(RV̄_{t-22,t}) + ε_{t+1}
        """
        rv = np.array(self._daily_rv)
        log_rv = np.log(np.maximum(rv, 1e-10))
        n = len(log_rv)

        # Build design matrix: rows are [1, rv_d, rv_w, rv_m]
        rows: list[list[float]] = []
        y_vals: list[float] = []

        for t in range(22, n):
            rv_d = log_rv[t - 1]
            rv_w = float(np.mean(log_rv[max(0, t - 5):t]))
            rv_m = float(np.mean(log_rv[max(0, t - 22):t]))
            rows.append([1.0, rv_d, rv_w, rv_m])
            y_vals.append(log_rv[t])

        # Apply rolling lookback window
        X = np.array(rows[-self.lookback_days:])
        y = np.array(y_vals[-self.lookback_days:])

        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

        # One-step-ahead forecast using the most recent observations
        rv_d = log_rv[-1]
        rv_w = float(np.mean(log_rv[-5:]))
        rv_m = float(np.mean(log_rv[-22:]))
        x_new = np.array([1.0, rv_d, rv_w, rv_m])
        log_rv_hat = float(x_new @ beta)
        sigma_hat = float(np.exp(log_rv_hat))

        return RVForecast(
            sigma_hat=sigma_hat,
            beta_vector=beta,
            fit_date=time.time(),
            n_obs=len(y),
        )


# ── Trio task ──────────────────────────────────────────────────────────────────

async def har_rv_task(
    channels: Channels,
    settings: Settings,
    bootstrap_closes: list[float] | None = None,
) -> None:
    """
    Trio task: consumes 5-min Candle events, runs the HAR-RV fit in a thread
    worker when a new day completes, and emits RVForecast into rv_forecast_send.
    """
    model = HARRVModel(
        lookback_days=settings.har_lookback_days,
        candles_per_day=settings.candles_per_day,
    )
    logger.info("HAR-RV task started (lookback=%d days)", settings.har_lookback_days)

    if bootstrap_closes:
        def _bootstrap_seed() -> RVForecast | None:
            last: RVForecast | None = None
            for px in bootstrap_closes:
                out = model.push_candle(px)
                if out is not None:
                    last = out
            return last

        seeded = await trio.to_thread.run_sync(_bootstrap_seed, abandon_on_cancel=False)
        # region agent log
        debug_log(
            hypothesis_id="H7",
            location="scalp/models/har_rv.py:har_rv_task",
            message="har_bootstrap_seed_applied",
            data={
                "bootstrap_closes": len(bootstrap_closes),
                "daily_window": len(model._daily_rv),
                "has_forecast": seeded is not None,
            },
        )
        # endregion
        if seeded is not None:
            logger.info(
                "HAR-RV bootstrap forecast: σ̂=%.4f  n_obs=%d",
                seeded.sigma_hat, seeded.n_obs,
            )
            await channels.rv_forecast_send.send(seeded)

    async for candle in channels.candle_recv:
        global _HAR_CANDLE_COUNT
        _HAR_CANDLE_COUNT += 1
        if _HAR_CANDLE_COUNT <= 3 or _HAR_CANDLE_COUNT % 100 == 0:
            # region agent log
            debug_log(
                hypothesis_id="H5",
                location="scalp/models/har_rv.py:har_rv_task",
                message="har_received_candle",
                data={"count": _HAR_CANDLE_COUNT, "close": candle.close},
            )
            # endregion
        close = candle.close

        forecast = await trio.to_thread.run_sync(
            lambda: model.push_candle(close),
            abandon_on_cancel=True,
        )
        if forecast is None and (_HAR_CANDLE_COUNT <= 3 or _HAR_CANDLE_COUNT % 200 == 0):
            # region agent log
            debug_log(
                hypothesis_id="H7",
                location="scalp/models/har_rv.py:har_rv_task",
                message="har_no_forecast_yet",
                data={
                    "candles_seen": _HAR_CANDLE_COUNT,
                    "intraday_returns": len(model._intraday_returns),
                    "daily_window": len(model._daily_rv),
                    "daily_needed": 23,
                    "candles_per_day": model.candles_per_day,
                },
            )
            # endregion

        if forecast is not None:
            logger.info(
                "HAR-RV forecast: σ̂=%.4f  n_obs=%d  betas=%s",
                forecast.sigma_hat, forecast.n_obs,
                np.round(forecast.beta_vector, 4),
            )
            # region agent log
            debug_log(
                hypothesis_id="H5",
                location="scalp/models/har_rv.py:har_rv_task",
                message="har_emitted_forecast",
                data={"sigma_hat": forecast.sigma_hat, "n_obs": forecast.n_obs},
            )
            # endregion
            await channels.rv_forecast_send.send(forecast)
