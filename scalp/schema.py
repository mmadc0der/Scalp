"""
Canonical message types flowing through trio MemoryChannels.

All dataclasses are treated as immutable once constructed — do NOT mutate
them after sending into a channel.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


# ── Raw WebSocket event ────────────────────────────────────────────────────────

@dataclass
class RawWSEvent:
    """Un-parsed OKX WebSocket push message, routed by channel name."""
    channel: str
    arg: dict[str, str]
    data: list[dict[str, Any]]
    timestamp: float = field(default_factory=time.time)


# ── Market data ────────────────────────────────────────────────────────────────

@dataclass
class OBSnapshot:
    """
    Per-expiry volatility snapshot ready for SVI fitting.

    Arrays are parallel: log_strikes[i] / total_variances[i] are a pair.
    Total variance: w = IV² × T  (where IV is mid implied vol, annualised).
    """
    underlying: str
    expiry_str: str            # OKX format, e.g. "250328"
    forward_price: float       # OKX fwdPx for this expiry
    log_strikes: np.ndarray    # k = ln(K/F), shape (N,)
    total_variances: np.ndarray  # w = mid_iv² × T, shape (N,)
    time_to_expiry: float      # years, T
    timestamp: float = field(default_factory=time.time)


@dataclass
class Candle:
    """5-minute OHLCV bar for an index instrument."""
    instrument: str
    open_time: float           # unix ms (bar open)
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class FillEvent:
    """Order fill received from private WS or REST poll."""
    order_id: str
    client_order_id: str
    instrument: str
    side: str                  # "buy" | "sell"
    quantity: float            # absolute value, always positive
    price: float
    fee: float
    order_type: str            # "TRADE" | "HEDGE"
    timestamp: float = field(default_factory=time.time)


# ── Volatility surface ─────────────────────────────────────────────────────────

@dataclass
class SVIParams:
    """Raw SVI parameter set for one expiry slice."""
    a: float
    b: float
    rho: float
    m: float
    sigma_svi: float


@dataclass
class VolSurface:
    """
    Full fitted volatility surface: one SVIParams per active expiry.

    atm_iv is the ATM implied vol from the nearest expiry, used directly
    in the κ signal computation.
    """
    underlying: str
    futures_price: float
    expiries: dict[str, SVIParams]     # expiry_str → params
    expiry_times: dict[str, float]     # expiry_str → T (years)
    atm_iv: float                      # nearest expiry ATM IV (annualised)
    timestamp: float = field(default_factory=time.time)

    def iv_at(self, expiry_str: str, log_strike: float) -> float:
        """Compute IV from the fitted SVI slice for (expiry, log-strike)."""
        from scalp.models.svi import svi_implied_vol  # avoid circular at module level
        params = self.expiries[expiry_str]
        T = self.expiry_times[expiry_str]
        return svi_implied_vol(log_strike, params, T)


# ── Volatility forecast ────────────────────────────────────────────────────────

@dataclass
class RVForecast:
    """HAR-RV one-step-ahead forecast."""
    sigma_hat: float           # annualised forecasted RV (volatility, not variance)
    beta_vector: np.ndarray    # OLS coefficients [β₀, β_d, β_w, β_m]
    fit_date: float            # unix timestamp of the fit
    n_obs: int                 # number of observations used in OLS


# ── Signal / regime ────────────────────────────────────────────────────────────

class Regime(Enum):
    NEUTRAL = "NEUTRAL"
    SHORT_VOL = "SHORT_VOL"
    LONG_VOL = "LONG_VOL"


@dataclass
class RegimeEvent:
    """
    Emitted by SignalEngine only on state transitions.

    kappa = σ_SVI(K_ATM) / RV_forecast.  Values > 1 mean IV expensive.

    nearest_expiry and atm_strike are included so the OrderManager can
    construct option instruments without re-reading the VolSurface.
    """
    regime: Regime
    kappa: float
    prev_regime: Regime
    nearest_expiry: str = ""     # OKX expiry string, e.g. "250328"
    atm_strike: float = 0.0      # nearest round strike to ATM
    timestamp: float = field(default_factory=time.time)


# ── Orders ─────────────────────────────────────────────────────────────────────

class OrderType(Enum):
    TRADE = "TRADE"
    HEDGE = "HEDGE"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


@dataclass
class Order:
    """Unified order record for both TRADE and HEDGE orders."""
    instrument: str
    side: OrderSide
    quantity: float            # positive, absolute size in base currency
    order_type: OrderType
    order_id: str = ""         # filled by OrderManager after REST placement
    client_order_id: str = ""
    status: OrderStatus = OrderStatus.PENDING
    timestamp: float = field(default_factory=time.time)


# ── Portfolio state ────────────────────────────────────────────────────────────

@dataclass
class Leg:
    """A single live position (option or futures)."""
    instrument: str
    quantity: float            # signed: positive = long, negative = short
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0


@dataclass
class PortfolioState:
    """
    Aggregate portfolio Greeks, updated on fills and vol surface refreshes.
    Emitted by PortfolioTracker into portfolio_state_ch for HedgeEngine.
    """
    legs: dict[str, Leg] = field(default_factory=dict)  # instId → Leg
    delta_net: float = 0.0
    gamma_net: float = 0.0
    theta_net: float = 0.0
    vega_net: float = 0.0
    futures_price: float = 0.0
    timestamp: float = field(default_factory=time.time)
