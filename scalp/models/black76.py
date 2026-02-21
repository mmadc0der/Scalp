"""
Black-76 option pricing and Greeks for futures-settled options.

All functions are pure numpy — no side effects, no I/O — so they run safely
in free-threaded (no-GIL) worker threads and vectorise across strike arrays.

Sign convention for Theta: dV/dt (rate of change with calendar time advancing,
T decreasing).  Theta is therefore negative for long option positions.

For crypto stablecoin-settled options r ≈ 0; the r parameter is kept for
correctness but defaults to zero.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


# ── Core helpers ──────────────────────────────────────────────────────────────

def _d1d2(
    F: np.ndarray | float,
    K: np.ndarray | float,
    sigma: np.ndarray | float,
    T: float,
    r: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute d₁ and d₂ for Black-76."""
    sigma_sqrt_T = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T
    return d1, d2


# ── Pricing ───────────────────────────────────────────────────────────────────

def call_price(
    F: np.ndarray | float,
    K: np.ndarray | float,
    sigma: np.ndarray | float,
    T: float,
    r: float = 0.0,
) -> np.ndarray | float:
    d1, d2 = _d1d2(F, K, sigma, T, r)
    return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))


def put_price(
    F: np.ndarray | float,
    K: np.ndarray | float,
    sigma: np.ndarray | float,
    T: float,
    r: float = 0.0,
) -> np.ndarray | float:
    d1, d2 = _d1d2(F, K, sigma, T, r)
    return np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


# ── Greeks ────────────────────────────────────────────────────────────────────

def delta_call(
    F: np.ndarray | float,
    K: np.ndarray | float,
    sigma: np.ndarray | float,
    T: float,
    r: float = 0.0,
) -> np.ndarray | float:
    d1, _ = _d1d2(F, K, sigma, T, r)
    return np.exp(-r * T) * norm.cdf(d1)


def delta_put(
    F: np.ndarray | float,
    K: np.ndarray | float,
    sigma: np.ndarray | float,
    T: float,
    r: float = 0.0,
) -> np.ndarray | float:
    d1, _ = _d1d2(F, K, sigma, T, r)
    return np.exp(-r * T) * (norm.cdf(d1) - 1.0)


def gamma(
    F: np.ndarray | float,
    K: np.ndarray | float,
    sigma: np.ndarray | float,
    T: float,
    r: float = 0.0,
) -> np.ndarray | float:
    d1, _ = _d1d2(F, K, sigma, T, r)
    return np.exp(-r * T) * norm.pdf(d1) / (F * sigma * np.sqrt(T))


def theta_call(
    F: np.ndarray | float,
    K: np.ndarray | float,
    sigma: np.ndarray | float,
    T: float,
    r: float = 0.0,
) -> np.ndarray | float:
    """
    Θ_C = dC/dt  (negative for long calls).

    Correct Black-76 formula:
      Θ_C = e^{-rT} [-F N'(d₁) σ / (2√T)  +  r F N(d₁)  -  r K N(d₂)]
    Note the + sign on the r terms (contrast with the sign error in some texts).
    """
    d1, d2 = _d1d2(F, K, sigma, T, r)
    disc = np.exp(-r * T)
    decay = F * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
    carry = r * F * norm.cdf(d1) - r * K * norm.cdf(d2)
    return disc * (-decay + carry)


def theta_put(
    F: np.ndarray | float,
    K: np.ndarray | float,
    sigma: np.ndarray | float,
    T: float,
    r: float = 0.0,
) -> np.ndarray | float:
    d1, d2 = _d1d2(F, K, sigma, T, r)
    disc = np.exp(-r * T)
    decay = F * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
    carry = -r * F * norm.cdf(-d1) + r * K * norm.cdf(-d2)
    return disc * (-decay + carry)


def vega(
    F: np.ndarray | float,
    K: np.ndarray | float,
    sigma: np.ndarray | float,
    T: float,
    r: float = 0.0,
) -> np.ndarray | float:
    """ν = dV/dσ — same for calls and puts."""
    d1, _ = _d1d2(F, K, sigma, T, r)
    return F * np.exp(-r * T) * norm.pdf(d1) * np.sqrt(T)


# ── Greeks bundle ─────────────────────────────────────────────────────────────

def greeks(
    F: float,
    K: float,
    sigma: float,
    T: float,
    is_call: bool,
    r: float = 0.0,
) -> dict[str, float]:
    """Return all four first-order Greeks as a dict."""
    d1, d2 = _d1d2(F, K, sigma, T, r)
    disc = float(np.exp(-r * T))
    npdf_d1 = float(norm.pdf(d1))
    sqrt_T = float(np.sqrt(T))

    g = disc * npdf_d1 / (F * sigma * sqrt_T)  # Gamma same for C and P

    if is_call:
        δ = disc * float(norm.cdf(d1))
        θ_decay = F * npdf_d1 * sigma / (2.0 * sqrt_T)
        θ_carry = r * F * float(norm.cdf(d1)) - r * K * float(norm.cdf(d2))
        θ = disc * (-θ_decay + θ_carry)
    else:
        δ = disc * (float(norm.cdf(d1)) - 1.0)
        θ_decay = F * npdf_d1 * sigma / (2.0 * sqrt_T)
        θ_carry = -r * F * float(norm.cdf(-d1)) + r * K * float(norm.cdf(-d2))
        θ = disc * (-θ_decay + θ_carry)

    ν = F * disc * npdf_d1 * sqrt_T

    return {"delta": δ, "gamma": g, "theta": θ, "vega": ν}


# ── Implied volatility solver (Newton-Raphson) ────────────────────────────────

def implied_vol(
    F: float,
    K: float,
    market_price: float,
    T: float,
    is_call: bool,
    r: float = 0.0,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """
    Newton-Raphson IV solver for a single option.

    Returns NaN if the solver fails to converge or if the market price is
    outside no-arbitrage bounds.
    """
    disc = np.exp(-r * T)
    intrinsic = disc * max(F - K, 0.0) if is_call else disc * max(K - F, 0.0)
    if market_price <= intrinsic:
        return float("nan")

    sigma = 0.5  # seed
    for _ in range(max_iter):
        price = call_price(F, K, sigma, T, r) if is_call else put_price(F, K, sigma, T, r)
        ν = float(vega(F, K, sigma, T, r))
        if ν < 1e-12:
            break
        diff = float(price) - market_price
        if abs(diff) < tol:
            return sigma
        sigma -= diff / ν
        sigma = max(sigma, 1e-6)

    return float("nan")
