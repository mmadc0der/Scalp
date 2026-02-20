# Formal Specification: Linear Gamma Scalping & Parametric Volatility Modeling

## 1. Abstract
This document defines the mathematical architecture for a **Delta-Neutral Gamma Scalping** system operating on **Linear (Stablecoin-Margined)** crypto-derivatives (e.g., BTC/USDT, ETH/USDC). The core objective is to harvest the variance risk premium (VRP) by arbitraging the spread between **Market Implied Volatility ($IV$)** and **Forecasted Realized Volatility ($RV$)**.

The system utilizes two distinct parametric regression models:
1.  **SVI (Stochastic Volatility Inspired)** to fit the instantaneous market volatility surface.
2.  **HAR-RV (Heterogeneous Autoregressive)** to statistically forecast realized volatility.

## 2. Instrument Definition & Payoff Structure

The system targets **Linear Payoffs**, eliminating the convexity adjustments required for Inverse (Coin-Margined) contracts.

*   **Underlying ($S$)**: Spot price in USDT/USDC.
*   **Futures ($F$)**: Linear Perpetual Futures.
*   **Options**: Linear European Options (settled in Stablecoin).

### 2.1 Pricing Primitive (Black-76)
Since the underlying is a Future ($F$) rather than Spot, we use the **Black-76** model for pricing and Greeks.

$$
d_1 = \frac{\ln(F/K) + (\frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}
$$

**Call Option Value ($C$):**
$$ C = e^{-rT} [F N(d_1) - K N(d_2)] $$

**Put Option Value ($P$):**
$$ P = e^{-rT} [K N(-d_2) - F N(-d_1)] $$

**Key Greeks (Linear):**
*   $\Delta_C = e^{-rT} N(d_1)$
*   $\Gamma = \frac{e^{-rT} N'(d_1)}{F \sigma \sqrt{T}}$
*   $\Theta_C = -e^{-rT} \left[ \frac{F N'(d_1) \sigma}{2\sqrt{T}} + r F N(d_1) - r K N(d_2) \right]$
*   $\nu = F e^{-rT} N'(d_1) \sqrt{T}$

*Note: The Theta formula accounts for the drift of the futures price and time decay.*

---

## 3. Parametric Volatility Surface (The "Market" Fit)

To obtain an accurate Market IV ($\sigma_{mkt}$) for any strike $K$, we fit the **Raw SVI (Stochastic Volatility Inspired)** parameterization to the active orderbook. This ensures we trade against a smooth surface rather than noisy discrete points.

### 3.1 The Raw SVI Parametrization
Total variance $w(k) = \sigma_{BS}^2 T$ is modeled as a function of log-strike $k = \ln(K/F)$.

$$ w(k; a, b, \rho, m, \sigma_{svi}) = a + b \{ \rho(k - m) + \sqrt{(k - m)^2 + \sigma_{svi}^2} \} $$

**Parameters:**
*   $a \in \mathbb{R}$: Overall level of variance (vertical translation).
*   $b \ge 0$: Angle between asymptotes (vol of vol).
*   $\rho \in [-1, 1]$: Orientation (skew/correlation).
*   $m \in \mathbb{R}$: Horizontal shift (translation).
*   $\sigma_{svi} > 0$: Smoothing of the vertex (atm curvature).

### 3.2 Regression Task (Surface Fitting)
For a given expiry $T$, we minimize the sum of squared errors against mid-market prices converted to variances $w_i$:

$$ \min_{\{a, b, \rho, m, \sigma_{svi}\}} \sum_{i} (w_{model}(k_i) - w_{market, i})^2 $$

---

## 4. Parametric Volatility Forecasting (The "Alpha" Fit)

To determine if Options are "Cheap" or "Expensive", we do not rely on simple historical averages. We use the **HAR-RV (Heterogeneous Autoregressive model of Realized Volatility)**. This captures the "long memory" property of crypto volatility.

### 4.1 Realized Volatility Calculation
First, we compute the realized variance $RV^2$ over a day $t$ using high-frequency returns (e.g., 5-min candles):

$$ RV_t^2 = \sum_{j=1}^{M} r_{t,j}^2, \quad \text{where } r_{t,j} = \ln(F_{t,j}) - \ln(F_{t,j-1}) $$

### 4.2 The HAR-RV Regression Model
We model the future volatility $\sigma_{t+1}$ as a linear function of volatility components over different time horizons (Daily, Weekly, Monthly).

$$ \ln(RV_{t+1}) = \beta_0 + \beta_d \ln(RV_t) + \beta_w \ln(RV_{t-5, t}) + \beta_m \ln(RV_{t-22, t}) + \epsilon_{t+1} $$

Where:
*   $RV_t$: Realized Volatility of the previous day.
*   $RV_{t-5, t}$: Average RV over the last week.
*   $RV_{t-22, t}$: Average RV over the last month.

### 4.3 Signal Generation: Regime Switching with Hysteresis
We fit the $\beta$ parameters using OLS (Ordinary Least Squares) on a rolling lookback window (e.g., 365 days).

**Trading Signal ($\kappa$):**
$$ \kappa = \frac{\sigma_{SVI}(K_{ATM})}{\text{Forecasted } \widehat{RV}_{t+1}} $$

To prevent frequent turnover (whipsawing) near the neutral regime, we utilize a dual-threshold **Hysteresis** model:

1.  **Entry Threshold ($\delta_{entry}$)**: The required spread to transition from `NEUTRAL` to a Vol-active regime.
    *   `SHORT_VOL` if $\kappa > 1.0 + \delta_{entry}$
    *   `LONG_VOL` if $\kappa < 1.0 - \delta_{entry}$
2.  **Exit Threshold ($\delta_{exit}$)**: The required convergence to transition from an active regime back to `NEUTRAL`.
    *   `EXIT` to `NEUTRAL` if $|\kappa - 1.0| < \delta_{exit}$
3.  **Maintenance**: If $\delta_{exit} \le |\kappa - 1.0| \le \delta_{entry}$, the system maintains its current regime (Hysteresis Band).

Typically, $\delta_{entry} \approx 0.05$ and $\delta_{exit} \approx 0.02$.

---

## 5. Execution Logic: Delta-Neutral Scalping

Once a position is entered based on the $\kappa$ signal, the system engages the hedging loop to isolate the **Gamma** PnL.

### 5.1 Dynamic Hedging: The Whalley-Wilmott Band
Instead of a fixed delta threshold, the system utilizes the **Whalley-Wilmott (1997)** optimal hedging bandwidth to balance transaction costs against variance risk.

**Net Delta ($\Delta_{Net}$):**
$$ \Delta_{Net} = \sum (Q_{opt} \times \Delta_{opt}) + Q_{perp} $$

**Rebalance Trigger ($\Delta_{threshold}$):**
A hedge order is triggered if $|\Delta_{Net}| > w$, where the half-width $w$ is:
$$ w = \left( \frac{3 \cdot \phi \cdot \Gamma_{Net}^2 \cdot F}{2 \cdot \lambda} \right)^{1/3} $$

Where:
*   $\phi$: Proportional transaction cost (fees + slippage).
*   $\Gamma_{Net}$: Total portfolio Gamma.
*   $F$: Current Futures price.
*   $\lambda$: Risk aversion parameter.

### 5.2 Separation of Concerns: Trading vs. Hedging
The execution engine distinguishes between two distinct classes of orders:

1.  **Trading Orders (`TRADE`)**: Driven by the `desired_positions` vector from the Strategy Signal. Rebalanced using a cubic-root threshold based on ATM Implied Volatility to optimize option entry/exit.
2.  **Hedging Orders (`HEDGE`)**: Driven by the `hedge_target` to neutralize $\Delta_{Net}$. Rebalanced using the Whalley-Wilmott bandwidth $w$ based on portfolio Gamma.

### 5.3 Gamma Scalping PnL
By maintaining $\Delta \approx 0$, the PnL is driven by:
$$ PnL \approx \underbrace{\frac{1}{2} \Gamma F^2 (\text{Realized Return}^2)}_{\text{Gamma PnL}} - \underbrace{\Theta \cdot \Delta t}_{\text{Theta Cost}} $$

*   **Long Gamma**: Profitable if Realized Variance > Implied Variance paid.
*   **Short Gamma**: Profitable if Realized Variance < Implied Variance received.
