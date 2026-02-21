from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).parent.parent / "demo.env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Exchange credentials ───────────────────────────────────────────────────
    exchange_name: str = "okx"
    okx_api_key: str = ""
    okx_api_secret: str = ""
    okx_passphrase: str = ""
    okx_simulated: bool = True

    # ── Instruments ───────────────────────────────────────────────────────────
    # OKX "underlying" for options, e.g. "BTC-USD" (inverse) or "BTC-USDT" (linear).
    # The demo account primarily has BTC-USD options; change to BTC-USDT when
    # linear options are available.
    uly: str = "BTC-USD"
    futures_inst_id: str = "BTC-USD-SWAP"   # perpetual used for delta hedging
    index_inst_id: str = "BTC-USD"          # index for 5-min candles (HAR-RV)

    # ── Signal parameters (hysteresis thresholds) ─────────────────────────────
    delta_entry: float = 0.05   # κ must exceed 1 ± δ_entry to enter a regime
    delta_exit: float = 0.02    # κ must fall within 1 ± δ_exit to exit to NEUTRAL

    # ── Whalley-Wilmott hedge bandwidth ───────────────────────────────────────
    phi: float = 0.0005         # proportional transaction cost (fees + slippage)
    lambda_risk: float = 1.0    # risk-aversion parameter λ

    # ── Hedging execution ─────────────────────────────────────────────────────
    hedge_poll_interval: float = 0.5   # seconds between Δ-check ticks
    hedge_min_qty: float = 0.001       # minimum futures hedge size (BTC/ETH)
    hedge_cooldown: float = 30.0       # seconds between hedge send attempts

    # ── Logging ───────────────────────────────────────────────────────────────
    log_file: str = "logs/execution.log"                 # empty = stdout only
    log_max_bytes: int = 10_485_760    # 10 MB per file
    log_backup_count: int = 5

    # ── HAR-RV model ──────────────────────────────────────────────────────────
    har_lookback_days: int = 365   # rolling OLS window in calendar days
    candles_per_day: int = 288     # 24 × 60 / 5  (5-minute candles)

    # ── Channel backpressure ──────────────────────────────────────────────────
    channel_maxsize: int = 64

    # ── OKX WebSocket endpoints ───────────────────────────────────────────────
    # Simulated-trading paper account endpoints
    ws_public_url: str = "wss://wspap.okx.com/ws/v5/public"
    ws_business_url: str = "wss://wspap.okx.com/ws/v5/business"
    ws_private_url: str = "wss://wspap.okx.com/ws/v5/private"

    # ── OKX REST flag ─────────────────────────────────────────────────────────
    @property
    def okx_flag(self) -> str:
        """OKX API flag: '1' = simulated, '0' = live."""
        return "1" if self.okx_simulated else "0"


settings = Settings()
