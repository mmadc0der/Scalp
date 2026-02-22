"""
Scalp engine entry point.

Wires all trio tasks together inside a single nursery and handles graceful
shutdown on SIGINT / SIGTERM (Ctrl-C).

Boot sequence
-------------
1. Load settings from demo.env
2. REST bootstrap:
   a. Fetch live option instruments (to warm up normalizer expiry tracking)
   b. Fetch historical candles and push to HAR-RV model (cold-start mitigation)
   c. Fetch current open positions to initialise PortfolioTracker
3. Open all MemoryChannels
4. Spawn all tasks inside a nursery

Tasks lifecycle
---------------
All tasks run indefinitely.  If any task raises an unhandled exception, trio
cancels the nursery and the process exits with a non-zero code.  In
production you would wrap individual tasks in retry/restart logic; for the
demo a crash-fast approach is appropriate.

Free-threaded note
------------------
Run with the no-GIL interpreter:
    python3.14t -m scalp
or:
    python3.14t -c "from scalp.main import main; import trio; trio.run(main)"
"""

from __future__ import annotations

import logging
import signal
import sys
from logging.handlers import RotatingFileHandler
from typing import Any, Callable

import trio

from scalp.channels import Channels, create_channels
from scalp.config import Settings, settings as _settings
from scalp.core.hedge import hedge_engine_task
from scalp.core.portfolio import portfolio_tracker_task
from scalp.core.signal import signal_engine_task
from scalp.exchange.normalizer import normalizer_task
from scalp.exchange.okx_rest import OKXRestClient
from scalp.exchange.okx_ws import ws_listener_task
from scalp.execution.order_manager import order_manager_task
from scalp.models.har_rv import har_rv_task
from scalp.models.svi import svi_fitter_task

logger = logging.getLogger(__name__)


# ── Logging setup ─────────────────────────────────────────────────────────────

def _configure_logging(settings: Settings) -> None:
    log_format = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    log_datefmt = "%Y-%m-%dT%H:%M:%S"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=log_datefmt,
        stream=sys.stdout,
    )
    if settings.log_file:
        file_handler = RotatingFileHandler(
            filename=settings.log_file,
            maxBytes=settings.log_max_bytes,
            backupCount=settings.log_backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=log_datefmt))
        logging.getLogger().addHandler(file_handler)

    # Quieten noisy libraries
    logging.getLogger("trio_websocket").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)


# ── REST bootstrap ────────────────────────────────────────────────────────────

async def _retry_rest_bootstrap_call(
    *,
    op_name: str,
    func: Callable[[], Any],
    attempts: int = 4,
    base_delay_s: float = 1.0,
    max_delay_s: float = 8.0,
) -> Any | None:
    """Run a blocking REST call with bounded exponential-backoff retries."""
    last_exc: Exception | None = None
    for attempt in range(1, max(attempts, 1) + 1):
        try:
            return await trio.to_thread.run_sync(func, abandon_on_cancel=False)
        except Exception as exc:  # network/SDK transient failures
            last_exc = exc
            if attempt >= attempts:
                break
            delay = min(base_delay_s * (2 ** (attempt - 1)), max_delay_s)
            logger.warning(
                "Bootstrap: %s failed (%s), retry %d/%d in %.1fs",
                op_name,
                exc,
                attempt,
                attempts,
                delay,
            )
            await trio.sleep(delay)

    logger.error(
        "Bootstrap: %s failed after %d attempt(s): %s",
        op_name,
        max(attempts, 1),
        last_exc,
    )
    return None

async def _bootstrap(settings: Settings) -> tuple[list[float], list[dict[str, Any]]]:
    """
    Pre-flight REST calls:
      • Confirm API credentials work (fetch balance)
      • Fetch historical candles for HAR-RV warm-up
    Runs before the nursery starts so failures are visible immediately.

    Returns:
      - chronological close list for HAR-RV bootstrap
      - current open positions snapshot
    """
    rest = OKXRestClient(settings)
    bootstrap_closes: list[float] = []

    logger.info("Bootstrap: connecting to OKX (simulated=%s)", settings.okx_simulated)

    # Credentials check
    balances = await _retry_rest_bootstrap_call(
        op_name="get account balance",
        func=rest.get_balance,
    ) or []
    if balances:
        logger.info("Bootstrap: account balance OK (%d entries)", len(balances))
    else:
        logger.warning("Bootstrap: empty balance response — check credentials")

    # HAR-RV warm-up from historical candles.
    # We need enough 5m candles to build >=23 daily RV observations.
    logger.info("Bootstrap: fetching historical 5m candles for HAR-RV warm-up")
    target_closes = settings.candles_per_day * 25  # ~25 days for margin over 23-day minimum
    before = ""
    max_pages = 40

    for _ in range(max_pages):
        page = await _retry_rest_bootstrap_call(
            op_name=f"get historical candles before={before or 'latest'}",
            func=lambda b=before: rest.get_historical_candles(
                settings.index_inst_id,
                bar="5m",
                limit=300,
                before=b,
            ),
        )
        if page is None:
            logger.warning("Bootstrap: stopping historical pagination after repeated failures")
            break
        if not page:
            break

        # OKX returns newest-first; reverse so close series stays chronological.
        for row in reversed(page):
            try:
                bootstrap_closes.append(float(row[4]))
            except (IndexError, TypeError, ValueError):
                continue

        try:
            oldest_ts = min(int(row[0]) for row in page if len(row) > 0)
        except Exception:
            break

        next_before = str(oldest_ts)
        if not next_before or next_before == before:
            break
        before = next_before

        if len(bootstrap_closes) >= target_closes:
            break

    if bootstrap_closes:
        logger.info(
            "Bootstrap: harvested %d closes (%.1f days) for HAR-RV seed",
            len(bootstrap_closes),
            len(bootstrap_closes) / settings.candles_per_day,
        )
    else:
        logger.warning("Bootstrap: no historical candle data returned")

    # Fetch current positions so PortfolioTracker can seed itself
    positions = await _retry_rest_bootstrap_call(
        op_name="get open positions",
        func=rest.get_positions,
    ) or []
    if positions:
        logger.info("Bootstrap: %d open position(s) found", len(positions))
        for pos in positions:
            logger.debug("  position: %s qty=%s", pos.get("instId"), pos.get("pos"))
    else:
        logger.info("Bootstrap: no open positions")
    return bootstrap_closes, positions


# ── Graceful shutdown ─────────────────────────────────────────────────────────

def _install_signal_handlers(cancel_scope: trio.CancelScope) -> None:
    """Install SIGINT / SIGTERM handlers that cancel the root scope."""
    def _handle(signum: int, _frame: object) -> None:
        logger.info("Signal %s received — initiating graceful shutdown", signum)
        cancel_scope.cancel()

    signal.signal(signal.SIGINT, _handle)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle)


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    s = _settings
    _configure_logging(s)

    logger.info("=" * 60)
    logger.info("Scalp Engine starting — uly=%s  simulated=%s", s.uly, s.okx_simulated)
    logger.info("=" * 60)

    bootstrap_closes, bootstrap_positions = await _bootstrap(s)

    channels: Channels = create_channels(s)

    with trio.CancelScope() as root_scope:
        _install_signal_handlers(root_scope)

        async with trio.open_nursery() as nursery:
            # Data ingestion
            nursery.start_soon(ws_listener_task, channels, s)
            nursery.start_soon(normalizer_task, channels, s)

            # Analytics
            nursery.start_soon(svi_fitter_task, channels, s)
            nursery.start_soon(har_rv_task, channels, s, bootstrap_closes)

            # Strategy
            nursery.start_soon(signal_engine_task, channels, s)
            nursery.start_soon(portfolio_tracker_task, channels, s, bootstrap_positions)
            nursery.start_soon(hedge_engine_task, channels, s)

            # Execution
            nursery.start_soon(order_manager_task, channels, s)

            logger.info("All tasks started — engine is live")

    logger.info("Scalp Engine stopped cleanly")


def run() -> None:
    """Entry point for `scalp` console script."""
    trio.run(main)


if __name__ == "__main__":
    run()
