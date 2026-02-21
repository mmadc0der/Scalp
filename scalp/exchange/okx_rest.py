"""
OKX REST API wrapper.

The python-okx SDK is synchronous.  All methods here are plain sync functions
designed to be called via `trio.to_thread.run_sync(...)`.  They MUST NOT be
called directly from an async context without the thread wrapper.

Usage
-----
    rest = OKXRestClient(settings)

    # From within a trio task:
    instruments = await trio.to_thread.run_sync(
        rest.get_option_instruments, abandon_on_cancel=True
    )
    result = await trio.to_thread.run_sync(
        lambda: rest.place_order(inst_id, side, qty, order_type), abandon_on_cancel=True
    )
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import okx.Account as Account
import okx.MarketData as MarketData
import okx.PublicData as PublicData
import okx.Trade as Trade

from scalp.config import Settings

logger = logging.getLogger(__name__)


class OKXRestClient:
    def __init__(self, settings: Settings) -> None:
        flag = settings.okx_flag
        key = settings.okx_api_key
        secret = settings.okx_api_secret
        passphrase = settings.okx_passphrase

        self._trade = Trade.TradeAPI(key, secret, passphrase, False, flag)
        self._public = PublicData.PublicAPI(flag=flag)
        self._market = MarketData.MarketAPI(flag=flag)
        self._account = Account.AccountAPI(key, secret, passphrase, False, flag)

    # ── Instrument discovery ──────────────────────────────────────────────────

    def get_option_instruments(self, uly: str) -> list[dict[str, Any]]:
        """Return all active OPTION instruments for a given underlying."""
        result = self._public.get_instruments(instType="OPTION", uly=uly)
        instruments: list[dict[str, Any]] = result.get("data", [])
        logger.debug("Fetched %d option instruments for %s", len(instruments), uly)
        return instruments

    def get_futures_instrument(self, inst_id: str) -> dict[str, Any] | None:
        """Return instrument details for a SWAP."""
        result = self._public.get_instruments(instType="SWAP", instId=inst_id)
        data = result.get("data", [])
        return data[0] if data else None

    # ── Account state ─────────────────────────────────────────────────────────

    def get_positions(self) -> list[dict[str, Any]]:
        """Fetch all current open positions."""
        result = self._account.get_positions()
        return result.get("data", [])

    def get_balance(self) -> list[dict[str, Any]]:
        """Fetch account balances."""
        result = self._account.get_account_balance()
        return result.get("data", [])

    # ── Order management ──────────────────────────────────────────────────────

    def place_order(
        self,
        inst_id: str,
        side: str,             # "buy" | "sell"
        size: str,             # quantity as string (OKX requires str)
        td_mode: str = "cross",
        ord_type: str = "market",
        client_order_id: str = "",
        price: str = "",
    ) -> dict[str, Any]:
        """
        Place a market order.

        Returns the full OKX response dict.  The caller is responsible for
        checking `data[0]["sCode"] == "0"` for success.
        """
        if not client_order_id:
            client_order_id = uuid.uuid4().hex[:32]

        kwargs: dict[str, str] = {
            "instId": inst_id,
            "tdMode": td_mode,
            "side": side,
            "ordType": ord_type,
            "sz": size,
            "clOrdId": client_order_id,
        }
        if price:
            kwargs["px"] = price
        result = self._trade.place_order(**kwargs)
        logger.debug("place_order %s %s %s → %s", side, size, inst_id, result)
        return result

    def cancel_order(
        self,
        inst_id: str,
        order_id: str = "",
        client_order_id: str = "",
    ) -> dict[str, Any]:
        kwargs: dict[str, str] = {"instId": inst_id}
        if order_id:
            kwargs["ordId"] = order_id
        if client_order_id:
            kwargs["clOrdId"] = client_order_id
        result = self._trade.cancel_order(**kwargs)
        logger.debug("cancel_order %s → %s", inst_id, result)
        return result

    def get_order(
        self,
        inst_id: str,
        order_id: str = "",
        client_order_id: str = "",
    ) -> dict[str, Any] | None:
        kwargs: dict[str, str] = {"instId": inst_id}
        if order_id:
            kwargs["ordId"] = order_id
        if client_order_id:
            kwargs["clOrdId"] = client_order_id
        result = self._trade.get_order(**kwargs)
        data = result.get("data", [])
        return data[0] if data else None

    # ── Market data ───────────────────────────────────────────────────────────

    def get_mark_price(self, inst_id: str) -> float | None:
        """Fetch current mark price for a futures instrument."""
        result = self._market.get_mark_price(instType="SWAP", instId=inst_id)
        data = result.get("data", [])
        if data:
            try:
                return float(data[0]["markPx"])
            except (KeyError, ValueError):
                pass
        return None

    def get_ticker(self, inst_id: str) -> dict[str, Any] | None:
        """Fetch ticker snapshot for an instrument."""
        result = self._market.get_ticker(instId=inst_id)
        data = result.get("data", [])
        return data[0] if data else None

    def get_historical_candles(
        self,
        inst_id: str,
        bar: str = "5m",
        limit: int = 300,
        before: str = "",
        after: str = "",
    ) -> list[dict[str, Any]]:
        """
        Fetch historical index candles for HAR-RV bootstrap.

        Returns list of dicts with keys: ts, o, h, l, c, vol.
        """
        result = self._market.get_index_candlesticks(
            instId=inst_id,
            after=after,
            before=before,
            bar=bar,
            limit=str(limit),
        )
        return result.get("data", [])
