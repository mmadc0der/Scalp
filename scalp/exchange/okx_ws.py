"""
OKX WebSocket listener — fully native trio via trio-websocket.

Three persistent connections are maintained:
  • Public   — opt-summary, mark-price
  • Business — index-candle5m
  • Private  — orders (fills), authenticated via HMAC-SHA256

Both run inside a shared nursery with reconnect loops (exponential back-off
up to 60 s).  All incoming messages are forwarded into `channels.raw_send`
as `RawWSEvent` objects for the normalizer to fan out.

OKX heartbeat: the server sends a raw text "ping" frame every ~20 s.
We respond with a raw "pong" text frame.  If no message is received for
30 s we proactively close and reconnect.

Simulated trading endpoint: wss://wspap.okx.com/ws/v5/{public|private}
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time

import trio
from trio_websocket import (
    ConnectionClosed,
    HandshakeError,
    WebSocketConnection,
    open_websocket_url,
)

from scalp.channels import Channels
from scalp.config import Settings
from scalp.schema import RawWSEvent

logger = logging.getLogger(__name__)

_PING_INTERVAL_S = 20.0
_RECONNECT_BASE_S = 2.0
_RECONNECT_MAX_S = 60.0


# ── Authentication ────────────────────────────────────────────────────────────

def _okx_sign(secret: str, timestamp: str) -> str:
    """Compute OKX WebSocket login signature."""
    message = f"{timestamp}GET/users/self/verify"
    sig = hmac.new(
        secret.encode("utf-8"),
        message.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).digest()
    return base64.b64encode(sig).decode("utf-8")


async def _login(ws: WebSocketConnection, settings: Settings) -> bool:
    """Send login frame and wait for the success confirmation."""
    ts = str(int(time.time()))
    payload = {
        "op": "login",
        "args": [
            {
                "apiKey": settings.okx_api_key,
                "passphrase": settings.okx_passphrase,
                "timestamp": ts,
                "sign": _okx_sign(settings.okx_api_secret, ts),
            }
        ],
    }
    await ws.send_message(json.dumps(payload))

    with trio.move_on_after(10.0) as cancel_scope:
        while True:
            raw = await ws.get_message()
            if raw == "ping":
                await ws.send_message("pong")
                continue
            msg = json.loads(raw)
            event = msg.get("event")
            if event == "login":
                if msg.get("code") == "0":
                    logger.info("OKX private WS: login successful")
                    return True
                else:
                    logger.error("OKX private WS: login failed: %s", msg)
                    return False

    if cancel_scope.cancelled_caught:
        logger.error("OKX private WS: login timed out")
    return False


# ── Subscribe helpers ─────────────────────────────────────────────────────────

async def _subscribe(ws: WebSocketConnection, args: list[dict]) -> None:
    await ws.send_message(json.dumps({"op": "subscribe", "args": args}))


# ── Generic receive loop ──────────────────────────────────────────────────────

async def _recv_loop(
    ws: WebSocketConnection,
    channels: Channels,
    label: str,
) -> None:
    """
    Receive loop: parse messages and forward RawWSEvents.

    Handles raw "ping" heartbeats and logs subscribe confirmations.
    Raises ConnectionClosed if the peer closes the connection.
    """
    while True:
        try:
            raw = await ws.get_message()
        except ConnectionClosed as exc:
            logger.warning("%s: websocket closed (%s)", label, exc)
            return
        except OSError as exc:
            logger.warning("%s: socket error while receiving (%s)", label, exc)
            return

        if raw == "ping":
            await ws.send_message("pong")
            continue
        if raw == "pong":
            continue

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("%s: non-JSON message: %r", label, raw[:120])
            continue

        event = msg.get("event")
        if event in ("subscribe", "unsubscribe"):
            logger.debug("%s: %s ack: %s", label, event, msg.get("arg"))
            continue
        if event == "error":
            logger.warning("%s: WS error: %s", label, msg)
            continue

        arg = msg.get("arg", {})
        channel = arg.get("channel", "")
        data = msg.get("data", [])

        if not channel or not data:
            continue

        ev = RawWSEvent(channel=channel, arg=arg, data=data)
        try:
            channels.raw_send.send_nowait(ev)
        except trio.WouldBlock:
            logger.debug("%s: raw_ch full, dropping %s event", label, channel)


async def _keepalive_loop(ws: WebSocketConnection, label: str) -> None:
    """
    Send periodic ping frames to keep connections alive when channels are idle.
    """
    while True:
        await trio.sleep(_PING_INTERVAL_S)
        try:
            await ws.send_message("ping")
        except (ConnectionClosed, OSError) as exc:
            logger.warning("%s: keepalive stopped (%s)", label, exc)
            return
        except Exception as exc:
            logger.warning("%s: keepalive unexpected error: %s", label, exc)
            return


async def _run_stream_session(
    ws: WebSocketConnection,
    channels: Channels,
    label: str,
) -> None:
    """
    Run recv and keepalive loops; exit cleanly on disconnect.
    """
    async with trio.open_nursery() as nursery:
        nursery.start_soon(_keepalive_loop, ws, label)
        await _recv_loop(ws, channels, label)
        nursery.cancel_scope.cancel()


# ── Public connection ─────────────────────────────────────────────────────────

async def _public_ws_task(channels: Channels, settings: Settings) -> None:
    backoff = _RECONNECT_BASE_S
    while True:
        try:
            async with open_websocket_url(settings.ws_public_url) as ws:
                backoff = _RECONNECT_BASE_S  # reset on successful connect

                await _subscribe(ws, [
                    {"channel": "opt-summary", "instFamily": settings.uly},
                    {"channel": "mark-price", "instId": settings.futures_inst_id},
                ])
                logger.info("OKX public WS connected: %s", settings.ws_public_url)
                await _run_stream_session(ws, channels, "public")

        except (ConnectionClosed, HandshakeError, OSError) as exc:
            logger.warning("Public WS disconnected (%s), retry in %.0fs", exc, backoff)
            await trio.sleep(backoff)
            backoff = min(backoff * 2, _RECONNECT_MAX_S)
        except Exception as exc:
            logger.warning("Public WS unexpected error (%s), retry in %.0fs", exc, backoff)
            await trio.sleep(backoff)
            backoff = min(backoff * 2, _RECONNECT_MAX_S)


# ── Business connection ───────────────────────────────────────────────────────

async def _business_ws_task(channels: Channels, settings: Settings) -> None:
    backoff = _RECONNECT_BASE_S
    while True:
        try:
            async with open_websocket_url(settings.ws_business_url) as ws:
                backoff = _RECONNECT_BASE_S

                await _subscribe(ws, [
                    {"channel": "index-candle5m", "instId": settings.index_inst_id},
                ])
                logger.info("OKX business WS connected: %s", settings.ws_business_url)
                await _run_stream_session(ws, channels, "business")

        except (ConnectionClosed, HandshakeError, OSError) as exc:
            logger.warning("Business WS disconnected (%s), retry in %.0fs", exc, backoff)
            await trio.sleep(backoff)
            backoff = min(backoff * 2, _RECONNECT_MAX_S)
        except Exception as exc:
            logger.warning("Business WS unexpected error (%s), retry in %.0fs", exc, backoff)
            await trio.sleep(backoff)
            backoff = min(backoff * 2, _RECONNECT_MAX_S)


# ── Private connection ────────────────────────────────────────────────────────

async def _private_ws_task(channels: Channels, settings: Settings) -> None:
    backoff = _RECONNECT_BASE_S
    while True:
        try:
            async with open_websocket_url(settings.ws_private_url) as ws:
                backoff = _RECONNECT_BASE_S

                if not await _login(ws, settings):
                    logger.error("Private WS login failed; retrying in %.0fs", backoff)
                    await trio.sleep(backoff)
                    backoff = min(backoff * 2, _RECONNECT_MAX_S)
                    continue

                # Subscribe to fills for options and the hedge swap
                await _subscribe(ws, [
                    {"channel": "orders", "instType": "OPTION", "instFamily": settings.uly},
                    {"channel": "orders", "instType": "SWAP",
                     "instId": settings.futures_inst_id},
                ])
                logger.info("OKX private WS connected and subscribed")
                await _run_stream_session(ws, channels, "private")

        except (ConnectionClosed, HandshakeError, OSError) as exc:
            logger.warning("Private WS disconnected (%s), retry in %.0fs", exc, backoff)
            await trio.sleep(backoff)
            backoff = min(backoff * 2, _RECONNECT_MAX_S)
        except Exception as exc:
            logger.warning("Private WS unexpected error (%s), retry in %.0fs", exc, backoff)
            await trio.sleep(backoff)
            backoff = min(backoff * 2, _RECONNECT_MAX_S)


# ── Top-level task ────────────────────────────────────────────────────────────

async def ws_listener_task(channels: Channels, settings: Settings) -> None:
    """
    Spawn and supervise public, business, and private WebSocket connections.
    Either connection failing independently will reconnect without tearing
    down the other.
    """
    async with trio.open_nursery() as nursery:
        nursery.start_soon(_public_ws_task, channels, settings)
        nursery.start_soon(_business_ws_task, channels, settings)
        nursery.start_soon(_private_ws_task, channels, settings)
