"""
Central channel registry.

All inter-task communication happens through these typed trio MemoryChannels.
Call `create_channels(settings)` once at startup, then pass the `Channels`
object into every task.

Broadcast note: `surface_ch` has two consumers (SignalEngine and
PortfolioTracker). trio MemoryChannels are single-consumer, so we use
two separate channels: `surface_signal` and `surface_portfolio`.
The SVIFitter task sends to both.
"""

from __future__ import annotations

from dataclasses import dataclass

import trio

from scalp.config import Settings
from scalp.schema import (
    Candle,
    FillEvent,
    OBSnapshot,
    Order,
    PortfolioState,
    RVForecast,
    RawWSEvent,
    RegimeEvent,
    VolSurface,
)


@dataclass
class Channels:
    # OKX WS → Normalizer
    raw_send: trio.MemorySendChannel[RawWSEvent]
    raw_recv: trio.MemoryReceiveChannel[RawWSEvent]

    # Normalizer → SVIFitter
    ob_snapshot_send: trio.MemorySendChannel[OBSnapshot]
    ob_snapshot_recv: trio.MemoryReceiveChannel[OBSnapshot]

    # Normalizer → HARRVForecaster
    candle_send: trio.MemorySendChannel[Candle]
    candle_recv: trio.MemoryReceiveChannel[Candle]

    # Normalizer + OrderManager (fills) → PortfolioTracker
    fill_send: trio.MemorySendChannel[FillEvent]
    fill_recv: trio.MemoryReceiveChannel[FillEvent]

    # SVIFitter → SignalEngine
    surface_signal_send: trio.MemorySendChannel[VolSurface]
    surface_signal_recv: trio.MemoryReceiveChannel[VolSurface]

    # SVIFitter → PortfolioTracker (vol repricing)
    surface_portfolio_send: trio.MemorySendChannel[VolSurface]
    surface_portfolio_recv: trio.MemoryReceiveChannel[VolSurface]

    # HARRVForecaster → SignalEngine
    rv_forecast_send: trio.MemorySendChannel[RVForecast]
    rv_forecast_recv: trio.MemoryReceiveChannel[RVForecast]

    # SignalEngine → OrderManager (regime-driven trade orders)
    signal_send: trio.MemorySendChannel[RegimeEvent]
    signal_recv: trio.MemoryReceiveChannel[RegimeEvent]

    # HedgeEngine → OrderManager (delta-neutralising hedge orders)
    order_send: trio.MemorySendChannel[Order]
    order_recv: trio.MemoryReceiveChannel[Order]

    # PortfolioTracker → HedgeEngine (state pushes on every update)
    portfolio_state_send: trio.MemorySendChannel[PortfolioState]
    portfolio_state_recv: trio.MemoryReceiveChannel[PortfolioState]


def create_channels(settings: Settings) -> Channels:
    n = settings.channel_maxsize

    raw_s, raw_r = trio.open_memory_channel[RawWSEvent](n)
    ob_s, ob_r = trio.open_memory_channel[OBSnapshot](n)
    candle_s, candle_r = trio.open_memory_channel[Candle](n)
    fill_s, fill_r = trio.open_memory_channel[FillEvent](n)
    surf_sig_s, surf_sig_r = trio.open_memory_channel[VolSurface](8)
    surf_port_s, surf_port_r = trio.open_memory_channel[VolSurface](8)
    rv_s, rv_r = trio.open_memory_channel[RVForecast](8)
    sig_s, sig_r = trio.open_memory_channel[RegimeEvent](16)
    ord_s, ord_r = trio.open_memory_channel[Order](n)
    ps_s, ps_r = trio.open_memory_channel[PortfolioState](8)

    return Channels(
        raw_send=raw_s,
        raw_recv=raw_r,
        ob_snapshot_send=ob_s,
        ob_snapshot_recv=ob_r,
        candle_send=candle_s,
        candle_recv=candle_r,
        fill_send=fill_s,
        fill_recv=fill_r,
        surface_signal_send=surf_sig_s,
        surface_signal_recv=surf_sig_r,
        surface_portfolio_send=surf_port_s,
        surface_portfolio_recv=surf_port_r,
        rv_forecast_send=rv_s,
        rv_forecast_recv=rv_r,
        signal_send=sig_s,
        signal_recv=sig_r,
        order_send=ord_s,
        order_recv=ord_r,
        portfolio_state_send=ps_s,
        portfolio_state_recv=ps_r,
    )
