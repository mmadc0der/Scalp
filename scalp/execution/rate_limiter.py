"""
Token-bucket rate limiter for OKX REST API endpoints.

OKX rate limits (as of 2025) relevant to this system:

    POST /api/v5/trade/order          — 60 req/2s  per instrument family
    POST /api/v5/trade/cancel-order   — 60 req/2s
    GET  /api/v5/trade/order          — 60 req/2s

We track one bucket per "endpoint class" (identified by a string key).
`acquire()` is an async method that sleeps until a token is available.

Usage
-----
    limiter = RateLimiter()
    await limiter.acquire("trade")  # before any trade REST call
"""

from __future__ import annotations

import time

import trio


class _TokenBucket:
    """
    Token bucket with a fixed capacity and a constant refill rate.

    capacity: maximum burst size in tokens
    rate:     tokens added per second
    """

    def __init__(self, capacity: float, rate: float) -> None:
        self._capacity = capacity
        self._rate = rate
        self._tokens = capacity
        self._last_refill = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now

    def consume(self, tokens: float = 1.0) -> float:
        """
        Attempt to consume `tokens`.

        Returns 0.0 if successful, otherwise the number of seconds to wait
        before the tokens will be available.
        """
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return 0.0
        deficit = tokens - self._tokens
        return deficit / self._rate


class RateLimiter:
    """
    Multi-endpoint token-bucket rate limiter.

    Pre-configured buckets:
      "trade"   — place / cancel orders: 60 req / 2 s → 30 req/s
      "query"   — order status polls: 60 req / 2 s → 30 req/s
      "public"  — public REST endpoints: 20 req/2 s → 10 req/s
    """

    _PROFILES: dict[str, tuple[float, float]] = {
        "trade":  (30.0, 30.0),   # (capacity, rate/s)
        "query":  (30.0, 30.0),
        "public": (10.0, 10.0),
    }

    def __init__(self) -> None:
        self._buckets: dict[str, _TokenBucket] = {
            key: _TokenBucket(cap, rate)
            for key, (cap, rate) in self._PROFILES.items()
        }

    async def acquire(self, endpoint: str = "trade", tokens: float = 1.0) -> None:
        """
        Block until the token bucket for `endpoint` has capacity.

        If `endpoint` is unknown, the call passes through immediately.
        """
        bucket = self._buckets.get(endpoint)
        if bucket is None:
            return

        while True:
            wait = bucket.consume(tokens)
            if wait == 0.0:
                return
            await trio.sleep(wait)
