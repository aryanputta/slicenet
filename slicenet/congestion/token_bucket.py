"""
Token Bucket and Leaky Bucket rate limiters.

Token Bucket:
- Tokens accumulate at rate R up to burst size B
- Each packet consumes tokens equal to its byte size
- Allows bursts up to B bytes, then enforces rate R
- Used for: per-slice rate limiting, SLA enforcement

Leaky Bucket:
- Output is smoothed at a constant rate regardless of input burst
- Models a FIFO queue draining at a fixed rate
"""

from __future__ import annotations
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TokenBucket:
    """
    Token bucket rate limiter.
    Thread-unsafe — designed for single-threaded simulation loop.
    """

    def __init__(self, rate_bps: float, burst_bytes: int, name: str = ""):
        """
        Args:
            rate_bps: Sustained token generation rate in bits per second
            burst_bytes: Max burst size in bytes (bucket capacity)
            name: Identifier for logging
        """
        self.rate_bps = rate_bps
        self.rate_bytes_per_sec = rate_bps / 8.0
        self.burst_bytes = float(burst_bytes)
        self.name = name

        self._tokens: float = float(burst_bytes)  # start full
        self._last_refill: float = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        new_tokens = elapsed * self.rate_bytes_per_sec
        self._tokens = min(self._tokens + new_tokens, self.burst_bytes)
        self._last_refill = now

    def consume(self, packet_bytes: int) -> bool:
        """
        Attempt to consume tokens for a packet.
        Returns True if packet is allowed, False if it must be dropped/deferred.
        """
        self._refill()
        if self._tokens >= packet_bytes:
            self._tokens -= packet_bytes
            return True
        logger.debug(
            "TokenBucket[%s] insufficient tokens: need=%d have=%.0f",
            self.name, packet_bytes, self._tokens
        )
        return False

    def available_bytes(self) -> float:
        self._refill()
        return self._tokens

    @property
    def fill_ratio(self) -> float:
        self._refill()
        return self._tokens / self.burst_bytes if self.burst_bytes > 0 else 0.0

    def reset(self) -> None:
        self._tokens = self.burst_bytes
        self._last_refill = time.monotonic()

    def update_rate(self, new_rate_bps: float) -> None:
        """Dynamically adjust rate (used by adaptive controller)."""
        self._refill()  # apply current rate before changing
        self.rate_bps = new_rate_bps
        self.rate_bytes_per_sec = new_rate_bps / 8.0


class LeakyBucket:
    """
    Leaky bucket rate limiter / traffic shaper.
    Enforces a strictly constant output rate, smoothing bursts.
    """

    def __init__(self, rate_bps: float, capacity_bytes: int, name: str = ""):
        self.rate_bps = rate_bps
        self.rate_bytes_per_sec = rate_bps / 8.0
        self.capacity_bytes = capacity_bytes
        self.name = name

        self._water: float = 0.0  # current bytes in bucket
        self._last_leak: float = time.monotonic()

    def _leak(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_leak
        leaked = elapsed * self.rate_bytes_per_sec
        self._water = max(0.0, self._water - leaked)
        self._last_leak = now

    def add(self, packet_bytes: int) -> bool:
        """
        Attempt to add packet to the leaky bucket.
        Returns True if accepted, False if bucket is full (overflow/drop).
        """
        self._leak()
        if self._water + packet_bytes <= self.capacity_bytes:
            self._water += packet_bytes
            return True
        logger.debug(
            "LeakyBucket[%s] overflow: water=%.0f cap=%d pkt=%d",
            self.name, self._water, self.capacity_bytes, packet_bytes
        )
        return False

    @property
    def fill_ratio(self) -> float:
        self._leak()
        return self._water / self.capacity_bytes if self.capacity_bytes > 0 else 0.0

    def wait_time_ms(self, packet_bytes: int) -> float:
        """Returns how long (ms) until bucket has room for this packet."""
        self._leak()
        overflow = (self._water + packet_bytes) - self.capacity_bytes
        if overflow <= 0:
            return 0.0
        return (overflow / self.rate_bytes_per_sec) * 1000.0
