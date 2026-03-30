"""
Congestion Controller.

Monitors per-slice and system-wide congestion state.
Enforces rate limits, triggers RED, and signals the adaptive controller.

Combines:
- Per-slice token bucket rate limiting
- RED active queue management
- Backpressure signaling
- System-wide congestion detection
"""

from __future__ import annotations
import logging
import time
from typing import Dict, Optional, Tuple

from slicenet.core.packet import Packet, Protocol
from slicenet.core.constants import (
    TOKEN_BUCKET_MAX_BURST, BANDWIDTH_GUARANTEE_MBPS,
    RED_MIN_THRESHOLD, RED_MAX_THRESHOLD, RED_MAX_PROBABILITY, RED_WEIGHT,
    MAX_QUEUE_SIZE
)
from slicenet.congestion.token_bucket import TokenBucket
from slicenet.congestion.red import REDQueueManager

logger = logging.getLogger(__name__)


class SliceCongestionState:
    """Per-slice congestion tracking and enforcement."""

    def __init__(self, slice_id: str, rate_mbps: float, burst_bytes: int):
        self.slice_id = slice_id
        self.token_bucket = TokenBucket(
            rate_bps=rate_mbps * 1e6,
            burst_bytes=burst_bytes,
            name=f"tb:{slice_id}"
        )
        self.red = REDQueueManager(
            min_thresh=RED_MIN_THRESHOLD,
            max_thresh=RED_MAX_THRESHOLD,
            max_prob=RED_MAX_PROBABILITY,
            weight=RED_WEIGHT,
            name=f"red:{slice_id}"
        )
        self.in_backpressure: bool = False
        self.backpressure_since: float = 0.0

    def apply_backpressure(self) -> None:
        if not self.in_backpressure:
            self.in_backpressure = True
            self.backpressure_since = time.monotonic()
            logger.warning("Backpressure activated for slice: %s", self.slice_id)

    def release_backpressure(self) -> None:
        if self.in_backpressure:
            duration = time.monotonic() - self.backpressure_since
            self.in_backpressure = False
            logger.info(
                "Backpressure released for slice %s after %.1fms",
                self.slice_id, duration * 1000
            )


class CongestionController:
    """
    System-wide congestion controller.

    Sits between the traffic generator and the scheduler.
    Makes admit/drop decisions per packet based on:
      1. Token bucket rate limiting (per slice)
      2. RED probabilistic drop (per slice queue state)
      3. System-level backpressure
    """

    def __init__(self, queue_sizes: Dict[str, int]):
        self._states: Dict[str, SliceCongestionState] = {}
        self._queue_sizes = queue_sizes  # current queue lengths (updated externally)

        _MAX_BURST_BYTES = 1_000_000  # 1 MB hard cap
        for slice_id, rate_mbps in BANDWIDTH_GUARANTEE_MBPS.items():
            burst = int(rate_mbps * 1e6 / 8 * 0.1)  # 100ms burst budget
            burst = max(TOKEN_BUCKET_MAX_BURST, min(burst, _MAX_BURST_BYTES))
            self._states[slice_id] = SliceCongestionState(
                slice_id=slice_id,
                rate_mbps=rate_mbps * 2.0,  # allow 2x guarantee before limiting
                burst_bytes=burst,
            )
            logger.info(
                "Congestion controller initialized for slice %s: "
                "rate=%.1fMbps burst=%d bytes",
                slice_id, rate_mbps * 2.0, burst
            )

    def admit(
        self, packet: Packet, current_queue_len: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Admit or drop a packet.
        Returns (admitted: bool, reason: Optional[str]).

        For UDP: rate limiting and queue overflow, but no retransmit signal.
        For TCP: same, plus signals congestion back to TCP engine.
        """
        state = self._states.get(packet.slice_id)
        if state is None:
            return True, None  # unknown slice — pass through

        # Check backpressure
        if state.in_backpressure:
            # UDP: drop silently; TCP: drop + signal loss
            if packet.protocol == Protocol.UDP:
                return False, "backpressure"
            return False, "backpressure_tcp"

        # RED probabilistic early drop
        if state.red.should_drop(current_queue_len):
            if packet.protocol == Protocol.TCP:
                return False, "red_tcp"
            return False, "red_udp"

        # Token bucket rate limit
        if not state.token_bucket.consume(packet.size_bytes):
            return False, "rate_limit"

        # Apply backpressure if queue is critically full
        if current_queue_len >= int(MAX_QUEUE_SIZE * 0.9):
            state.apply_backpressure()
        elif current_queue_len < int(MAX_QUEUE_SIZE * 0.5):
            state.release_backpressure()

        return True, None

    def update_queue_length(self, slice_id: str, queue_len: int) -> None:
        self._queue_sizes[slice_id] = queue_len

    def adjust_rate(self, slice_id: str, new_rate_mbps: float) -> None:
        """Adaptive rate adjustment called by the adaptive controller."""
        state = self._states.get(slice_id)
        if state:
            state.token_bucket.update_rate(new_rate_mbps * 1e6)
            logger.info(
                "Rate adjusted for slice %s: %.2f Mbps", slice_id, new_rate_mbps
            )

    def stats(self) -> Dict[str, dict]:
        return {
            sid: {
                "red": s.red.stats(),
                "token_bucket_fill": round(s.token_bucket.fill_ratio, 3),
                "in_backpressure": s.in_backpressure,
            }
            for sid, s in self._states.items()
        }
