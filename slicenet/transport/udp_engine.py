"""
UDP Transport Engine.

UDP characteristics:
- No retransmission
- No congestion control
- No connection state
- Low latency, loss-tolerant
- Used for: VoIP, video streaming, IoT telemetry

Simulates the fire-and-forget model with configurable loss injection.
"""

from __future__ import annotations
import logging
import random
from typing import Dict, Optional, Tuple

from slicenet.core.packet import Packet

logger = logging.getLogger(__name__)


class UDPFlowStats:
    """Per-flow UDP statistics."""
    __slots__ = ["flow_id", "sent", "lost", "bytes_sent"]

    def __init__(self, flow_id: str):
        self.flow_id = flow_id
        self.sent: int = 0
        self.lost: int = 0
        self.bytes_sent: int = 0

    @property
    def loss_rate(self) -> float:
        return self.lost / self.sent if self.sent > 0 else 0.0


class UDPEngine:
    """
    UDP transport simulation.

    Unlike TCP, UDP has no cwnd, no retransmissions.
    Packets are accepted or dropped based purely on link conditions and queue state.
    This models real VoIP/video stream behavior where:
      - Packet loss degrades quality but does not stall the stream
      - No head-of-line blocking (each datagram is independent)
      - Latency is bounded by link RTT only, not retransmit timers
    """

    def __init__(self, loss_rate: float = 0.005, max_queue_fill: float = 0.8):
        self.loss_rate = loss_rate
        self.max_queue_fill = max_queue_fill  # drop if queue above this fraction
        self._stats: Dict[str, UDPFlowStats] = {}

    def register_flow(self, flow_id: str) -> UDPFlowStats:
        stats = UDPFlowStats(flow_id)
        self._stats[flow_id] = stats
        return stats

    def process_packet(
        self, packet: Packet, queue_fill_ratio: float = 0.0
    ) -> Tuple[bool, Optional[str]]:
        """
        Simulate UDP send decision.
        Returns (accepted: bool, drop_reason: Optional[str]).

        UDP never blocks on cwnd, but drops under link loss or queue overflow.
        No retransmission is triggered on drop — packet is silently lost.
        """
        stats = self._stats.get(packet.flow_id)
        if stats is None:
            stats = self.register_flow(packet.flow_id)

        stats.sent += 1

        # Queue overflow drop (tail drop for UDP)
        if queue_fill_ratio >= self.max_queue_fill:
            stats.lost += 1
            logger.debug(
                "UDP drop: queue overflow flow=%s fill=%.2f",
                packet.flow_id, queue_fill_ratio
            )
            return False, "queue_overflow"

        # Simulated link-level loss
        if random.random() < self.loss_rate:
            stats.lost += 1
            logger.debug("UDP drop: link loss flow=%s", packet.flow_id)
            return False, "link_loss"

        stats.bytes_sent += packet.size_bytes
        return True, None

    def get_stats(self, flow_id: str) -> Optional[UDPFlowStats]:
        return self._stats.get(flow_id)

    def aggregate_loss_rate(self) -> float:
        total_sent = sum(s.sent for s in self._stats.values())
        total_lost = sum(s.lost for s in self._stats.values())
        return total_lost / total_sent if total_sent > 0 else 0.0
