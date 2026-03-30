"""
Network Slice definitions and QoS policy enforcement.
A slice is an isolated logical partition of network resources with guaranteed SLAs.
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from slicenet.core.constants import (
    LATENCY_SLA_MS, BANDWIDTH_GUARANTEE_MBPS, SLICE_PRIORITY_MAP,
    SLICE_WEIGHT_MAP, MAX_QUEUE_SIZE
)

logger = logging.getLogger(__name__)


@dataclass
class SliceSLA:
    """Service Level Agreement for a network slice."""
    max_latency_ms: float
    min_bandwidth_mbps: float
    max_packet_loss_rate: float = 0.001
    max_jitter_ms: float = 5.0
    priority: int = 4
    weight: int = 10


@dataclass
class SliceStats:
    """Runtime statistics for a slice."""
    packets_received: int = 0
    packets_transmitted: int = 0
    packets_dropped: int = 0
    bytes_transmitted: int = 0
    sla_violations: int = 0
    total_latency_ms: float = 0.0
    last_reset: float = field(default_factory=time.monotonic)

    @property
    def avg_latency_ms(self) -> float:
        if self.packets_transmitted == 0:
            return 0.0
        return self.total_latency_ms / self.packets_transmitted

    @property
    def loss_rate(self) -> float:
        total = self.packets_received
        return self.packets_dropped / total if total > 0 else 0.0

    @property
    def throughput_mbps(self) -> float:
        elapsed = time.monotonic() - self.last_reset
        return (self.bytes_transmitted * 8) / (elapsed * 1e6) if elapsed > 0 else 0.0


class NetworkSlice:
    """
    A network slice with enforced QoS policies.
    Owns its queue and enforces admission control against its SLA.
    """

    def __init__(self, slice_id: str, sla: SliceSLA, max_queue: int = MAX_QUEUE_SIZE):
        self.slice_id = slice_id
        self.sla = sla
        self.max_queue = max_queue
        self.stats = SliceStats()
        self._active = True

        logger.info(
            "Slice %s created | priority=%d weight=%d "
            "latency_sla=%.1fms bw_guarantee=%.1fMbps",
            slice_id, sla.priority, sla.weight,
            sla.max_latency_ms, sla.min_bandwidth_mbps
        )

    def check_sla_violation(self, latency_ms: float) -> bool:
        if latency_ms > self.sla.max_latency_ms:
            self.stats.sla_violations += 1
            return True
        return False

    def record_transmission(self, packet_size: int, latency_ms: float) -> None:
        self.stats.packets_transmitted += 1
        self.stats.bytes_transmitted += packet_size
        self.stats.total_latency_ms += latency_ms
        self.check_sla_violation(latency_ms)

    def record_drop(self) -> None:
        self.stats.packets_dropped += 1

    def record_arrival(self) -> None:
        self.stats.packets_received += 1

    @property
    def is_congested(self) -> bool:
        return self.stats.loss_rate > self.sla.max_packet_loss_rate

    def __repr__(self) -> str:
        return (
            f"NetworkSlice(id={self.slice_id}, priority={self.sla.priority}, "
            f"weight={self.sla.weight}, sla_lat={self.sla.max_latency_ms}ms)"
        )


def build_default_slices() -> Dict[str, NetworkSlice]:
    """
    Factory: builds the default slice topology used in all simulations.
    Returns a dict keyed by slice_id.
    """
    slices: Dict[str, NetworkSlice] = {}
    slice_ids = list(LATENCY_SLA_MS.keys())

    for sid in slice_ids:
        sla = SliceSLA(
            max_latency_ms=LATENCY_SLA_MS[sid],
            min_bandwidth_mbps=BANDWIDTH_GUARANTEE_MBPS[sid],
            priority=SLICE_PRIORITY_MAP[sid],
            weight=SLICE_WEIGHT_MAP[sid],
        )
        slices[sid] = NetworkSlice(slice_id=sid, sla=sla)

    return slices
