"""
Weighted Fair Queuing (WFQ) Scheduler.

WFQ assigns each flow/slice a weight and ensures each receives a proportional
share of the link bandwidth. Unlike priority scheduling, WFQ provides:
  - Starvation freedom: all flows eventually get service
  - Proportional fairness: throughput proportional to weight
  - Low latency for small-packet flows (e.g., VoIP)

Algorithm: Virtual Finish Time (VFT) based WFQ.
Each packet gets a virtual finish time:
    F(p) = max(F(prev_p_in_flow), V(arrival_time)) + packet_size / weight
Packets are served in increasing order of F(p).

This approximates GPS (Generalized Processor Sharing) for packet networks.
"""

from __future__ import annotations
import heapq
import logging
from typing import Dict, List, Optional, Tuple

from slicenet.core.packet import Packet
from slicenet.core.constants import SLICE_WEIGHT_MAP, MAX_QUEUE_SIZE

logger = logging.getLogger(__name__)


class WFQEntry:
    """Heap entry wrapping a packet with its virtual finish time."""
    __slots__ = ["virtual_finish_time", "seq", "packet"]

    def __init__(self, vft: float, seq: int, packet: Packet):
        self.virtual_finish_time = vft
        self.seq = seq  # tie-breaker
        self.packet = packet

    def __lt__(self, other: "WFQEntry") -> bool:
        if self.virtual_finish_time != other.virtual_finish_time:
            return self.virtual_finish_time < other.virtual_finish_time
        return self.seq < other.seq


class WFQScheduler:
    """
    Weighted Fair Queuing using Virtual Finish Time.

    Each slice has a configurable weight. Heavier slices drain faster
    (lower VFT means earlier service).
    """

    def __init__(
        self,
        weights: Optional[Dict[str, int]] = None,
        max_total_size: int = MAX_QUEUE_SIZE,
    ):
        self._weights: Dict[str, float] = {
            k: float(v) for k, v in (weights or SLICE_WEIGHT_MAP).items()
        }
        # Normalize weights so they sum to 100
        total = sum(self._weights.values())
        self._normalized_weights: Dict[str, float] = {
            k: v / total for k, v in self._weights.items()
        }
        self.max_total_size = max_total_size

        self._heap: List[WFQEntry] = []
        self._virtual_time: float = 0.0
        self._flow_vft: Dict[str, float] = {}  # last VFT per flow
        self._seq: int = 0

        self._enqueued: int = 0
        self._dequeued: int = 0
        self._dropped: int = 0

        logger.info(
            "WFQ initialized with weights: %s",
            {k: f"{v:.3f}" for k, v in self._normalized_weights.items()}
        )

    def enqueue(self, packet: Packet) -> bool:
        if len(self._heap) >= self.max_total_size:
            packet.mark_dropped("wfq_overflow")
            self._dropped += 1
            return False

        weight = self._normalized_weights.get(packet.slice_id, 0.1)

        # VFT = max(last finish time for this flow, virtual clock) + size/weight
        last_vft = self._flow_vft.get(packet.flow_id, self._virtual_time)
        start_time = max(last_vft, self._virtual_time)
        vft = start_time + (packet.size_bytes / weight)

        self._flow_vft[packet.flow_id] = vft

        entry = WFQEntry(vft=vft, seq=self._seq, packet=packet)
        self._seq += 1
        heapq.heappush(self._heap, entry)
        packet.mark_enqueued()
        self._enqueued += 1
        return True

    def dequeue(self) -> Optional[Packet]:
        if not self._heap:
            return None
        entry = heapq.heappop(self._heap)
        packet = entry.packet
        self._virtual_time = entry.virtual_finish_time
        packet.mark_dequeued()
        self._dequeued += 1
        return packet

    def drain(self, n: int) -> List[Packet]:
        result = []
        for _ in range(n):
            pkt = self.dequeue()
            if pkt is None:
                break
            result.append(pkt)
        return result

    @property
    def size(self) -> int:
        return len(self._heap)

    def fairness_index(self, per_slice_throughput: Dict[str, float]) -> float:
        """
        Jain's Fairness Index weighted by slice weights.
        1.0 = perfectly fair, lower = unfair.
        """
        if not per_slice_throughput:
            return 1.0
        slices = list(per_slice_throughput.keys())
        normalized = [
            per_slice_throughput[s] / max(self._weights.get(s, 1.0), 1e-9)
            for s in slices
        ]
        n = len(normalized)
        if n == 0:
            return 1.0
        sq_sum = sum(x for x in normalized) ** 2
        sum_sq = sum(x ** 2 for x in normalized)
        return sq_sum / (n * sum_sq) if sum_sq > 0 else 1.0

    def stats(self) -> dict:
        total = self._enqueued + self._dropped
        return {
            "scheduler": "wfq",
            "total_enqueued": self._enqueued,
            "total_dequeued": self._dequeued,
            "total_dropped": self._dropped,
            "drop_rate": round(self._dropped / total if total > 0 else 0.0, 4),
            "virtual_time": round(self._virtual_time, 2),
            "queue_size": len(self._heap),
        }
