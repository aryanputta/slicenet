"""
Deficit Round Robin (DRR) Scheduler.

DRR is a practical O(1) approximation of WFQ.
Invented by M. Shreedhar and G. Varghese (1995).

How it works:
- Each queue has a "deficit counter" (DC) and a quantum Q.
- Each round: DC += Q for each queue with packets.
- Dequeue packets from front while their size <= DC.
- DC carries over to next round (no wasted credit).

Key properties:
  - O(1) per-packet scheduling (unlike WFQ's O(log N) heap)
  - Max-min fairness approximation
  - Works well with variable-length packets (unlike round-robin)
  - Bounded latency per slice (quantum determines max service gap)

Used in: Linux tc DRR qdisc, Juniper MX series, and many ASIC schedulers.
"""

from __future__ import annotations
import logging
from collections import deque
from typing import Deque, Dict, List, Optional

from slicenet.core.packet import Packet
from slicenet.core.constants import DRR_QUANTUM_BYTES, SLICE_WEIGHT_MAP, MAX_QUEUE_SIZE

logger = logging.getLogger(__name__)


class DRRQueue:
    """Per-slice DRR queue state."""
    __slots__ = ["slice_id", "quantum", "deficit", "queue", "bytes_served"]

    def __init__(self, slice_id: str, quantum: int):
        self.slice_id = slice_id
        self.quantum = quantum
        self.deficit: int = 0
        self.queue: Deque[Packet] = deque()
        self.bytes_served: int = 0


class DRRScheduler:
    """
    Deficit Round Robin scheduler across multiple slices.

    Quantum is proportional to slice weight: heavier slices get larger quanta.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, int]] = None,
        base_quantum: int = DRR_QUANTUM_BYTES,
        max_per_queue: int = MAX_QUEUE_SIZE // 4,
    ):
        weights = weights or SLICE_WEIGHT_MAP
        self.base_quantum = base_quantum
        self.max_per_queue = max_per_queue

        self._queues: Dict[str, DRRQueue] = {}
        self._active_order: List[str] = []  # Round-robin order

        for slice_id, weight in weights.items():
            quantum = int(base_quantum * weight / 10)  # scale by weight
            self._queues[slice_id] = DRRQueue(slice_id=slice_id, quantum=quantum)
            self._active_order.append(slice_id)

        self._round_index: int = 0
        self._enqueued: int = 0
        self._dequeued: int = 0
        self._dropped: int = 0
        self._rounds: int = 0

        logger.info(
            "DRR initialized: slices=%s quanta=%s",
            list(self._queues.keys()),
            {sid: q.quantum for sid, q in self._queues.items()}
        )

    def enqueue(self, packet: Packet) -> bool:
        q = self._queues.get(packet.slice_id)
        if q is None:
            # Unknown slice — route to best_effort if available
            q = self._queues.get("best_effort")
            if q is None:
                packet.mark_dropped("unknown_slice")
                self._dropped += 1
                return False

        if len(q.queue) >= self.max_per_queue:
            packet.mark_dropped("drr_queue_overflow")
            self._dropped += 1
            return False

        packet.mark_enqueued()
        q.queue.append(packet)
        self._enqueued += 1
        return True

    def dequeue(self) -> Optional[Packet]:
        """
        DRR dequeue: iterate queues in round-robin, awarding deficit credits.
        Returns the next packet to transmit.
        """
        active_slices = [sid for sid in self._active_order if self._queues[sid].queue]
        if not active_slices:
            return None

        attempts = 0
        while attempts < len(active_slices) * 2:
            slice_id = self._active_order[self._round_index % len(self._active_order)]
            self._round_index += 1
            q = self._queues[slice_id]

            if not q.queue:
                continue

            # Replenish deficit
            q.deficit += q.quantum

            # Drain as many packets as deficit allows
            while q.queue and q.queue[0].size_bytes <= q.deficit:
                packet = q.queue.popleft()
                q.deficit -= packet.size_bytes
                q.bytes_served += packet.size_bytes
                packet.mark_dequeued()
                self._dequeued += 1
                return packet

            attempts += 1

        return None

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
        return sum(len(q.queue) for q in self._queues.values())

    def per_slice_sizes(self) -> Dict[str, int]:
        return {sid: len(q.queue) for sid, q in self._queues.items()}

    def stats(self) -> dict:
        total = self._enqueued + self._dropped
        return {
            "scheduler": "drr",
            "total_enqueued": self._enqueued,
            "total_dequeued": self._dequeued,
            "total_dropped": self._dropped,
            "drop_rate": round(self._dropped / total if total > 0 else 0.0, 4),
            "rounds": self._rounds,
            "per_slice_deficit": {
                sid: q.deficit for sid, q in self._queues.items()
            },
            "per_slice_bytes_served": {
                sid: q.bytes_served for sid, q in self._queues.items()
            },
        }
