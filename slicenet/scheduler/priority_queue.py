"""
Priority Queue Scheduler.

Strict priority scheduling: lower priority value = dequeued first.
Per-priority FIFO queues to avoid intra-priority reordering.

Tradeoffs vs FIFO:
  + VoIP/critical traffic gets immediate service
  - Low-priority flows can starve under sustained high-priority load
  - No fairness guarantee between same-priority flows

Real-world analog: DSCP-based priority queuing in Cisco QoS.
"""

from __future__ import annotations
import logging
from collections import deque
from typing import Deque, List, Optional

from slicenet.core.packet import Packet
from slicenet.core.constants import (
    PRIORITY_BEST_EFFORT, MAX_QUEUE_SIZE
)

logger = logging.getLogger(__name__)

NUM_PRIORITY_LEVELS = PRIORITY_BEST_EFFORT + 1


class PriorityScheduler:
    """
    Multi-level priority queue.
    Strict priority: all packets at level N are drained before level N+1.
    """

    def __init__(self, max_size_per_level: int = MAX_QUEUE_SIZE // NUM_PRIORITY_LEVELS):
        self.max_size_per_level = max_size_per_level
        self._queues: List[Deque[Packet]] = [
            deque() for _ in range(NUM_PRIORITY_LEVELS)
        ]
        self._enqueued: int = 0
        self._dequeued: int = 0
        self._dropped: int = 0

    def enqueue(self, packet: Packet) -> bool:
        level = max(0, min(packet.priority, NUM_PRIORITY_LEVELS - 1))
        q = self._queues[level]
        if len(q) >= self.max_size_per_level:
            packet.mark_dropped("priority_queue_overflow")
            self._dropped += 1
            logger.debug(
                "Priority queue overflow at level %d for flow %s",
                level, packet.flow_id
            )
            return False
        packet.mark_enqueued()
        q.append(packet)
        self._enqueued += 1
        return True

    def dequeue(self) -> Optional[Packet]:
        """Strict priority dequeue — lowest level number wins."""
        for level, q in enumerate(self._queues):
            if q:
                packet = q.popleft()
                packet.mark_dequeued()
                self._dequeued += 1
                return packet
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
        return sum(len(q) for q in self._queues)

    @property
    def level_sizes(self) -> List[int]:
        return [len(q) for q in self._queues]

    def stats(self) -> dict:
        return {
            "scheduler": "priority_queue",
            "total_enqueued": self._enqueued,
            "total_dequeued": self._dequeued,
            "total_dropped": self._dropped,
            "drop_rate": round(
                self._dropped / (self._enqueued + self._dropped)
                if (self._enqueued + self._dropped) > 0 else 0.0, 4
            ),
            "level_sizes": self.level_sizes,
        }
