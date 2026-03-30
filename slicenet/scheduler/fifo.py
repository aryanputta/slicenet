"""
FIFO Scheduler — baseline implementation.

Pure first-in, first-out. No priority awareness.
Establishes the performance baseline all other schedulers are compared against.

Limitations demonstrated in benchmarks:
- Head-of-line blocking: high-priority VoIP blocked by bulk transfers
- No starvation protection
- Unfair under heterogeneous traffic
"""

from __future__ import annotations
import logging
from collections import deque
from typing import Deque, Optional

from slicenet.core.packet import Packet

logger = logging.getLogger(__name__)


class FIFOScheduler:
    """Single global FIFO queue. No slice or priority awareness."""

    def __init__(self, max_size: int = 1000, name: str = "fifo"):
        self.max_size = max_size
        self.name = name
        self._queue: Deque[Packet] = deque()
        self._enqueued: int = 0
        self._dropped: int = 0
        self._dequeued: int = 0

    def enqueue(self, packet: Packet) -> bool:
        if len(self._queue) >= self.max_size:
            packet.mark_dropped("fifo_overflow")
            self._dropped += 1
            return False
        packet.mark_enqueued()
        self._queue.append(packet)
        self._enqueued += 1
        return True

    def dequeue(self) -> Optional[Packet]:
        if not self._queue:
            return None
        packet = self._queue.popleft()
        packet.mark_dequeued()
        self._dequeued += 1
        return packet

    def drain(self, n: int) -> list[Packet]:
        """Dequeue up to n packets in one call (batch scheduling)."""
        result = []
        for _ in range(min(n, len(self._queue))):
            pkt = self.dequeue()
            if pkt:
                result.append(pkt)
        return result

    @property
    def size(self) -> int:
        return len(self._queue)

    @property
    def drop_rate(self) -> float:
        total = self._enqueued + self._dropped
        return self._dropped / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            "scheduler": self.name,
            "queued": len(self._queue),
            "total_enqueued": self._enqueued,
            "total_dequeued": self._dequeued,
            "total_dropped": self._dropped,
            "drop_rate": round(self.drop_rate, 4),
        }
