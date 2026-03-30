"""
Random Early Detection (RED) Active Queue Management.

RED proactively drops packets before the queue fills completely,
signaling congestion to TCP senders early enough to avoid buffer bloat.

RFC 2309 / RFC 7567 compliant design.

Key parameters:
  min_thresh: queue avg below this → no drops
  max_thresh: queue avg above this → drop all (tail drop)
  between:    drop with linearly increasing probability
  max_prob:   max drop probability at max_thresh
  weight:     EWMA weight for avg queue length estimation
"""

from __future__ import annotations
import logging
import random

logger = logging.getLogger(__name__)


class REDQueueManager:
    """
    RED active queue management.

    Maintains an exponentially-weighted moving average (EWMA) of queue length
    and drops packets probabilistically to prevent synchronization of TCP flows.
    """

    def __init__(
        self,
        min_thresh: int = 200,
        max_thresh: int = 600,
        max_prob: float = 0.1,
        weight: float = 0.002,
        name: str = "",
    ):
        """
        Args:
            min_thresh: Min average queue length below which no drops occur
            max_thresh: Queue length above which all packets are dropped
            max_prob:   Maximum drop probability in the linear zone
            weight:     EWMA coefficient for average queue estimate (0 < w < 1)
            name:       Identifier for logging
        """
        if min_thresh >= max_thresh:
            raise ValueError("min_thresh must be < max_thresh")
        self.min_thresh = min_thresh
        self.max_thresh = max_thresh
        self.max_prob = max_prob
        self.weight = weight
        self.name = name

        self._avg_queue: float = 0.0
        self._count: int = 0          # packets since last drop (for gentle RED)
        self._total_arrivals: int = 0
        self._total_drops: int = 0

    def update(self, current_queue_len: int) -> None:
        """
        Update EWMA average queue length.
        Should be called on each packet arrival.
        """
        self._avg_queue = (
            (1 - self.weight) * self._avg_queue
            + self.weight * current_queue_len
        )

    def should_drop(self, current_queue_len: int) -> bool:
        """
        Decide whether to drop the arriving packet.

        Returns True if the packet should be dropped (congestion signal).
        Implements Gentle RED: probability ramps from 0 to max_prob between
        min and max thresholds.
        """
        self.update(current_queue_len)
        self._total_arrivals += 1

        avg = self._avg_queue

        if avg < self.min_thresh:
            # No congestion signal — queue is healthy
            self._count = 0
            return False

        if avg >= self.max_thresh:
            # Severe congestion — drop all
            self._total_drops += 1
            logger.debug(
                "RED[%s] hard drop: avg_q=%.1f >= max_thresh=%d",
                self.name, avg, self.max_thresh
            )
            return True

        # Linear probability zone: min_thresh <= avg < max_thresh
        # p_b = max_prob * (avg - min_thresh) / (max_thresh - min_thresh)
        p_b = self.max_prob * (avg - self.min_thresh) / (self.max_thresh - self.min_thresh)

        # Gentle RED: increase probability with consecutive non-drops
        self._count += 1
        p_a = p_b / (1.0 - self._count * p_b) if (1.0 - self._count * p_b) > 0 else p_b

        if random.random() < p_a:
            self._count = 0
            self._total_drops += 1
            logger.debug(
                "RED[%s] probabilistic drop: avg_q=%.1f p=%.4f",
                self.name, avg, p_a
            )
            return True

        return False

    @property
    def avg_queue_length(self) -> float:
        return self._avg_queue

    @property
    def drop_rate(self) -> float:
        if self._total_arrivals == 0:
            return 0.0
        return self._total_drops / self._total_arrivals

    def stats(self) -> dict:
        return {
            "name": self.name,
            "avg_queue": round(self._avg_queue, 2),
            "total_arrivals": self._total_arrivals,
            "total_drops": self._total_drops,
            "drop_rate": round(self.drop_rate, 4),
        }
