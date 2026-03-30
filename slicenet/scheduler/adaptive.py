"""
Adaptive Scheduler.

Dynamically switches between scheduling algorithms and adjusts weights
based on real-time telemetry: queue depths, SLA violations, latency trends.

Behavior:
- Starts in WFQ mode (balanced fairness)
- Under SLA violation: boosts violating slice weight temporarily
- Under sustained congestion: increases DRR quanta for critical slices
- Under low load: reverts to baseline FIFO-like behavior
- Continuously re-evaluates via sliding window metrics

This mirrors production adaptive QoS engines like NVIDIA BlueField DPU's
traffic manager or Cisco IOS QoS auto-tuning.
"""

from __future__ import annotations
import logging
import time
from collections import deque
from typing import Deque, Dict, Optional

from slicenet.scheduler.wfq import WFQScheduler
from slicenet.scheduler.drr import DRRScheduler
from slicenet.scheduler.priority_queue import PriorityScheduler
from slicenet.core.packet import Packet
from slicenet.core.constants import (
    SLICE_WEIGHT_MAP, LATENCY_SLA_MS, MAX_QUEUE_SIZE
)

logger = logging.getLogger(__name__)

# Thresholds for mode switching
CONGESTION_QUEUE_THRESHOLD = 0.7    # 70% queue fill triggers congestion mode
VIOLATION_RATE_THRESHOLD = 0.05     # 5% SLA violation rate triggers boost
LOW_LOAD_THRESHOLD = 0.1            # <10% queue fill = low load
ADAPTATION_INTERVAL_S = 0.5        # Re-evaluate every 500ms


class AdaptiveScheduler:
    """
    Adaptive QoS scheduler with runtime algorithm selection and weight tuning.

    Wraps WFQ and DRR, switching modes based on observed system state.
    """

    def __init__(self, max_size: int = MAX_QUEUE_SIZE):
        self.max_size = max_size
        self._weights = dict(SLICE_WEIGHT_MAP)
        self._baseline_weights = dict(SLICE_WEIGHT_MAP)

        self._wfq = WFQScheduler(weights=self._weights, max_total_size=max_size)
        self._drr = DRRScheduler(weights=self._weights, max_per_queue=max_size // 4)
        self._priority = PriorityScheduler()

        self._mode: str = "wfq"  # wfq | drr | priority
        self._last_adaptation: float = time.monotonic()

        # Sliding window: (slice_id, latency_ms) samples
        self._latency_window: Dict[str, Deque[float]] = {
            sid: deque(maxlen=200) for sid in LATENCY_SLA_MS
        }
        self._violation_count: Dict[str, int] = {sid: 0 for sid in LATENCY_SLA_MS}
        self._sample_count: Dict[str, int] = {sid: 0 for sid in LATENCY_SLA_MS}

        self._total_enqueued: int = 0
        self._total_dropped: int = 0

        logger.info("Adaptive scheduler initialized in WFQ mode")

    def enqueue(self, packet: Packet) -> bool:
        self._total_enqueued += 1
        if self._mode == "wfq":
            return self._wfq.enqueue(packet)
        elif self._mode == "drr":
            return self._drr.enqueue(packet)
        else:
            return self._priority.enqueue(packet)

    def dequeue(self) -> Optional[Packet]:
        if self._mode == "wfq":
            return self._wfq.dequeue()
        elif self._mode == "drr":
            return self._drr.dequeue()
        else:
            return self._priority.dequeue()

    def drain(self, n: int) -> list:
        return [pkt for pkt in (self.dequeue() for _ in range(n)) if pkt]

    def record_latency(self, slice_id: str, latency_ms: float) -> None:
        if slice_id in self._latency_window:
            self._latency_window[slice_id].append(latency_ms)
            self._sample_count[slice_id] += 1
            sla = LATENCY_SLA_MS.get(slice_id, 500.0)
            if latency_ms > sla:
                self._violation_count[slice_id] += 1

    def adapt(self, queue_fill_ratio: float) -> None:
        """
        Run adaptation logic. Call periodically (every ADAPTATION_INTERVAL_S).
        Adjusts scheduler mode and weights based on telemetry.
        """
        now = time.monotonic()
        if now - self._last_adaptation < ADAPTATION_INTERVAL_S:
            return
        self._last_adaptation = now

        prev_mode = self._mode

        # Compute per-slice violation rates
        violation_rates: Dict[str, float] = {}
        for sid in LATENCY_SLA_MS:
            samples = self._sample_count[sid]
            viols = self._violation_count[sid]
            violation_rates[sid] = viols / samples if samples > 0 else 0.0

        # --- Mode selection logic ---
        max_violation_rate = max(violation_rates.values()) if violation_rates else 0.0

        if queue_fill_ratio >= CONGESTION_QUEUE_THRESHOLD:
            # Heavy congestion → strict priority to protect critical traffic
            if self._mode != "priority":
                self._mode = "priority"
                logger.warning(
                    "Adaptive: switching to PRIORITY mode (queue_fill=%.2f)",
                    queue_fill_ratio
                )
        elif max_violation_rate > VIOLATION_RATE_THRESHOLD:
            # SLA violations present → DRR gives guaranteed quanta
            if self._mode != "drr":
                self._mode = "drr"
                self._boost_violating_slices(violation_rates)
                logger.info(
                    "Adaptive: switching to DRR mode (max_violation=%.3f)",
                    max_violation_rate
                )
        elif queue_fill_ratio < LOW_LOAD_THRESHOLD:
            # Low load → WFQ for fairness
            if self._mode != "wfq":
                self._mode = "wfq"
                self._reset_weights()
                logger.info("Adaptive: switching to WFQ mode (low load)")
        else:
            # Moderate load → stay in WFQ, tune weights
            if self._mode == "wfq":
                self._tune_wfq_weights(violation_rates)

        # Reset per-interval counters
        for sid in self._violation_count:
            self._violation_count[sid] = 0
            self._sample_count[sid] = 0

        if self._mode != prev_mode:
            logger.info(
                "Adaptive scheduler mode change: %s → %s", prev_mode, self._mode
            )

    def _boost_violating_slices(self, violation_rates: Dict[str, float]) -> None:
        """Temporarily increase quantum/weight for slices with SLA violations."""
        new_weights = dict(self._baseline_weights)
        for sid, rate in violation_rates.items():
            if rate > VIOLATION_RATE_THRESHOLD:
                boost = min(2.0, 1.0 + rate * 10)
                new_weights[sid] = int(new_weights[sid] * boost)
                logger.debug("Boosting slice %s weight by %.2fx", sid, boost)
        self._drr = DRRScheduler(weights=new_weights, max_per_queue=self.max_size // 4)

    def _tune_wfq_weights(self, violation_rates: Dict[str, float]) -> None:
        """Gradually adjust WFQ weights toward SLA compliance."""
        new_weights = dict(self._weights)
        changed = False
        for sid, rate in violation_rates.items():
            if rate > VIOLATION_RATE_THRESHOLD and new_weights.get(sid, 10) < 80:
                new_weights[sid] = min(80, int(new_weights[sid] * 1.2))
                changed = True
        if changed:
            self._weights = new_weights
            self._wfq = WFQScheduler(weights=new_weights, max_total_size=self.max_size)

    def _reset_weights(self) -> None:
        self._weights = dict(self._baseline_weights)
        self._wfq = WFQScheduler(weights=self._weights, max_total_size=self.max_size)

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def size(self) -> int:
        if self._mode == "wfq":
            return self._wfq.size
        elif self._mode == "drr":
            return self._drr.size
        return self._priority.size

    def stats(self) -> dict:
        base = {
            "scheduler": f"adaptive:{self._mode}",
            "mode": self._mode,
            "total_enqueued": self._total_enqueued,
            "queue_size": self.size,
        }
        if self._mode == "wfq":
            base.update(self._wfq.stats())
        elif self._mode == "drr":
            base.update(self._drr.stats())
        else:
            base.update(self._priority.stats())
        return base
