"""
Metrics Collector.

Collects, aggregates, and exports performance metrics for the QoS engine.

Metrics tracked:
- Per-slice latency (p50, p95, p99)
- Throughput (packets/sec, bits/sec)
- Packet loss rate
- SLA violation rate
- Scheduler efficiency
- Queue depth over time
- Jain's fairness index
- TCP cwnd progression
"""

from __future__ import annotations
import logging
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from slicenet.core.constants import METRICS_WINDOW_SIZE, METRICS_PERCENTILES, LATENCY_SLA_MS

logger = logging.getLogger(__name__)


class LatencyHistogram:
    """Sliding window latency tracker with percentile computation."""

    def __init__(self, window_size: int = METRICS_WINDOW_SIZE):
        self._window: Deque[float] = deque(maxlen=window_size)
        self._total: float = 0.0
        self._count: int = 0
        self._violations: int = 0
        self._sla_ms: float = float("inf")

    def set_sla(self, sla_ms: float) -> None:
        self._sla_ms = sla_ms

    def record(self, latency_ms: float) -> None:
        self._window.append(latency_ms)
        self._total += latency_ms
        self._count += 1
        if latency_ms > self._sla_ms:
            self._violations += 1

    def percentile(self, p: float) -> float:
        if not self._window:
            return 0.0
        return float(np.percentile(list(self._window), p))

    @property
    def p50(self) -> float:
        return self.percentile(50)

    @property
    def p95(self) -> float:
        return self.percentile(95)

    @property
    def p99(self) -> float:
        return self.percentile(99)

    @property
    def mean(self) -> float:
        return self._total / self._count if self._count > 0 else 0.0

    @property
    def violation_rate(self) -> float:
        return self._violations / self._count if self._count > 0 else 0.0

    @property
    def sample_count(self) -> int:
        return self._count

    def summary(self) -> dict:
        return {
            "samples": self._count,
            "mean_ms": round(self.mean, 3),
            "p50_ms": round(self.p50, 3),
            "p95_ms": round(self.p95, 3),
            "p99_ms": round(self.p99, 3),
            "sla_violation_rate": round(self.violation_rate, 4),
        }


class ThroughputCounter:
    """Exponentially weighted moving average throughput tracker."""

    def __init__(self, alpha: float = 0.1):
        self._alpha = alpha
        self._ewma_bps: float = 0.0
        self._ewma_pps: float = 0.0
        self._last_ts: float = time.monotonic()
        self._bytes_since_last: int = 0
        self._pkts_since_last: int = 0
        self._total_bytes: int = 0
        self._total_packets: int = 0

    def record(self, packet_bytes: int) -> None:
        self._bytes_since_last += packet_bytes
        self._pkts_since_last += 1
        self._total_bytes += packet_bytes
        self._total_packets += 1

    def flush(self) -> Tuple[float, float]:
        """
        Update EWMA with accumulated bytes/packets since last flush.
        Returns (bps, pps).
        """
        now = time.monotonic()
        elapsed = now - self._last_ts
        if elapsed <= 0:
            return self._ewma_bps, self._ewma_pps

        instant_bps = (self._bytes_since_last * 8) / elapsed
        instant_pps = self._pkts_since_last / elapsed

        self._ewma_bps = self._alpha * instant_bps + (1 - self._alpha) * self._ewma_bps
        self._ewma_pps = self._alpha * instant_pps + (1 - self._alpha) * self._ewma_pps

        self._bytes_since_last = 0
        self._pkts_since_last = 0
        self._last_ts = now

        return self._ewma_bps, self._ewma_pps

    @property
    def mbps(self) -> float:
        return self._ewma_bps / 1e6

    @property
    def pps(self) -> float:
        return self._ewma_pps


class MetricsCollector:
    """
    Central metrics aggregator for the QoS engine.
    Thread-safe via GIL (single-threaded simulation).
    """

    def __init__(self, slices: List[str]):
        self._slices = slices
        self._latency: Dict[str, LatencyHistogram] = {}
        self._throughput: Dict[str, ThroughputCounter] = {}
        self._drops: Dict[str, int] = {}
        self._admitted: Dict[str, int] = {}
        self._queue_depth_history: Dict[str, Deque[int]] = {}

        # System-level metrics
        self._scheduler_calls: int = 0
        self._scheduler_empty_calls: int = 0
        self._start_time: float = time.monotonic()

        for sid in slices:
            hist = LatencyHistogram(window_size=METRICS_WINDOW_SIZE)
            hist.set_sla(LATENCY_SLA_MS.get(sid, 500.0))
            self._latency[sid] = hist
            self._throughput[sid] = ThroughputCounter()
            self._drops[sid] = 0
            self._admitted[sid] = 0
            self._queue_depth_history[sid] = deque(maxlen=500)

    def record_packet_transmitted(
        self, slice_id: str, packet_bytes: int, latency_ms: float
    ) -> None:
        if slice_id in self._latency:
            self._latency[slice_id].record(latency_ms)
            self._throughput[slice_id].record(packet_bytes)
            self._admitted[slice_id] += 1

    def record_packet_dropped(self, slice_id: str) -> None:
        self._drops[slice_id] = self._drops.get(slice_id, 0) + 1

    def record_queue_depth(self, slice_id: str, depth: int) -> None:
        if slice_id in self._queue_depth_history:
            self._queue_depth_history[slice_id].append(depth)

    def record_scheduler_call(self, produced_packet: bool) -> None:
        self._scheduler_calls += 1
        if not produced_packet:
            self._scheduler_empty_calls += 1

    def flush_throughput(self) -> Dict[str, Tuple[float, float]]:
        return {sid: self._throughput[sid].flush() for sid in self._slices}

    def latency_summary(self, slice_id: str) -> dict:
        return self._latency[slice_id].summary()

    def loss_rate(self, slice_id: str) -> float:
        total = self._admitted.get(slice_id, 0) + self._drops.get(slice_id, 0)
        return self._drops.get(slice_id, 0) / total if total > 0 else 0.0

    def scheduler_efficiency(self) -> float:
        if self._scheduler_calls == 0:
            return 1.0
        return 1.0 - (self._scheduler_empty_calls / self._scheduler_calls)

    def jains_fairness_index(self) -> float:
        """
        Jain's fairness index across slices based on transmitted packet counts.
        1.0 = perfectly fair, lower = some slices starved.
        """
        values = [self._admitted.get(sid, 0) for sid in self._slices]
        if not values or sum(values) == 0:
            return 1.0
        n = len(values)
        sq_sum = sum(values) ** 2
        sum_sq = sum(v ** 2 for v in values)
        return sq_sum / (n * sum_sq) if sum_sq > 0 else 1.0

    def full_report(self) -> dict:
        elapsed = time.monotonic() - self._start_time
        self.flush_throughput()

        report = {
            "elapsed_s": round(elapsed, 3),
            "scheduler_efficiency": round(self.scheduler_efficiency(), 4),
            "jains_fairness_index": round(self.jains_fairness_index(), 4),
            "slices": {},
        }
        for sid in self._slices:
            report["slices"][sid] = {
                "latency": self.latency_summary(sid),
                "loss_rate": round(self.loss_rate(sid), 4),
                "throughput_mbps": round(self._throughput[sid].mbps, 3),
                "packets_admitted": self._admitted.get(sid, 0),
                "packets_dropped": self._drops.get(sid, 0),
            }
        return report

    def print_report(self) -> None:
        report = self.full_report()
        print(f"\n{'='*70}")
        print(f"  SliceNet QoS Engine — Metrics Report ({report['elapsed_s']:.1f}s)")
        print(f"{'='*70}")
        print(f"  Scheduler efficiency : {report['scheduler_efficiency']:.1%}")
        print(f"  Jain fairness index  : {report['jains_fairness_index']:.4f}")
        print(f"\n  {'Slice':<15} {'p50ms':>7} {'p95ms':>7} {'p99ms':>7} "
              f"{'Loss%':>7} {'Mbps':>8} {'SLA Viol%':>10}")
        print(f"  {'-'*65}")
        for sid, s in report["slices"].items():
            lat = s["latency"]
            print(
                f"  {sid:<15} {lat['p50_ms']:>7.2f} {lat['p95_ms']:>7.2f} "
                f"{lat['p99_ms']:>7.2f} {s['loss_rate']*100:>7.3f} "
                f"{s['throughput_mbps']:>8.2f} "
                f"{lat['sla_violation_rate']*100:>10.2f}"
            )
        print(f"{'='*70}\n")
