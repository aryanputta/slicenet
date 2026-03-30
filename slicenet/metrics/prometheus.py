"""
Prometheus text-format metrics exporter.

Renders SliceNet MetricsCollector data as Prometheus exposition format
(https://prometheus.io/docs/instrumenting/exposition_formats/).

Usage:
    from slicenet.metrics.prometheus import render_prometheus
    text = render_prometheus(engine.metrics)

Or via the REST API:
    GET /metrics/prometheus
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from slicenet.metrics.collector import MetricsCollector


def _gauge(name: str, value: float, labels: dict[str, str] | None = None) -> str:
    """Format a single Prometheus gauge line."""
    if labels:
        label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
        return f"{name}{{{label_str}}} {value}"
    return f"{name} {value}"


def render_prometheus(metrics: "MetricsCollector") -> str:
    """
    Render all SliceNet metrics as Prometheus text format.

    Metric families exported:
      slicenet_latency_p50_ms         (gauge, per slice)
      slicenet_latency_p95_ms         (gauge, per slice)
      slicenet_latency_p99_ms         (gauge, per slice)
      slicenet_latency_mean_ms        (gauge, per slice)
      slicenet_sla_violation_rate     (gauge, per slice)
      slicenet_throughput_mbps        (gauge, per slice)
      slicenet_loss_rate              (gauge, per slice)
      slicenet_packets_admitted_total (counter, per slice)
      slicenet_packets_dropped_total  (counter, per slice)
      slicenet_jains_fairness_index   (gauge, system)
      slicenet_scheduler_efficiency   (gauge, system)
      slicenet_scrape_timestamp_ms    (gauge, system)
    """
    lines: list[str] = []
    report = metrics.full_report()
    now_ms = time.monotonic() * 1000.0

    # --- Per-slice latency ---
    lines.append("# HELP slicenet_latency_p50_ms Median queuing latency (ms)")
    lines.append("# TYPE slicenet_latency_p50_ms gauge")
    lines.append("# HELP slicenet_latency_p95_ms 95th-percentile queuing latency (ms)")
    lines.append("# TYPE slicenet_latency_p95_ms gauge")
    lines.append("# HELP slicenet_latency_p99_ms 99th-percentile queuing latency (ms)")
    lines.append("# TYPE slicenet_latency_p99_ms gauge")
    lines.append("# HELP slicenet_latency_mean_ms Mean queuing latency (ms)")
    lines.append("# TYPE slicenet_latency_mean_ms gauge")
    lines.append("# HELP slicenet_sla_violation_rate Fraction of packets exceeding SLA latency")
    lines.append("# TYPE slicenet_sla_violation_rate gauge")
    lines.append("# HELP slicenet_throughput_mbps Current throughput (Mbps, EWMA)")
    lines.append("# TYPE slicenet_throughput_mbps gauge")
    lines.append("# HELP slicenet_loss_rate Packet loss rate per slice")
    lines.append("# TYPE slicenet_loss_rate gauge")
    lines.append("# HELP slicenet_packets_admitted_total Cumulative admitted packets")
    lines.append("# TYPE slicenet_packets_admitted_total counter")
    lines.append("# HELP slicenet_packets_dropped_total Cumulative dropped packets")
    lines.append("# TYPE slicenet_packets_dropped_total counter")

    for slice_id, data in report.get("slices", {}).items():
        lbl = {"slice": slice_id}
        lat = data.get("latency", {})

        def _v(key: str) -> float:
            val = lat.get(key)
            return round(val, 4) if val is not None else 0.0

        lines.append(_gauge("slicenet_latency_p50_ms", _v("p50"), lbl))
        lines.append(_gauge("slicenet_latency_p95_ms", _v("p95"), lbl))
        lines.append(_gauge("slicenet_latency_p99_ms", _v("p99"), lbl))
        lines.append(_gauge("slicenet_latency_mean_ms", _v("mean"), lbl))
        lines.append(_gauge(
            "slicenet_sla_violation_rate",
            round(lat.get("violation_rate", 0.0), 6),
            lbl,
        ))
        lines.append(_gauge(
            "slicenet_throughput_mbps",
            round(data.get("throughput_mbps", 0.0), 4),
            lbl,
        ))
        lines.append(_gauge(
            "slicenet_loss_rate",
            round(data.get("loss_rate", 0.0), 6),
            lbl,
        ))
        lines.append(_gauge(
            "slicenet_packets_admitted_total",
            int(data.get("admitted", 0)),
            lbl,
        ))
        lines.append(_gauge(
            "slicenet_packets_dropped_total",
            int(data.get("dropped", 0)),
            lbl,
        ))

    # --- System-level metrics ---
    sys_m = report.get("system", {})
    lines.append("")
    lines.append("# HELP slicenet_jains_fairness_index Jain's fairness index (1.0 = perfect)")
    lines.append("# TYPE slicenet_jains_fairness_index gauge")
    lines.append(_gauge(
        "slicenet_jains_fairness_index",
        round(sys_m.get("jains_fairness_index", 0.0), 6),
    ))

    lines.append("# HELP slicenet_scheduler_efficiency Fraction of non-empty scheduler calls")
    lines.append("# TYPE slicenet_scheduler_efficiency gauge")
    lines.append(_gauge(
        "slicenet_scheduler_efficiency",
        round(sys_m.get("scheduler_efficiency", 0.0), 6),
    ))

    lines.append("# HELP slicenet_scrape_timestamp_ms Unix timestamp of this scrape (ms)")
    lines.append("# TYPE slicenet_scrape_timestamp_ms gauge")
    lines.append(_gauge("slicenet_scrape_timestamp_ms", round(now_ms, 3)))

    lines.append("")  # trailing newline required by Prometheus spec
    return "\n".join(lines)
