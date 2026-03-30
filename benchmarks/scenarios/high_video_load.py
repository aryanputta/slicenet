"""
Scenario: High Video Load — VoIP Latency Protection Test.

Injects a large burst of video traffic and verifies that VoIP
latency remains within SLA (20ms p99) while video competes for bandwidth.

Expected outcome:
- WFQ/DRR/Adaptive: VoIP p99 stays under 20ms
- FIFO: VoIP latency degrades (head-of-line blocking)
"""

import sys
from pathlib import Path

import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from slicenet.engine import SliceNetEngine

logging.basicConfig(level=logging.WARNING)
VOIP_SLA_MS = 20.0


def run(scheduler: str = "adaptive") -> dict:
    engine = SliceNetEngine(
        scheduler=scheduler,
        load_factor=0.6,
        tcp_loss_rate=0.001,
        udp_loss_rate=0.002,
    )

    # Warm up
    engine.run(duration_ms=200.0, tick_ms=1.0, drain_per_tick=30)

    # Inject video burst to stress system
    engine.inject_burst("video", count=500)
    engine.inject_burst("bulk", count=200)

    # Run under stress
    report = engine.run(duration_ms=1000.0, tick_ms=1.0, drain_per_tick=50)

    voip = report["slices"].get("voip", {})
    voip_p99 = voip.get("latency", {}).get("p99_ms", float("inf"))
    sla_ok = voip_p99 <= VOIP_SLA_MS

    print(f"\n[{scheduler.upper()}] High Video Load Scenario")
    print(f"  VoIP p99 latency : {voip_p99:.2f}ms  ({'PASS' if sla_ok else 'FAIL'} SLA={VOIP_SLA_MS}ms)")
    print(f"  Video p95 latency: {report['slices'].get('video', {}).get('latency', {}).get('p95_ms', 0):.2f}ms")
    print(f"  Jain fairness    : {report.get('jains_fairness_index', 0):.4f}")

    engine.print_report()
    return report


if __name__ == "__main__":
    for sched in ["fifo", "wfq", "drr", "adaptive"]:
        run(sched)
