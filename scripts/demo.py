"""
SliceNet QoS Engine — Interactive Demo.

Runs all 4 demo scenarios and prints a summary.
Suitable for interview/recruiting demonstrations.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

from slicenet.engine import SliceNetEngine


BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║         SliceNet — High-Performance QoS-Aware Network Slicer    ║
║         TCP/UDP simulation | WFQ | DRR | Adaptive Scheduler     ║
╚══════════════════════════════════════════════════════════════════╝
"""


def demo_scheduler_comparison():
    print("\n[Demo 1] Scheduler Comparison — VoIP Latency Under Load")
    print("-" * 60)
    for sched in ["fifo", "priority", "wfq", "drr", "adaptive"]:
        engine = SliceNetEngine(scheduler=sched, load_factor=0.75)
        report = engine.run(duration_ms=1000.0, tick_ms=1.0, drain_per_tick=40)
        voip = report["slices"].get("voip", {}).get("latency", {})
        print(
            f"  {sched:12s} | VoIP p50={voip.get('p50_ms', 0):6.2f}ms "
            f"p95={voip.get('p95_ms', 0):6.2f}ms "
            f"p99={voip.get('p99_ms', 0):6.2f}ms "
            f"fair={report.get('jains_fairness_index', 0):.4f}"
        )


def demo_congestion_response():
    print("\n[Demo 2] Adaptive Scheduler — Congestion Response")
    print("-" * 60)

    engine = SliceNetEngine(scheduler="adaptive", load_factor=0.5)

    r_base = engine.run(duration_ms=300.0, tick_ms=1.0, drain_per_tick=30)
    voip_base = r_base["slices"].get("voip", {}).get("latency", {}).get("p95_ms", 0)

    print(f"  Baseline VoIP p95: {voip_base:.2f}ms")

    engine.inject_burst("video", 400)
    engine.inject_burst("bulk", 200)
    engine.set_load(0.95)

    r_congested = engine.run(duration_ms=300.0, tick_ms=1.0, drain_per_tick=50)
    voip_congested = r_congested["slices"].get("voip", {}).get("latency", {}).get("p95_ms", 0)

    print(f"  Under congestion VoIP p95: {voip_congested:.2f}ms")
    print(f"  Scheduler mode: {engine.scheduler.mode if hasattr(engine.scheduler, 'mode') else 'N/A'}")

    engine.set_load(0.4)
    r_recovery = engine.run(duration_ms=300.0, tick_ms=1.0, drain_per_tick=50)
    voip_recovery = r_recovery["slices"].get("voip", {}).get("latency", {}).get("p95_ms", 0)
    print(f"  After recovery VoIP p95: {voip_recovery:.2f}ms")


def demo_tcp_udp_under_loss():
    print("\n[Demo 3] TCP vs UDP Under Packet Loss")
    print("-" * 60)

    for loss in [0.001, 0.01, 0.05]:
        e = SliceNetEngine(scheduler="wfq", load_factor=0.6,
                           tcp_loss_rate=loss, udp_loss_rate=loss)
        r = e.run(duration_ms=800.0, tick_ms=1.0, drain_per_tick=40)

        iot_loss = r["slices"].get("iot", {}).get("loss_rate", 0)        # TCP
        voip_loss = r["slices"].get("voip", {}).get("loss_rate", 0)      # UDP
        iot_p95 = r["slices"].get("iot", {}).get("latency", {}).get("p95_ms", 0)
        voip_p95 = r["slices"].get("voip", {}).get("latency", {}).get("p95_ms", 0)

        print(
            f"  loss={loss*100:.1f}% | "
            f"TCP(IoT) loss={iot_loss*100:.2f}% p95={iot_p95:.1f}ms | "
            f"UDP(VoIP) loss={voip_loss*100:.2f}% p95={voip_p95:.1f}ms"
        )


def demo_before_after_optimization():
    print("\n[Demo 4] Before vs After Optimization")
    print("-" * 60)

    load = 0.85
    slices_of_interest = ["voip", "video"]
    metrics = ["p95_ms", "p99_ms"]

    before_engine = SliceNetEngine(scheduler="fifo", load_factor=load)
    before = before_engine.run(duration_ms=1000.0, tick_ms=1.0, drain_per_tick=40)

    after_engine = SliceNetEngine(scheduler="adaptive", load_factor=load)
    after = after_engine.run(duration_ms=1000.0, tick_ms=1.0, drain_per_tick=40)

    print(f"  {'Metric':<28} {'FIFO (before)':>15} {'Adaptive (after)':>18} {'Δ':>8}")
    print("  " + "-" * 72)

    for sid in slices_of_interest:
        for m in metrics:
            b_val = before["slices"].get(sid, {}).get("latency", {}).get(m, 0)
            a_val = after["slices"].get(sid, {}).get("latency", {}).get(m, 0)
            delta = b_val - a_val
            pct = (delta / b_val * 100) if b_val > 0 else 0
            print(
                f"  {sid} {m:<20} {b_val:>15.2f} {a_val:>18.2f} {pct:>+7.1f}%"
            )

    b_fair = before.get("jains_fairness_index", 0)
    a_fair = after.get("jains_fairness_index", 0)
    print(f"\n  Fairness index: {b_fair:.4f} → {a_fair:.4f}  ({(a_fair - b_fair)*100:+.2f}pp)")


def main():
    print(BANNER)
    demo_scheduler_comparison()
    demo_congestion_response()
    demo_tcp_udp_under_loss()
    demo_before_after_optimization()
    print("\n✓ Demo complete.\n")


if __name__ == "__main__":
    main()
