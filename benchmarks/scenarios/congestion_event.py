"""
Scenario: Congestion Event — Adaptive Scheduler Response.

Simulates a sudden traffic surge causing queue buildup.
Shows how the adaptive scheduler switches modes and restores service.

Stages:
1. Steady state (load=0.5)
2. Congestion onset (load=1.0, burst injection)
3. Recovery period
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
logging.basicConfig(level=logging.WARNING)

from slicenet.engine import SliceNetEngine


def run(scheduler: str = "adaptive") -> None:
    print(f"\n[Congestion Event Scenario — {scheduler.upper()}]")

    engine = SliceNetEngine(
        scheduler=scheduler,
        load_factor=0.5,
    )

    # Phase 1: Steady state
    print("  Phase 1: Steady state (load=0.5)...")
    r1 = engine.run(duration_ms=500.0, tick_ms=1.0, drain_per_tick=40)
    voip_p95_phase1 = r1["slices"].get("voip", {}).get("latency", {}).get("p95_ms", 0)
    print(f"    VoIP p95={voip_p95_phase1:.2f}ms queue_fill={engine.scheduler.size}")

    # Phase 2: Congestion — burst all traffic types
    print("  Phase 2: Congestion burst...")
    engine.set_load(1.0)
    engine.inject_burst("video", 300)
    engine.inject_burst("bulk", 200)
    engine.inject_burst("http", 100)
    r2 = engine.run(duration_ms=500.0, tick_ms=1.0, drain_per_tick=40, verbose=False)
    voip_p95_phase2 = r2["slices"].get("voip", {}).get("latency", {}).get("p95_ms", 0)
    print(f"    VoIP p95={voip_p95_phase2:.2f}ms queue_fill={engine.scheduler.size}")

    if scheduler == "adaptive":
        mode = engine.scheduler.mode if hasattr(engine.scheduler, "mode") else "N/A"
        print(f"    Adaptive mode after congestion: {mode}")

    # Phase 3: Recovery
    print("  Phase 3: Recovery (load=0.4)...")
    engine.set_load(0.4)
    r3 = engine.run(duration_ms=500.0, tick_ms=1.0, drain_per_tick=50)
    voip_p95_phase3 = r3["slices"].get("voip", {}).get("latency", {}).get("p95_ms", 0)
    print(f"    VoIP p95={voip_p95_phase3:.2f}ms queue_fill={engine.scheduler.size}")

    print(f"\n  Summary: VoIP p95 {voip_p95_phase1:.1f}ms → {voip_p95_phase2:.1f}ms → {voip_p95_phase3:.1f}ms")
    delta = voip_p95_phase2 - voip_p95_phase3
    print(f"  Recovery improvement: {delta:.1f}ms")


if __name__ == "__main__":
    for sched in ["fifo", "adaptive"]:
        run(sched)
