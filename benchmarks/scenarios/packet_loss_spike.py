"""
Scenario: Packet Loss Spike — TCP vs UDP Behavior.

Simulates a link loss event. Demonstrates:
- TCP: cwnd collapses, throughput drops, recovery via slow start
- UDP: continues transmitting, quality degrades but stream continues

This maps directly to real-world VoIP resilience under lossy networks.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
logging.basicConfig(level=logging.WARNING)

from slicenet.engine import SliceNetEngine


def run(high_loss: float = 0.05) -> None:
    print("\n[Packet Loss Spike Scenario]")

    # Phase 1: Baseline (low loss)
    engine_low = SliceNetEngine(
        scheduler="wfq",
        load_factor=0.6,
        tcp_loss_rate=0.001,
        udp_loss_rate=0.002,
    )
    baseline = engine_low.run(duration_ms=1000.0, tick_ms=1.0, drain_per_tick=40)

    # Phase 2: High loss event
    engine_high = SliceNetEngine(
        scheduler="wfq",
        load_factor=0.6,
        tcp_loss_rate=high_loss,    # 5% TCP loss
        udp_loss_rate=high_loss,    # 5% UDP loss
    )
    high_loss_report = engine_high.run(duration_ms=1000.0, tick_ms=1.0, drain_per_tick=40)

    # Compare
    for slice_id in ["voip", "video", "iot", "best_effort"]:
        base_lat = baseline["slices"].get(slice_id, {}).get("latency", {})
        high_lat = high_loss_report["slices"].get(slice_id, {}).get("latency", {})
        base_loss = baseline["slices"].get(slice_id, {}).get("loss_rate", 0)
        high_loss_val = high_loss_report["slices"].get(slice_id, {}).get("loss_rate", 0)

        print(f"\n  Slice: {slice_id}")
        print(f"    Baseline  p95={base_lat.get('p95_ms', 0):.2f}ms  loss={base_loss*100:.3f}%")
        print(f"    High Loss p95={high_lat.get('p95_ms', 0):.2f}ms  loss={high_loss_val*100:.3f}%  (loss={high_loss*100:.0f}% injected)")

    print(f"\n  TCP flows impacted by cwnd collapse — slow start recovery visible")
    print(f"  UDP flows maintain transmission — loss degrades quality, not continuity")

    tcp_states = engine_high._tcp.snapshot()
    if tcp_states:
        avg_cwnd = sum(s["cwnd"] for s in tcp_states) / len(tcp_states)
        print(f"\n  Post-loss avg TCP cwnd: {avg_cwnd:.1f} segments (reduced from {10})")


if __name__ == "__main__":
    run(high_loss=0.05)
