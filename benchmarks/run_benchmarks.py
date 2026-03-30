"""
SliceNet Benchmarking Suite.

Runs all scheduler algorithms under identical load conditions.
Produces comparative metrics for p50/p95/p99 latency, throughput,
packet loss, and fairness.

Output: benchmarks/results/benchmark_results.json
        benchmarks/results/benchmark_plots.png
"""

from __future__ import annotations
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("benchmark")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SCHEDULERS = ["fifo", "priority", "wfq", "drr", "adaptive"]
DURATION_MS = 2000.0
TICK_MS = 1.0
LOAD_FACTOR = 0.8
DRAIN_PER_TICK = 40


def run_single(scheduler_name: str, load: float = LOAD_FACTOR) -> dict:
    from slicenet.engine import SliceNetEngine
    engine = SliceNetEngine(
        scheduler=scheduler_name,
        load_factor=load,
        tcp_loss_rate=0.002,
        udp_loss_rate=0.005,
    )
    start = time.perf_counter()
    report = engine.run(
        duration_ms=DURATION_MS,
        tick_ms=TICK_MS,
        drain_per_tick=DRAIN_PER_TICK,
        verbose=False,
    )
    elapsed = time.perf_counter() - start
    report["wall_time_s"] = round(elapsed, 4)
    report["scheduler_name"] = scheduler_name
    return report


def run_scalability_test() -> List[dict]:
    """Test all schedulers under increasing load."""
    results = []
    loads = [0.3, 0.5, 0.7, 0.9, 1.0]
    for load in loads:
        for sched in ["fifo", "wfq", "adaptive"]:
            from slicenet.engine import SliceNetEngine
            engine = SliceNetEngine(scheduler=sched, load_factor=load)
            report = engine.run(duration_ms=1000.0, tick_ms=1.0, drain_per_tick=40)
            row = {
                "scheduler": sched,
                "load": load,
                "jains_fairness": report.get("jains_fairness_index", 0),
                "scheduler_efficiency": report.get("scheduler_efficiency", 0),
            }
            for sid, sdata in report.get("slices", {}).items():
                row[f"{sid}_p95_ms"] = sdata["latency"]["p95_ms"]
                row[f"{sid}_loss_rate"] = sdata["loss_rate"]
            results.append(row)
            print(f"  load={load:.1f} sched={sched:10s} fair={row['jains_fairness']:.4f}")
    return results


def print_comparison_table(results: Dict[str, dict]) -> None:
    slices = ["voip", "video", "iot", "best_effort"]
    metrics = ["p50_ms", "p95_ms", "p99_ms"]

    for metric in metrics:
        print(f"\n--- {metric.upper()} Latency (ms) ---")
        header = f"{'Slice':<15}" + "".join(f"{s:>12}" for s in SCHEDULERS)
        print(header)
        print("-" * len(header))
        for sid in slices:
            row = f"{sid:<15}"
            for sched in SCHEDULERS:
                val = results[sched]["slices"].get(sid, {}).get("latency", {}).get(metric, 0)
                row += f"{val:>12.2f}"
            print(row)

    print("\n--- Packet Loss Rate (%) ---")
    header = f"{'Slice':<15}" + "".join(f"{s:>12}" for s in SCHEDULERS)
    print(header)
    print("-" * len(header))
    for sid in slices:
        row = f"{sid:<15}"
        for sched in SCHEDULERS:
            val = results[sched]["slices"].get(sid, {}).get("loss_rate", 0) * 100
            row += f"{val:>12.3f}"
        print(row)

    print("\n--- System Metrics ---")
    for sched in SCHEDULERS:
        r = results[sched]
        print(
            f"  {sched:12s} | fairness={r.get('jains_fairness_index', 0):.4f} "
            f"| sched_eff={r.get('scheduler_efficiency', 0):.3f} "
            f"| wall={r.get('wall_time_s', 0):.3f}s"
        )


def try_plot(results: Dict[str, dict]) -> None:
    """Generate comparison plots if matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not installed — skipping plots")
        return

    slices = ["voip", "video", "iot", "best_effort"]
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("SliceNet QoS Engine — Scheduler Comparison", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    colors = {"fifo": "#e74c3c", "priority": "#e67e22", "wfq": "#2ecc71",
              "drr": "#3498db", "adaptive": "#9b59b6"}

    # Plot 1: p95 latency per slice per scheduler
    ax1 = fig.add_subplot(gs[0, :2])
    x = np.arange(len(slices))
    width = 0.15
    for i, sched in enumerate(SCHEDULERS):
        vals = [
            results[sched]["slices"].get(sid, {}).get("latency", {}).get("p95_ms", 0)
            for sid in slices
        ]
        ax1.bar(x + i * width, vals, width, label=sched, color=colors[sched], alpha=0.85)
    ax1.set_xticks(x + width * 2)
    ax1.set_xticklabels(slices)
    ax1.set_ylabel("p95 Latency (ms)")
    ax1.set_title("p95 Latency by Slice and Scheduler")
    ax1.legend(fontsize=8)
    ax1.set_yscale("log")

    # Plot 2: Fairness index
    ax2 = fig.add_subplot(gs[0, 2])
    fair_vals = [results[s].get("jains_fairness_index", 0) for s in SCHEDULERS]
    ax2.bar(SCHEDULERS, fair_vals, color=[colors[s] for s in SCHEDULERS], alpha=0.85)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Jain's Fairness Index")
    ax2.set_title("Fairness (higher = better)")
    ax2.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8)

    # Plot 3: VoIP p99 latency (SLA critical)
    ax3 = fig.add_subplot(gs[1, 0])
    voip_p99 = [
        results[s]["slices"].get("voip", {}).get("latency", {}).get("p99_ms", 0)
        for s in SCHEDULERS
    ]
    ax3.bar(SCHEDULERS, voip_p99, color=[colors[s] for s in SCHEDULERS], alpha=0.85)
    ax3.axhline(y=20.0, color="red", linestyle="--", linewidth=1.5, label="SLA (20ms)")
    ax3.set_ylabel("VoIP p99 Latency (ms)")
    ax3.set_title("VoIP SLA Compliance")
    ax3.legend(fontsize=8)

    # Plot 4: Loss rates
    ax4 = fig.add_subplot(gs[1, 1])
    for i, sched in enumerate(SCHEDULERS):
        vals = [
            results[sched]["slices"].get(sid, {}).get("loss_rate", 0) * 100
            for sid in slices
        ]
        ax4.bar(x + i * width, vals, width, label=sched, color=colors[sched], alpha=0.85)
    ax4.set_xticks(x + width * 2)
    ax4.set_xticklabels(slices)
    ax4.set_ylabel("Loss Rate (%)")
    ax4.set_title("Packet Loss Rate by Slice")
    ax4.legend(fontsize=8)

    # Plot 5: Scheduler efficiency
    ax5 = fig.add_subplot(gs[1, 2])
    eff_vals = [results[s].get("scheduler_efficiency", 0) for s in SCHEDULERS]
    ax5.bar(SCHEDULERS, eff_vals, color=[colors[s] for s in SCHEDULERS], alpha=0.85)
    ax5.set_ylim(0, 1.05)
    ax5.set_ylabel("Scheduler Efficiency")
    ax5.set_title("Scheduler Efficiency")

    out_path = RESULTS_DIR / "benchmark_plots.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlots saved to: {out_path}")
    plt.close()


def main():
    print("=" * 70)
    print("  SliceNet QoS Engine — Benchmark Suite")
    print(f"  Duration: {DURATION_MS}ms | Tick: {TICK_MS}ms | Load: {LOAD_FACTOR}")
    print("=" * 70)

    results = {}

    for sched in SCHEDULERS:
        print(f"\nRunning {sched.upper()} scheduler...", end=" ", flush=True)
        r = run_single(sched)
        results[sched] = r
        voip_p95 = r["slices"].get("voip", {}).get("latency", {}).get("p95_ms", 0)
        fair = r.get("jains_fairness_index", 0)
        print(f"done | VoIP p95={voip_p95:.2f}ms | fairness={fair:.4f}")

    print("\n")
    print_comparison_table(results)

    print("\nRunning scalability test...")
    scalability = run_scalability_test()

    # Save results
    out = {"scheduler_comparison": results, "scalability": scalability}
    results_path = RESULTS_DIR / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    try_plot(results)

    print("\nDone.")


if __name__ == "__main__":
    main()
