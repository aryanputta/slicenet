"""
SliceNet QoS Engine — command-line interface.

Commands:
  run       Run a simulation and print metrics
  demo      Run all 4 built-in demo scenarios
  bench     Run the full benchmark suite
  serve     Start the FastAPI REST + WebSocket server
  topo      Print a topology summary and shortest paths

Examples:
    python -m slicenet.cli run --scheduler adaptive --load 0.85 --duration 3000
    python -m slicenet.cli run --scheduler wfq --transport bbr --loss 0.01
    python -m slicenet.cli serve --port 8000
    python -m slicenet.cli topo
    python -m slicenet.cli demo
    python -m slicenet.cli bench
"""

from __future__ import annotations

import argparse
import sys


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------

def _cmd_run(args: argparse.Namespace) -> None:
    from slicenet.engine import SliceNetEngine

    print(
        f"[SliceNet] Starting simulation | scheduler={args.scheduler} "
        f"transport={args.transport} load={args.load} "
        f"duration={args.duration}ms loss_tcp={args.loss} loss_udp={args.loss}"
    )

    engine = SliceNetEngine(
        scheduler=args.scheduler,
        load_factor=args.load,
        tcp_loss_rate=args.loss,
        udp_loss_rate=args.loss,
        link_capacity_mbps=args.capacity,
    )
    engine.run(
        duration_ms=args.duration,
        tick_ms=args.tick,
        drain_per_tick=args.drain,
        verbose=args.verbose,
    )
    engine.metrics.print_report()


def _cmd_demo(args: argparse.Namespace) -> None:
    import runpy
    import os

    demo_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "scripts", "demo.py"
    )
    runpy.run_path(demo_path, run_name="__main__")


def _cmd_bench(args: argparse.Namespace) -> None:
    import runpy
    import os

    bench_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "benchmarks", "run_benchmarks.py"
    )
    runpy.run_path(bench_path, run_name="__main__")


def _cmd_serve(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError:
        print("ERROR: uvicorn is required. Install with: pip install uvicorn fastapi pydantic")
        sys.exit(1)

    print(f"[SliceNet] Starting API server on http://0.0.0.0:{args.port}")
    print(f"[SliceNet] WebSocket stream: ws://0.0.0.0:{args.port}/ws/metrics")
    print(f"[SliceNet] Prometheus endpoint: http://0.0.0.0:{args.port}/metrics/prometheus")
    uvicorn.run(
        "slicenet.api.server:app",
        host=args.host,
        port=args.port,
        log_level="info",
        reload=args.reload,
    )


def _cmd_topo(args: argparse.Namespace) -> None:
    from slicenet.topology.network import FaultTolerantTopology

    topo = FaultTolerantTopology(name="5G-Core")

    # Build a representative 5G core topology:
    #
    #   UE ─── gNB ─── UPF-Edge ─── UPF-Core ─── DN (Data Network)
    #                     │                          │
    #                   RAN-BH ──────────────── IMS-Server
    #
    nodes = [
        ("ue", "User Equipment"),
        ("gnb", "gNodeB (5G RAN)"),
        ("upf_edge", "UPF Edge (MEC)"),
        ("upf_core", "UPF Core"),
        ("dn", "Data Network"),
        ("ran_bh", "RAN Backhaul"),
        ("ims", "IMS Server"),
    ]
    for node_id, label in nodes:
        topo.add_node(node_id, label)

    links = [
        ("ue", "gnb", 1.0, 100e6),          # 1ms, 100Mbps air interface
        ("gnb", "upf_edge", 5.0, 10e9),     # 5ms, 10Gbps fronthaul
        ("gnb", "ran_bh", 3.0, 25e9),       # 3ms, 25Gbps backhaul
        ("upf_edge", "upf_core", 8.0, 100e9),
        ("ran_bh", "ims", 4.0, 10e9),
        ("upf_core", "dn", 2.0, 400e9),
        ("upf_core", "ims", 3.0, 10e9),
        ("ims", "dn", 5.0, 10e9),
    ]
    for src, dst, delay_ms, bw_bps in links:
        topo.add_link(src, dst, delay_ms, bw_bps)

    print(f"\n{'='*60}")
    print(f"  Topology: {topo.name}")
    print(f"{'='*60}")
    summary = topo.summary()
    print(f"  Nodes   : {summary['nodes']}  Links: {summary['links']}")
    print()

    queries = [
        ("ue", "dn"),
        ("ue", "ims"),
        ("gnb", "upf_core"),
    ]
    for src, dst in queries:
        path = topo.shortest_path(src, dst)
        ecmp = topo.ecmp_paths(src, dst)
        if path:
            print(f"  {src} → {dst}")
            print(f"    Shortest path : {' → '.join(path.nodes)}")
            print(f"    Delay         : {path.total_propagation_delay_ms:.1f} ms")
            print(f"    RTT (prop.)   : {path.rtt_propagation_ms:.1f} ms")
            print(f"    Bottleneck BW : {path.bottleneck_bandwidth_bps/1e9:.1f} Gbps")
            print(f"    BDP           : {path.bdp_bytes()/1024:.1f} KB")
            print(f"    ECMP paths    : {len(ecmp)}")
        else:
            print(f"  {src} → {dst}: NO PATH")
        print()

    # Simulate a link failure and show rerouting
    print("  [Failure] Failing upf_edge → upf_core ...")
    topo.fail_link("upf_edge", "upf_core")
    path_after = topo.shortest_path("ue", "dn")
    if path_after:
        print(f"  ue → dn after failure: {' → '.join(path_after.nodes)}")
        print(f"    New delay : {path_after.total_propagation_delay_ms:.1f} ms")
    else:
        print("  ue → dn: UNREACHABLE after failure")

    topo.recover_link("upf_edge", "upf_core")
    print("  [Recovery] Link restored")
    path_recovered = topo.shortest_path("ue", "dn")
    if path_recovered:
        print(f"  ue → dn restored: {' → '.join(path_recovered.nodes)}")
    print()
    print(f"  Failure log: {topo.failure_log()}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Parser setup
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m slicenet.cli",
        description="SliceNet QoS Engine — control-plane CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- run --
    p_run = sub.add_parser("run", help="Run a simulation and print metrics")
    p_run.add_argument(
        "--scheduler", "-s",
        choices=["fifo", "priority", "wfq", "drr", "adaptive"],
        default="adaptive",
        help="Packet scheduler (default: adaptive)",
    )
    p_run.add_argument(
        "--transport", "-t",
        choices=["reno", "cubic", "bbr"],
        default="reno",
        help="TCP variant (default: reno)",
    )
    p_run.add_argument("--load", "-l", type=float, default=0.75, help="Load factor 0–1 (default: 0.75)")
    p_run.add_argument("--duration", "-d", type=int, default=2000, help="Duration in ms (default: 2000)")
    p_run.add_argument("--tick", type=float, default=1.0, help="Tick interval in ms (default: 1.0)")
    p_run.add_argument("--drain", type=int, default=10, help="Packets drained per tick (default: 10)")
    p_run.add_argument("--loss", type=float, default=0.001, help="Link loss rate (default: 0.001)")
    p_run.add_argument("--capacity", type=float, default=100.0, help="Link capacity Mbps (default: 100)")
    p_run.add_argument("--verbose", "-v", action="store_true", help="Print per-tick stats")
    p_run.set_defaults(func=_cmd_run)

    # -- demo --
    p_demo = sub.add_parser("demo", help="Run all 4 built-in demo scenarios")
    p_demo.set_defaults(func=_cmd_demo)

    # -- bench --
    p_bench = sub.add_parser("bench", help="Run the full benchmark suite")
    p_bench.set_defaults(func=_cmd_bench)

    # -- serve --
    p_serve = sub.add_parser("serve", help="Start the FastAPI REST + WebSocket server")
    p_serve.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    p_serve.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    p_serve.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    p_serve.set_defaults(func=_cmd_serve)

    # -- topo --
    p_topo = sub.add_parser("topo", help="Print 5G topology summary and path analysis")
    p_topo.set_defaults(func=_cmd_topo)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
