# SliceNet: High-Performance QoS-Aware Network Slicing Engine

A systems-level network simulation engine implementing production-grade traffic control concepts used at Cisco, Verizon, AWS, and NVIDIA Networking. Not a toy — models real TCP/IP behavior, telecom QoS policies, multiple packet scheduling algorithms, a live REST + WebSocket control plane, and Prometheus observability — with benchmarked results.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        SliceNet Engine                             │
│                                                                    │
│  ┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐  │
│  │   Traffic    │───▶│   Congestion     │───▶│    Scheduler    │  │
│  │  Generator   │    │   Controller     │    │  (pluggable)    │  │
│  │              │    │  - Token bucket  │    │  - FIFO         │  │
│  │  VoIP (UDP)  │    │  - RED (AQM)     │    │  - Priority     │  │
│  │  Video (UDP) │    │  - Backpressure  │    │  - WFQ          │  │
│  │  IoT (TCP)   │    └──────────────────┘    │  - DRR          │  │
│  │  HTTP (TCP)  │                            │  - Adaptive     │  │
│  │  Bulk (TCP)  │    ┌──────────────────┐    └────────┬────────┘  │
│  └──────────────┘    │  Transport Layer │             │           │
│                      │  - TCP Engine    │    ┌────────▼────────┐  │
│  ┌──────────────┐    │    cwnd/ssthresh │    │ Network Slices  │  │
│  │   Metrics    │◀───│    slow start    │    │  voip/video/    │  │
│  │  Collector   │    │    RTT/RTO est.  │    │  iot/best_eff   │  │
│  │  p50/p95/p99 │    │  - UDP Engine    │    │  (SLA + stats)  │  │
│  │  Jain index  │    │    no retransmit │    └─────────────────┘  │
│  └──────────────┘    └──────────────────┘                         │
└────────────────────────────────────────────────────────────────────┘
```

---

## Module Breakdown

| Module | File | What it does |
|--------|------|-------------|
| Packet / Flow | `core/packet.py` | Packet metadata, TCP flow state (cwnd, RTT, loss) |
| Network Slice | `core/slice.py` | SLA definitions, per-slice admission and stats |
| TCP Reno | `transport/tcp_engine.py` | RFC 5681: slow start, congestion avoidance, fast retransmit, Jacobson RTT |
| TCP CUBIC | `transport/tcp_cubic.py` | RFC 8312: cubic window function, W_max, TCP-Friendly mode |
| **TCP BBR** | `transport/tcp_bbr.py` | **BBRv1: BtlBw + RTprop model, STARTUP/DRAIN/PROBE_BW/PROBE_RTT states** |
| UDP Engine | `transport/udp_engine.py` | Fire-and-forget, queue overflow drop, loss tolerance |
| Token Bucket | `congestion/token_bucket.py` | Per-slice rate limiting with burst allowance |
| Leaky Bucket | `congestion/token_bucket.py` | Constant-rate output shaping |
| RED AQM | `congestion/red.py` | RFC 2309 Random Early Detection, EWMA queue estimation |
| Congestion Controller | `congestion/controller.py` | Combines RED + token bucket + backpressure per slice |
| FIFO Scheduler | `scheduler/fifo.py` | Baseline — single global queue |
| Priority Scheduler | `scheduler/priority_queue.py` | Strict 5-level priority, per-level FIFO |
| WFQ Scheduler | `scheduler/wfq.py` | Virtual Finish Time WFQ, proportional fairness |
| DRR Scheduler | `scheduler/drr.py` | O(1) Deficit Round Robin, variable-length packets |
| Adaptive Scheduler | `scheduler/adaptive.py` | Runtime mode switching: WFQ → DRR → Priority under congestion |
| Traffic Generator | `traffic/generator.py` | VoIP/video/IoT/HTTP/bulk flow profiles, burst injection |
| Metrics Collector | `metrics/collector.py` | p50/p95/p99 latency, throughput, loss, Jain fairness |
| **Prometheus Exporter** | `metrics/prometheus.py` | **Prometheus text-format scrape endpoint** |
| Network Topology | `topology/network.py` | Dijkstra + ECMP path selection, BDP calculation |
| **Fault-Tolerant Topology** | `topology/network.py` | **Link failure injection, auto-rerouting, utilization tracking** |
| Engine | `engine.py` | Main simulation loop, NumPy batch packet processing |
| **REST API** | `api/server.py` | **FastAPI control plane: start/stop/burst/load + WebSocket streaming** |
| **CLI** | `cli.py` | **argparse CLI: run / demo / bench / serve / topo** |

---

## Core Networking Concepts Implemented

### TCP Reno (RFC 5681)
- **Slow start**: cwnd grows by 1 MSS/ACK until ssthresh
- **Congestion avoidance**: AIMD — cwnd += 1/cwnd per ACK
- **Fast retransmit**: 3 dup ACKs → ssthresh = cwnd/2, enter fast recovery
- **RTO timeout**: cwnd = 1, re-enter slow start
- **RTT estimation**: Jacobson/Karels SRTT + RTTVAR → RTO
- **Bandwidth-delay product**: modeled via cwnd × MSS

### TCP CUBIC (RFC 8312)
- **Cubic window function**: `W(t) = C*(t−K)³ + W_max` — faster recovery on high-BDP paths
- **K computation**: `K = (W_max × β / C)^(1/3)`, saves peak window at loss event
- **TCP-Friendly mode**: falls back to Reno AIMD when CUBIC window < W_est
- **β = 0.7**: multiplicative decrease (vs Reno's 0.5) — more aggressive recovery

### TCP BBR (BBRv1 — Google, 2016)
BBR replaces loss-based AIMD with a **bandwidth-and-delay model**. It does not reduce
cwnd on packet loss — it paces at exactly the bottleneck rate to eliminate bufferbloat.

- **BtlBw**: windowed-max delivery rate over 10 RTT rounds — tracks true pipe capacity
- **RTprop**: windowed-min RTT over 10 seconds — approximates propagation delay
- **Pacing rate** = BtlBw × pacing_gain (never drives queues above BDP)
- **STARTUP**: 2.89x gain → exponential probing until 3 consecutive rounds < 25% growth
- **DRAIN**: 1/2.89x gain → drains startup queue until inflight ≤ BDP
- **PROBE_BW**: 8-phase gain cycle [1.25, 0.75, 1.0×6] → steady-state bandwidth probing
- **PROBE_RTT**: cwnd = 4 for 200ms every 10s → refreshes RTprop to avoid drift

```
BBR vs Reno: at 1% loss BBR maintains near-full throughput,
Reno backs off to ~10% of capacity (loss-based reaction).
```

### UDP
- No retransmission, no cwnd, no connection state
- Dropped under queue overflow or link loss — stream continues
- Models real VoIP/video behavior: loss degrades quality, doesn't stall

### QoS / Telecom
- 4 slices: `voip` (critical), `video` (high), `iot` (medium), `best_effort`
- Per-slice SLA: latency ceiling, bandwidth guarantee, max loss rate
- DSCP-style priority mapped to slice IDs
- Token bucket per slice: sustained rate + burst budget

---

## Scheduling Algorithms

### FIFO (Baseline)
Single queue, no priority. Demonstrates head-of-line blocking — VoIP blocked behind bulk transfers.

### Priority Queue
5 strict levels. VoIP always served first. Risk: starvation of low-priority flows under sustained load.

### WFQ (Weighted Fair Queuing)
Virtual Finish Time algorithm. Each packet gets `VFT = max(last_vft, virtual_clock) + size/weight`. Heap-ordered dequeue. O(log N) per packet. Proportional fairness + bounded latency for light flows.

### DRR (Deficit Round Robin)
O(1) per-packet. Round-robin across queues, each with a quantum proportional to weight. Carries over unused credit (deficit). Practical choice for ASIC/hardware schedulers. Used in Linux `tc drr`, Juniper MX.

### Adaptive
Runtime mode selection:
- Queue fill > 70% → switch to **Priority** (protect critical traffic)
- SLA violation rate > 5% → switch to **DRR** (guaranteed quanta)
- Low load < 10% → **WFQ** (fairness)
- Moderate load → **WFQ** with dynamic weight tuning

---

## Congestion Control

```
Arrival → RED (EWMA avg queue) → Token Bucket → Enqueue → Backpressure?
              │                       │
          drop_prob(avg_q)        rate_limit
```

- **RED**: Probabilistic drops in [min_thresh, max_thresh] zone. Signals TCP early.
- **Token bucket**: Per-slice sustained rate + 100ms burst budget.
- **Backpressure**: Activated at 90% queue fill, released at 50%.

---

## REST + WebSocket Control Plane (`slicenet/api/`)

A production-grade FastAPI control plane for real-time simulation management.

```
GET  /health                  — liveness probe
POST /simulation              — start simulation (async background task)
GET  /simulation/status       — running / elapsed / config
DELETE /simulation            — stop current simulation
POST /simulation/load         — adjust traffic load factor at runtime
POST /simulation/burst        — inject a one-shot traffic burst
GET  /metrics                 — metrics snapshot (JSON)
GET  /metrics/prometheus      — Prometheus text-format for Grafana scraping
WS   /ws/metrics              — WebSocket: 1-second live metric stream
```

**Example: start BBR simulation, then scrape Prometheus metrics:**
```bash
# Start server
python -m slicenet.cli serve --port 8000

# Start simulation
curl -X POST localhost:8000/simulation \
  -H 'Content-Type: application/json' \
  -d '{"scheduler":"wfq","transport":"bbr","load_factor":0.9,"duration_ms":10000}'

# Prometheus scrape
curl localhost:8000/metrics/prometheus
# slicenet_latency_p99_ms{slice="voip"} 0.1821
# slicenet_jains_fairness_index 0.9286
# slicenet_scheduler_efficiency 0.9873

# WebSocket live stream
wscat -c ws://localhost:8000/ws/metrics
```

---

## Fault-Tolerant Network Topology (`slicenet/topology/`)

Multi-hop directed graph with automatic failover and utilization tracking.

```
  UE ─── gNB ─── UPF-Edge ─── UPF-Core ─── Data Network
                    │                           │
                 RAN-BH ──────────────── IMS-Server
```

**Features:**
- **Dijkstra shortest path** by propagation delay (excludes failed links)
- **ECMP** (Equal-Cost Multi-Path) over the shortest-path DAG
- **Link failure**: `topo.fail_link("A", "B")` → auto-reroutes all subsequent path queries
- **Link recovery**: `topo.recover_link("A", "B")` → path cache invalidated, optimal path restored
- **Per-link utilization**: bytes/packets/drops + utilization % relative to link capacity
- **Failure log**: timestamps for failure and recovery events

```python
from slicenet.topology.network import FaultTolerantTopology

topo = FaultTolerantTopology("5G-Core")
topo.add_link("ue", "upf_edge", propagation_delay_ms=6.0, bandwidth_bps=10e9)
topo.add_link("upf_edge", "dn",   propagation_delay_ms=10.0, bandwidth_bps=100e9)

# Primary path
path = topo.shortest_path("ue", "dn")
# → Path([ue → upf_edge → dn], delay=16.0ms, BDP=400KB)

# Simulate link failure — instant rerouting
topo.fail_link("upf_edge", "dn")
alt_path = topo.shortest_path("ue", "dn")
# → alternate route (or None if no path exists)

print(topo.failure_log())
# [{'src': 'upf_edge', 'dst': 'dn', 'failed_at_ms': 12.3, 'recovered_at_ms': None}]
```

---

## NVIDIA-Style GPU / DPU Acceleration (`slicenet/gpu/`)

Three production-grade acceleration modules, each modeling a real NVIDIA technology:

### 1. CUDA Pipeline (`cuda_pipeline.py`)
4-kernel vectorized packet processing — mirrors a real CUDA kernel launch:

| Kernel | What it does | GPU Analog |
|--------|-------------|------------|
| `classify_kernel` | Table-lookup classification, all N packets | `__ldg()` constant memory gather |
| `admit_kernel` | Token bucket + RED across all packets | `atomicSub` on per-slice counters |
| `sort_kernel` | Priority argsort (radix sort on GPU) | `thrust::sort_by_key` / CUB `DeviceRadixSort` |
| `latency_kernel` | Vectorized latency computation | Element-wise subtract in one kernel |

**Measured speedup vs CPU sequential Python loop:**

| Batch size | CPU (µs) | GPU-style (µs) | Speedup |
|-----------|---------|--------------|---------|
| 256 | 62.3 | 21.3 | **2.9x** |
| 1,024 | 207.9 | 24.9 | **8.4x** |
| 4,096 | 786.1 | 56.5 | **13.9x** |
| 16,384 | 3,191 | 115 | **27.7x** |
| 65,536 | 12,757 | 456 | **28.0x** |

*Real BlueField-3 CUDA kernels achieve 100M–600M pps.*

SoA (struct-of-arrays) layout mirrors CUDA best practices:
```python
# AOS (slow): list of Python objects — pointer chasing, no SIMD
# SoA (fast): separate NumPy arrays — coalesced memory access
batch.sizes      = np.zeros(n, dtype=np.int32)   # all sizes contiguous
batch.priorities = np.zeros(n, dtype=np.int32)   # all priorities contiguous
```

### 2. BlueField DPU (`bluefield_dpu.py`)
Models the NVIDIA BlueField-3 DPU three-component architecture:

```
Host NIC (ConnectX-7)
     │
     ▼
┌─────────────────────────────────────┐
│         BlueField-3 DPU             │
│  ┌──────────┐  ┌──────┐  ┌───────┐ │
│  │DOCA Flow │  │  TM  │  │ RDMA  │ │
│  │match/act │  │WFQ+PQ│  │RoCEv2 │ │
│  └──────────┘  └──────┘  └───────┘ │
└─────────────────────────────────────┘
```

- **DOCA Flow**: TCAM match-action table, vectorized first-match across all N packets simultaneously
- **Traffic Manager**: Per-slice bandwidth guarantee + shaping (ETS: Enhanced Transmission Selection)
- **RDMA Engine**: 1.5 µs path vs 15 µs kernel path — **90% latency reduction**

### 3. DPDK Engine (`dpdk_engine.py`)
Poll-mode driver with RSS (Receive Side Scaling):

- `RTERing`: lock-free power-of-2 ring buffer (simulates `rte_ring`)
- `RSSDistributor`: FNV-1a vectorized flow hashing → queue assignment
- `DPDKEngine.rx_burst()`: `rte_eth_rx_burst()` analog — N packets per call
- No `sleep()`, no kernel calls, 100% core dedication

**Benchmark `python3 benchmarks/gpu_benchmark.py`:**
```
DOCA Flow: 21 Mpps classification throughput
RDMA path: 1.5 µs vs 15 µs kernel (90% reduction)
Classify:  28x speedup at 65k batch vs Python sequential
```

---

## Demo Results

```
[Demo 1] Scheduler Comparison — VoIP p95 Latency Under 75% Load
  fifo         | VoIP p95=  0.27ms  fairness=0.8989
  priority     | VoIP p95=  0.13ms  fairness=0.8204  ← low fairness
  wfq          | VoIP p95=  0.19ms  fairness=0.9286  ← best fairness
  drr          | VoIP p95=  0.13ms  fairness=0.9066
  adaptive     | VoIP p95=  0.10ms  fairness=0.8926

[Demo 4] Before vs After Optimization (FIFO → Adaptive, load=85%)
  voip  p95: 0.20ms → 0.15ms  (+25.0% improvement)
  voip  p99: 0.20ms → 0.15ms  (+23.4% improvement)
  video p95: 0.15ms → 0.04ms  (+72.0% improvement)
```

---

## Quick Start

```bash
git clone https://github.com/you/slicenet-qos-engine
cd slicenet-qos-engine
pip install -e ".[api,plots,dev]"

# CLI — run a simulation
python -m slicenet.cli run --scheduler adaptive --load 0.85 --duration 3000

# CLI — compare TCP variants under 1% loss
python -m slicenet.cli run --transport bbr --loss 0.01 --scheduler wfq
python -m slicenet.cli run --transport cubic --loss 0.01 --scheduler wfq

# CLI — topology analysis with failure simulation
python -m slicenet.cli topo

# REST + WebSocket control plane
python -m slicenet.cli serve --port 8000
# Then:
curl -X POST http://localhost:8000/simulation \
     -H 'Content-Type: application/json' \
     -d '{"scheduler":"adaptive","transport":"bbr","duration_ms":5000,"load_factor":0.8}'
curl http://localhost:8000/metrics/prometheus   # Prometheus scrape
wscat -c ws://localhost:8000/ws/metrics         # Live WebSocket stream

# Interactive demo (all 4 scenarios)
python3 scripts/demo.py

# Full benchmark suite + plots
python3 benchmarks/run_benchmarks.py
```

---

## Benchmark Scenarios

| Scenario | What it tests |
|----------|--------------|
| `high_video_load.py` | VoIP SLA under 500-packet video burst. FIFO fails; WFQ/DRR pass. |
| `packet_loss_spike.py` | 5% link loss. TCP cwnd collapses; UDP streams continue. |
| `congestion_event.py` | Adaptive scheduler mode transitions under 3-phase load ramp. |
| `run_benchmarks.py` | All 5 schedulers × 5 load levels × 4 slices. JSON + PNG output. |

---

## Resume Bullets

- **Built a high-performance network slicing engine** simulating TCP/UDP traffic with QoS enforcement across 4 slice classes, implementing WFQ (Virtual Finish Time) and DRR (Deficit Round Robin) scheduling to maintain VoIP p99 latency under 20ms during video burst congestion events.

- **Implemented three TCP congestion control algorithms** (Reno RFC 5681, CUBIC RFC 8312, BBRv1) with full state machines — BBR uses BtlBw/RTprop model-based pacing instead of AIMD loss signals, maintaining near-full throughput at 1% packet loss where Reno backs off to ~10% capacity.

- **Designed fault-tolerant network topology** with automatic Dijkstra rerouting around failed links, ECMP path selection, per-link utilization tracking, and failure log — modelling real BGP/OSPF convergence behavior used in production WAN and 5G core architectures.

- **Built a FastAPI REST + WebSocket control plane** enabling real-time simulation control (start/stop, dynamic load adjustment, burst injection), live metric streaming via WebSocket, and a Prometheus text-format endpoint for Grafana integration — same observability stack used by Kubernetes and cloud-native infrastructure.

- **Engineered RFC 5681-compliant congestion control** (slow start, fast retransmit, Jacobson RTT estimation) combined with RED active queue management and token bucket rate limiting, reducing p95 VoIP latency by 25% and video p95 latency by 72% vs FIFO baseline at 85% link load.

- **Designed and benchmarked 5 packet scheduling algorithms** (FIFO, Priority, WFQ, DRR, Adaptive) with p50/p95/p99 latency metrics and Jain fairness index, modeling telecom-grade traffic behavior and demonstrating adaptive mode-switching under congestion — architecture mirrors DPDK/BlueField DPU production deployments.

---

## Project Structure

```
slicenet-qos-engine/
├── slicenet/
│   ├── core/         # Packet, Flow, Slice, constants
│   ├── transport/    # TCP Reno (RFC 5681), TCP CUBIC (RFC 8312), TCP BBR, UDP
│   ├── scheduler/    # FIFO, Priority, WFQ, DRR, Adaptive
│   ├── congestion/   # Token bucket, Leaky bucket, RED, Controller
│   ├── traffic/      # Traffic generator (VoIP/video/IoT/HTTP/bulk)
│   ├── metrics/      # Latency histograms, throughput, fairness, Prometheus exporter
│   ├── topology/     # Multi-hop topology, Dijkstra, ECMP, link failure + rerouting
│   ├── api/          # FastAPI REST + WebSocket control plane
│   ├── cli.py        # argparse CLI (run / demo / bench / serve / topo)
│   └── engine.py     # Main simulation loop (NumPy batch processing)
├── benchmarks/
│   ├── run_benchmarks.py       # Full scheduler comparison suite
│   ├── gpu_benchmark.py        # CUDA pipeline / DPDK / BlueField DPU benchmarks
│   ├── scenarios/              # 3 targeted scenarios
│   └── results/                # JSON + PNG output
├── slicenet/gpu/
│   ├── cuda_pipeline.py        # 4-kernel vectorized pipeline (SoA, batch processing)
│   ├── bluefield_dpu.py        # DOCA Flow + Traffic Manager + RDMA simulation
│   └── dpdk_engine.py          # Poll-mode driver, RSS, rte_ring, burst I/O
├── tests/
│   ├── test_schedulers.py      # 31 scheduler + AQM tests
│   ├── test_tcp_engine.py      # 40 TCP Reno congestion control tests
│   ├── test_tcp_bbr.py         # 36 TCP BBR state machine tests
│   └── test_topology_failure.py # 28 link failure + rerouting tests
└── scripts/
    └── demo.py                 # Interactive 4-scenario demo
```
