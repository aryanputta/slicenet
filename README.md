# SliceNet: High-Performance QoS-Aware Network Slicing Engine

A systems-level network simulation engine implementing production-grade traffic control concepts used at Cisco, Verizon, AWS, and NVIDIA Networking. Not a toy — models real TCP/IP behavior, telecom QoS policies, and multiple packet scheduling algorithms with benchmarked results.

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
| TCP Engine | `transport/tcp_engine.py` | RFC 5681: slow start, congestion avoidance, fast retransmit, Jacobson RTT |
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
| Engine | `engine.py` | Main simulation loop, NumPy batch packet processing |

---

## Core Networking Concepts Implemented

### TCP (RFC 5681)
- **Slow start**: cwnd grows by 1 MSS/ACK until ssthresh
- **Congestion avoidance**: AIMD — cwnd += 1/cwnd per ACK
- **Fast retransmit**: 3 dup ACKs → ssthresh = cwnd/2, enter fast recovery
- **RTO timeout**: cwnd = 1, re-enter slow start
- **RTT estimation**: Jacobson/Karels SRTT + RTTVAR → RTO
- **Bandwidth-delay product**: modeled via cwnd × MSS

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

## GPU-Inspired / High-Performance Design

The engine processes packets in **NumPy batches** per tick — analogous to how NVIDIA BlueField DPU processes packet bursts in parallel on its ARM cores:

```python
sizes = np.array([p.size_bytes for p in packets], dtype=np.int32)
total_bytes = int(sizes.sum())  # batch rate accounting
```

Conceptual scaling path:
- **DPDK**: Bypass kernel, poll-mode driver, zero-copy packet I/O
- **RDMA**: Zero-copy network buffers, direct NIC-to-memory DMA
- **BlueField DPU**: Offload packet classification + scheduling to DPU ASIC
- **GPU packet processing**: Batch classify 65K packets/warp on CUDA cores

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
pip install -r requirements.txt

# Run interactive demo (all 4 scenarios)
python3 scripts/demo.py

# Run full benchmark suite + plots
python3 benchmarks/run_benchmarks.py

# Run individual scenarios
python3 benchmarks/scenarios/high_video_load.py
python3 benchmarks/scenarios/packet_loss_spike.py
python3 benchmarks/scenarios/congestion_event.py
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

- **Engineered RFC 5681-compliant congestion control** (slow start, fast retransmit, Jacobson RTT estimation) combined with RED active queue management and token bucket rate limiting, reducing p95 VoIP latency by 25% and video p95 latency by 72% vs FIFO baseline at 85% link load.

- **Designed and benchmarked 5 packet scheduling algorithms** (FIFO, Priority, WFQ, DRR, Adaptive) with p50/p95/p99 latency metrics and Jain fairness index, modeling telecom-grade traffic behavior and demonstrating adaptive mode-switching under congestion — architecture mirrors DPDK/BlueField DPU production deployments.

---

## Project Structure

```
slicenet-qos-engine/
├── slicenet/
│   ├── core/         # Packet, Flow, Slice, constants
│   ├── transport/    # TCP engine (RFC 5681), UDP engine
│   ├── scheduler/    # FIFO, Priority, WFQ, DRR, Adaptive
│   ├── congestion/   # Token bucket, Leaky bucket, RED, Controller
│   ├── traffic/      # Traffic generator (VoIP/video/IoT/HTTP/bulk)
│   ├── metrics/      # Latency histograms, throughput, fairness
│   └── engine.py     # Main simulation loop (NumPy batch processing)
├── benchmarks/
│   ├── run_benchmarks.py       # Full suite
│   ├── scenarios/              # 3 targeted scenarios
│   └── results/                # JSON + PNG output
└── scripts/
    └── demo.py                 # Interactive 4-scenario demo
```
