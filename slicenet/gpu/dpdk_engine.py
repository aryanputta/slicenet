"""
DPDK-Style Poll-Mode Driver (PMD) Simulation.

DPDK (Data Plane Development Kit) bypasses the Linux kernel network stack
and gives applications direct access to NIC queues via poll-mode drivers.

Key concepts modeled:
  - Poll-mode: no interrupts, continuous busy-poll on RX queues
  - Burst I/O: rte_eth_rx_burst() / rte_eth_tx_burst() — N packets per call
  - Zero-copy: rte_mbuf descriptors point to DMA memory, no memcpy
  - RSS (Receive Side Scaling): distribute flows across RX queues
  - Lock-free ring buffers: rte_ring (power-of-2 SPSC/MPSC queues)

Real DPDK throughput: 100M+ pps on commodity hardware (vs ~1M pps Linux kernel)

This module:
  1. RXQueue: ring buffer with burst receive, simulating rte_ring
  2. TXQueue: burst transmit with batching
  3. RSS: flow-to-queue hash distribution
  4. PMD: poll-mode driver loop (no sleep, no yield)
  5. Comparison: PMD burst vs interrupt-driven sequential
"""

from __future__ import annotations
import logging
from collections import deque
from typing import Deque, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# DPDK-style constants
RTE_MBUF_DEFAULT_BUF_SIZE = 2176  # default rte_mbuf size (bytes)
RX_BURST_SIZE = 32                # rte_eth_rx_burst default
TX_BURST_SIZE = 32                # rte_eth_tx_burst default
RING_SIZE = 4096                  # rte_ring size (must be power of 2)


class RTERing:
    """
    Lock-free ring buffer — simulates DPDK rte_ring.

    Real rte_ring: power-of-2 capacity, CAS-based MPSC enqueue,
    masked index arithmetic for zero-branch wraparound.

    Here: deque-backed with burst enqueue/dequeue to match the API.
    """

    def __init__(self, name: str, capacity: int = RING_SIZE):
        assert (capacity & (capacity - 1)) == 0, "capacity must be power of 2"
        self.name = name
        self.capacity = capacity
        self._ring: Deque = deque(maxlen=capacity)
        self._enqueue_count: int = 0
        self._dequeue_count: int = 0
        self._drop_count: int = 0

    def enqueue_burst(self, items: list) -> int:
        """
        Enqueue up to len(items). Returns number actually enqueued.
        Drops tail if ring is full (head-drop on overflow).
        """
        available = self.capacity - len(self._ring)
        to_enqueue = items[:available]
        dropped = len(items) - len(to_enqueue)
        self._ring.extend(to_enqueue)
        self._enqueue_count += len(to_enqueue)
        self._drop_count += dropped
        return len(to_enqueue)

    def dequeue_burst(self, n: int = RX_BURST_SIZE) -> list:
        """Dequeue up to n items. Returns actual items dequeued."""
        count = min(n, len(self._ring))
        result = [self._ring.popleft() for _ in range(count)]
        self._dequeue_count += count
        return result

    @property
    def size(self) -> int:
        return len(self._ring)

    @property
    def fill_ratio(self) -> float:
        return len(self._ring) / self.capacity


class RSSDistributor:
    """
    Receive Side Scaling (RSS) flow distributor.

    RSS hashes (src_ip, dst_ip, src_port, dst_port) to select an RX queue.
    This spreads flow across CPU cores to parallelize packet processing.

    Real RSS: Toeplitz hash function in NIC hardware.
    Here: NumPy-vectorized integer hash (FNV-1a approximation).
    """

    def __init__(self, num_queues: int = 4):
        self.num_queues = num_queues
        self._queue_counts = np.zeros(num_queues, dtype=np.int64)

    def hash_flows(self, flow_ids: np.ndarray) -> np.ndarray:
        """
        Map flow_id integers to queue indices.
        Vectorized: all flows hashed simultaneously.

        GPU analog: GPU RSS implementation hashes all packet headers
        in a single kernel launch (one thread per packet).
        """
        # FNV-1a approximation (fast, good distribution)
        h = flow_ids.astype(np.uint64)
        h ^= h >> 33
        h *= np.uint64(0xFF51AFD7ED558CCD)
        h ^= h >> 33
        h *= np.uint64(0xC4CEB9FE1A85EC53)
        h ^= h >> 33
        queue_ids = (h % self.num_queues).astype(np.int32)
        np.add.at(self._queue_counts, queue_ids, 1)
        return queue_ids

    def balance_stats(self) -> Dict[int, int]:
        return {i: int(self._queue_counts[i]) for i in range(self.num_queues)}


class DPDKEngine:
    """
    DPDK-style poll-mode packet processing engine.

    Architecture:
      RX Queues (per-core)
           │
           ▼
      RSS Distributor
           │
           ▼
      PMD Worker Loop  (no sleep, no kernel calls)
           │
           ├── Burst classify (DPDK rte_flow or manual)
           ├── Burst admit
           ├── Burst forward to TX queues
           └── Metrics update
           │
           ▼
      TX Queues (per-core)

    Burst processing is the critical DPDK optimization:
    - Amortizes per-batch overhead (cache warmup, function call, ring ops)
    - Enables SIMD processing of packet descriptor arrays
    """

    def __init__(self, num_rx_queues: int = 4, burst_size: int = RX_BURST_SIZE):
        self.num_rx_queues = num_rx_queues
        self.burst_size = burst_size
        self.rss = RSSDistributor(num_queues=num_rx_queues)

        self._rx_rings = [
            RTERing(name=f"rx_q{i}", capacity=RING_SIZE)
            for i in range(num_rx_queues)
        ]
        self._tx_ring = RTERing(name="tx", capacity=RING_SIZE)

        self._pmd_iterations: int = 0
        self._pmd_empty_polls: int = 0
        self._total_rx_packets: int = 0
        self._total_tx_packets: int = 0
        self._total_drop: int = 0

        self._rx_burst_sizes: List[int] = []  # for PMD efficiency analysis

    def rx_burst(self, packets: list) -> Dict[int, list]:
        """
        Simulate NIC RX burst: distribute packets across queues via RSS.
        Returns dict: queue_id → packets.

        Real DPDK: rte_eth_rx_burst(port, queue, pkts, burst_size)
        polls the NIC descriptor ring and returns up to burst_size packets.
        """
        if not packets:
            return {}

        # RSS hash all flow IDs in one vectorized op
        flow_ids = np.array([hash(p.flow_id) for p in packets], dtype=np.int64)
        queue_ids = self.rss.hash_flows(flow_ids)

        # Group packets by queue
        queued: Dict[int, list] = {}
        for pkt, qid in zip(packets, queue_ids):
            queued.setdefault(int(qid), []).append(pkt)

        # Enqueue each group to its RX ring
        for qid, pkts in queued.items():
            enqueued = self._rx_rings[qid].enqueue_burst(pkts)
            self._total_rx_packets += enqueued
            self._total_drop += len(pkts) - enqueued

        return queued

    def pmd_poll(self, queue_id: int = 0) -> list:
        """
        Poll-mode dequeue from one RX queue.
        Returns burst of up to burst_size packets.

        Real DPDK: tight while(1) loop calling rte_eth_rx_burst().
        PMD never sleeps — uses 100% of one CPU core.
        This is the DPDK trade-off: latency for CPU core dedication.
        """
        self._pmd_iterations += 1
        ring = self._rx_rings[queue_id % self.num_rx_queues]
        burst = ring.dequeue_burst(self.burst_size)
        if not burst:
            self._pmd_empty_polls += 1
        else:
            self._rx_burst_sizes.append(len(burst))
        return burst

    def tx_burst(self, packets: list) -> int:
        """
        Batch transmit: enqueue to TX ring.
        Real DPDK: rte_eth_tx_burst() sends directly to NIC TX descriptor ring.
        """
        sent = self._tx_ring.enqueue_burst(packets)
        self._total_tx_packets += sent
        return sent

    def pmd_efficiency(self) -> float:
        """
        Fraction of PMD polls that returned at least one packet.
        High empty-poll ratio = wasted CPU (over-provisioned DPDK workers).
        Real DPDK: empty polls still consume 100% CPU (by design).
        """
        if self._pmd_iterations == 0:
            return 0.0
        return 1.0 - (self._pmd_empty_polls / self._pmd_iterations)

    def avg_burst_size(self) -> float:
        if not self._rx_burst_sizes:
            return 0.0
        return float(np.mean(self._rx_burst_sizes))

    def stats(self) -> dict:
        return {
            "total_rx": self._total_rx_packets,
            "total_tx": self._total_tx_packets,
            "total_drop": self._total_drop,
            "pmd_iterations": self._pmd_iterations,
            "pmd_empty_polls": self._pmd_empty_polls,
            "pmd_efficiency": round(self.pmd_efficiency(), 4),
            "avg_burst_size": round(self.avg_burst_size(), 2),
            "rx_ring_fills": [
                round(r.fill_ratio, 3) for r in self._rx_rings
            ],
            "rss_balance": self.rss.balance_stats(),
        }
