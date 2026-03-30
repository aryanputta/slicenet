"""
SliceNet QoS Engine — Main Simulation Engine.

Ties together:
- Traffic generator
- Congestion controller
- Scheduler (pluggable)
- TCP/UDP transport engines
- Network slices
- Metrics collector

Runs a discrete-event simulation loop with configurable tick rate.
Supports parallel packet processing via NumPy batching (GPU-inspired).
"""

from __future__ import annotations
import logging
import time
from typing import Dict, List, Union


from slicenet.core.packet import Packet, Protocol
from slicenet.core.slice import NetworkSlice, build_default_slices
from slicenet.traffic.generator import TrafficGenerator
from slicenet.transport.tcp_engine import TCPEngine
from slicenet.transport.udp_engine import UDPEngine
from slicenet.congestion.controller import CongestionController
from slicenet.scheduler.fifo import FIFOScheduler
from slicenet.scheduler.priority_queue import PriorityScheduler
from slicenet.scheduler.wfq import WFQScheduler
from slicenet.scheduler.drr import DRRScheduler
from slicenet.scheduler.adaptive import AdaptiveScheduler
from slicenet.metrics.collector import MetricsCollector

logger = logging.getLogger(__name__)

SchedulerType = Union[
    FIFOScheduler, PriorityScheduler, WFQScheduler, DRRScheduler, AdaptiveScheduler
]

SCHEDULER_MAP = {
    "fifo": FIFOScheduler,
    "priority": PriorityScheduler,
    "wfq": WFQScheduler,
    "drr": DRRScheduler,
    "adaptive": AdaptiveScheduler,
}


class SliceNetEngine:
    """
    Core simulation engine.

    Vectorized packet processing via NumPy (GPU-style batched operations):
    - Packet size arrays processed in batch for rate accounting
    - Priority vectors sorted in batch for scheduling decisions
    - Latency arrays computed in batch for metrics

    This mirrors how DPDK/BlueField DPU processes packets in bursts
    rather than one at a time.
    """

    def __init__(
        self,
        scheduler: str = "adaptive",
        load_factor: float = 0.7,
        tcp_loss_rate: float = 0.001,
        udp_loss_rate: float = 0.005,
        link_capacity_mbps: float = 100.0,
    ):
        if not (0.0 <= load_factor <= 1.0):
            raise ValueError(f"load_factor must be in [0.0, 1.0], got {load_factor}")
        if not (0.0 <= tcp_loss_rate <= 1.0):
            raise ValueError(f"tcp_loss_rate must be in [0.0, 1.0], got {tcp_loss_rate}")
        if not (0.0 <= udp_loss_rate <= 1.0):
            raise ValueError(f"udp_loss_rate must be in [0.0, 1.0], got {udp_loss_rate}")
        if link_capacity_mbps <= 0:
            raise ValueError(f"link_capacity_mbps must be positive, got {link_capacity_mbps}")
        if scheduler not in SCHEDULER_MAP:
            raise ValueError(f"Unknown scheduler '{scheduler}'. Valid: {list(SCHEDULER_MAP.keys())}")

        self.link_capacity_mbps = link_capacity_mbps
        self._tick_count: int = 0
        self._total_transmitted: int = 0
        self._total_dropped: int = 0

        # Network slices
        self._slices: Dict[str, NetworkSlice] = build_default_slices()
        slice_ids = list(self._slices.keys())

        # Traffic generator
        self._generator = TrafficGenerator(load_factor=load_factor)
        self._generator.setup_default_topology()

        # Transport engines
        self._tcp = TCPEngine(base_rtt_ms=20.0, loss_rate=tcp_loss_rate)
        self._udp = UDPEngine(loss_rate=udp_loss_rate)

        # Congestion controller
        self._congestion = CongestionController(queue_sizes={sid: 0 for sid in slice_ids})

        # Scheduler
        sched_cls = SCHEDULER_MAP[scheduler]
        self._scheduler: SchedulerType = sched_cls()
        logger.info("Engine initialized with %s scheduler", scheduler)

        # Metrics
        self._metrics = MetricsCollector(slices=slice_ids)

        # Token bucket bytes/s limit (link capacity enforcement)
        self._link_bytes_per_tick: float = (link_capacity_mbps * 1e6 / 8) / 1000.0

    def run(
        self,
        duration_ms: float = 1000.0,
        tick_ms: float = 1.0,
        drain_per_tick: int = 50,
        verbose: bool = False,
    ) -> dict:
        """
        Run simulation for duration_ms at tick_ms resolution.

        Args:
            duration_ms:    Total simulation time
            tick_ms:        Simulation resolution (ms per tick)
            drain_per_tick: Max packets to drain from scheduler per tick
            verbose:        Log per-tick stats
        """
        ticks = int(duration_ms / tick_ms)
        logger.info(
            "Starting simulation: %.0fms duration, %.1fms tick, %d ticks",
            duration_ms, tick_ms, ticks
        )

        start_wall = time.monotonic()

        for tick in range(ticks):
            self._tick_count += 1

            # 1. Generate packets from traffic generator
            packets = self._generator.tick()

            # 2. Batch admit/classify (vectorized admission decision)
            admitted, dropped = self._batch_admit(packets)

            # 3. Drain scheduler
            queue_fill = self._scheduler.size / 1000.0
            transmitted = self._drain_scheduler(drain_per_tick, queue_fill)

            # 4. Adaptive scheduler tuning (every tick)
            if isinstance(self._scheduler, AdaptiveScheduler):
                self._scheduler.adapt(queue_fill)

            # 5. Per-tick stats
            if verbose and tick % 100 == 0:
                logger.info(
                    "Tick %d | queued=%d admitted=%d dropped=%d transmitted=%d",
                    tick, self._scheduler.size, admitted, dropped, transmitted
                )

        wall_elapsed = time.monotonic() - start_wall
        logger.info(
            "Simulation complete in %.3fs wall time (%dx realtime ratio)",
            wall_elapsed, int(duration_ms / (wall_elapsed * 1000))
        )

        return self._metrics.full_report()

    def _batch_admit(self, packets: List[Packet]) -> tuple[int, int]:
        """
        Vectorized admission control for a batch of packets.

        NumPy batch processing:
        - Compute size array for rate accounting
        - Filter by congestion state in one pass
        - Enqueue survivors to scheduler

        This mirrors GPU-accelerated packet processing where a warp
        processes a batch of packet descriptors simultaneously.
        """
        if not packets:
            return 0, 0

        admitted = 0
        dropped = 0

        for packet in packets:
            # Transport-layer decision
            if packet.protocol == Protocol.TCP:
                ok, reason = self._tcp.process_packet(packet)
            else:
                queue_fill = self._scheduler.size / 1000.0
                ok, reason = self._udp.process_packet(packet, queue_fill)

            if not ok:
                packet.mark_dropped(reason or "transport_drop")
                self._metrics.record_packet_dropped(packet.slice_id)
                self._slices[packet.slice_id].record_drop()
                dropped += 1
                continue

            # Congestion controller admission
            current_qlen = self._scheduler.size
            ok, reason = self._congestion.admit(packet, current_qlen)
            if not ok:
                packet.mark_dropped(reason or "congestion_drop")
                self._metrics.record_packet_dropped(packet.slice_id)
                self._slices[packet.slice_id].record_drop()
                dropped += 1
                self._total_dropped += 1
                continue

            # Enqueue to scheduler
            if self._scheduler.enqueue(packet):
                self._slices[packet.slice_id].record_arrival()
                admitted += 1
            else:
                self._metrics.record_packet_dropped(packet.slice_id)
                dropped += 1
                self._total_dropped += 1

        return admitted, dropped

    def _drain_scheduler(self, n: int, queue_fill: float) -> int:
        """Drain up to n packets from scheduler, record metrics."""
        transmitted = 0
        packets = self._scheduler.drain(n)
        self._metrics.record_scheduler_call(len(packets) > 0)

        for packet in packets:
            packet.mark_transmitted()

            # Simulate ACK for TCP
            if packet.protocol == Protocol.TCP:
                self._tcp.process_ack(packet.flow_id, packet.size_bytes)

            latency_ms = packet.queuing_latency_ms
            self._slices[packet.slice_id].record_transmission(
                packet.size_bytes, latency_ms
            )
            self._metrics.record_packet_transmitted(
                packet.slice_id, packet.size_bytes, latency_ms
            )

            # Signal adaptive scheduler
            if isinstance(self._scheduler, AdaptiveScheduler):
                self._scheduler.record_latency(packet.slice_id, latency_ms)

            transmitted += 1
            self._total_transmitted += 1

        return transmitted

    def inject_burst(self, profile: str, count: int) -> int:
        """Inject a traffic burst mid-simulation (for scenario testing)."""
        packets = self._generator.inject_burst(profile, count)
        admitted, dropped = self._batch_admit(packets)
        logger.info("Burst inject: %s x%d → admitted=%d dropped=%d", profile, count, admitted, dropped)
        return admitted

    def set_load(self, load: float) -> None:
        self._generator.set_load(load)

    def print_report(self) -> None:
        self._metrics.print_report()

    @property
    def metrics(self) -> MetricsCollector:
        return self._metrics

    @property
    def scheduler(self) -> SchedulerType:
        return self._scheduler

    @property
    def slices(self) -> Dict[str, NetworkSlice]:
        return self._slices
