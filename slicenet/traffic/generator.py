"""
Traffic Generator.

Produces realistic network flow mixtures:
- VoIP: small UDP packets, constant bit rate (CBR), tight timing
- Video: larger UDP bursts, variable bit rate (VBR), periodic I-frames
- IoT: small TCP/UDP packets, low rate, bursty
- Bulk HTTP/data: large TCP transfers, high throughput demand

Includes burst mode and steady-state mode.
Implements bandwidth-delay product concepts for TCP flow pacing.
"""

from __future__ import annotations
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

from slicenet.core.packet import Packet, Protocol, TrafficClass
from slicenet.core.constants import (
    TCP_MSS, MTU_BYTES, SLICE_PRIORITY_MAP, TRAFFIC_SLICE_MAP
)

logger = logging.getLogger(__name__)


@dataclass
class FlowProfile:
    """Defines traffic characteristics for a flow type."""
    traffic_class: TrafficClass
    protocol: Protocol
    packet_size_bytes: int          # Average packet size
    packet_size_jitter: int         # ±jitter in bytes
    inter_arrival_ms: float         # Average inter-packet gap (ms)
    inter_arrival_jitter_ms: float  # ±jitter (ms)
    burst_size: int = 1             # Packets per burst
    burst_interval_ms: float = 0.0  # ms between bursts (0 = steady)


# Realistic traffic profiles based on ITU-T G.711, H.264, MQTT specs
FLOW_PROFILES: Dict[str, FlowProfile] = {
    "voip": FlowProfile(
        traffic_class=TrafficClass.VOIP,
        protocol=Protocol.UDP,
        packet_size_bytes=172,        # G.711 20ms frame + headers
        packet_size_jitter=8,
        inter_arrival_ms=20.0,        # 50 pps
        inter_arrival_jitter_ms=2.0,
        burst_size=1,
    ),
    "video": FlowProfile(
        traffic_class=TrafficClass.VIDEO,
        protocol=Protocol.UDP,
        packet_size_bytes=1200,       # H.264 slice (MTU-constrained)
        packet_size_jitter=200,
        inter_arrival_ms=4.0,         # ~250 pps for 2.4 Mbps video
        inter_arrival_jitter_ms=1.0,
        burst_size=8,                 # I-frame burst
        burst_interval_ms=33.0,       # 30fps video frame interval
    ),
    "iot": FlowProfile(
        traffic_class=TrafficClass.IOT,
        protocol=Protocol.TCP,
        packet_size_bytes=128,        # MQTT sensor payload
        packet_size_jitter=64,
        inter_arrival_ms=100.0,       # 10 samples/sec
        inter_arrival_jitter_ms=20.0,
        burst_size=1,
    ),
    "http": FlowProfile(
        traffic_class=TrafficClass.HTTP,
        protocol=Protocol.TCP,
        packet_size_bytes=TCP_MSS,    # Full MSS TCP segment
        packet_size_jitter=0,
        inter_arrival_ms=0.5,         # High throughput
        inter_arrival_jitter_ms=0.1,
        burst_size=10,
        burst_interval_ms=50.0,
    ),
    "bulk": FlowProfile(
        traffic_class=TrafficClass.BULK,
        protocol=Protocol.TCP,
        packet_size_bytes=TCP_MSS,
        packet_size_jitter=0,
        inter_arrival_ms=0.1,         # Background bulk transfer
        inter_arrival_jitter_ms=0.05,
        burst_size=64,
        burst_interval_ms=100.0,
    ),
}


class Flow:
    """Active flow state for the generator."""

    def __init__(self, profile_name: str, profile: FlowProfile):
        self.flow_id = str(uuid.uuid4())[:16]
        self.profile_name = profile_name
        self.profile = profile
        self.slice_id = TRAFFIC_SLICE_MAP.get(profile_name, "best_effort")
        self.priority = SLICE_PRIORITY_MAP.get(self.slice_id, 4)
        self.seq_num: int = 0
        self.active: bool = True
        self._next_send_at: float = time.monotonic()
        self._burst_remaining: int = 0

    def next_packet(self, now: float) -> Optional[Packet]:
        """Generate the next packet if due, else return None."""
        if now < self._next_send_at:
            return None

        profile = self.profile

        # Packet size with jitter
        size = max(64, profile.packet_size_bytes + random.randint(
            -profile.packet_size_jitter, profile.packet_size_jitter
        ))

        pkt = Packet(
            flow_id=self.flow_id,
            seq_num=self.seq_num,
            size_bytes=size,
            protocol=profile.protocol,
            traffic_class=profile.traffic_class,
            slice_id=self.slice_id,
            priority=self.priority,
        )
        self.seq_num += 1

        # Schedule next packet
        jitter = random.gauss(0, profile.inter_arrival_jitter_ms) / 1000.0
        self._next_send_at = now + (profile.inter_arrival_ms / 1000.0) + jitter

        return pkt


class TrafficGenerator:
    """
    Produces mixed traffic across all flow types.

    Supports:
    - Steady-state traffic generation
    - Burst injection (for congestion testing)
    - Per-slice flow multiplicity
    - Configurable load level (0.0 to 1.0)
    """

    def __init__(self, load_factor: float = 0.5):
        """
        Args:
            load_factor: 0.0 = minimal traffic, 1.0 = full link saturation
        """
        self.load_factor = load_factor
        self._flows: List[Flow] = []
        self._total_generated: int = 0

    def add_flows(
        self, profile_name: str, count: int = 1
    ) -> None:
        profile = FLOW_PROFILES[profile_name]
        for _ in range(count):
            self._flows.append(Flow(profile_name, profile))
        logger.info("Added %d %s flows (total flows: %d)", count, profile_name, len(self._flows))

    def setup_default_topology(self) -> None:
        """
        Standard mixed topology:
        - 5 VoIP flows (CBR, critical)
        - 3 video flows (VBR, high)
        - 10 IoT flows (low rate)
        - 2 HTTP flows (medium)
        - 1 bulk transfer (background)
        """
        self.add_flows("voip", 5)
        self.add_flows("video", 3)
        self.add_flows("iot", 10)
        self.add_flows("http", 2)
        self.add_flows("bulk", 1)
        logger.info("Default topology configured: %d total flows", len(self._flows))

    def inject_burst(self, profile_name: str, count: int = 100) -> List[Packet]:
        """
        Inject a burst of packets immediately (for congestion testing).
        All packets have created_at = now.
        """
        profile = FLOW_PROFILES.get(profile_name)
        if profile is None:
            logger.error("Unknown profile: %s", profile_name)
            return []

        slice_id = TRAFFIC_SLICE_MAP.get(profile_name, "best_effort")
        priority = SLICE_PRIORITY_MAP.get(slice_id, 4)
        flow_id = str(uuid.uuid4())[:16]

        packets = []
        for i in range(count):
            size = max(64, profile.packet_size_bytes + random.randint(
                -profile.packet_size_jitter, profile.packet_size_jitter
            ))
            pkt = Packet(
                flow_id=flow_id,
                seq_num=i,
                size_bytes=size,
                protocol=profile.protocol,
                traffic_class=profile.traffic_class,
                slice_id=slice_id,
                priority=priority,
            )
            packets.append(pkt)

        self._total_generated += count
        logger.info("Burst injected: %d %s packets", count, profile_name)
        return packets

    def tick(self) -> List[Packet]:
        """
        Advance simulation clock and collect all due packets.
        Call once per simulation tick.
        """
        now = time.monotonic()
        packets: List[Packet] = []

        for flow in self._flows:
            if not flow.active:
                continue
            pkt = flow.next_packet(now)
            if pkt is not None:
                # Apply load factor: randomly suppress at low load
                if random.random() < self.load_factor:
                    packets.append(pkt)
                    self._total_generated += 1

        return packets

    def generate_n(self, n: int, profile_name: str = "video") -> List[Packet]:
        """Generate exactly N packets from a given profile (for benchmarks)."""
        return self.inject_burst(profile_name, n)

    def set_load(self, load_factor: float) -> None:
        self.load_factor = max(0.0, min(1.0, load_factor))
        logger.info("Load factor set to %.2f", self.load_factor)

    @property
    def total_generated(self) -> int:
        return self._total_generated

    @property
    def flow_count(self) -> int:
        return len(self._flows)
