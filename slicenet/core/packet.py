"""
Packet and Flow data structures.
Designed to mirror real network packet metadata without full payload simulation.
"""

from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class Protocol(Enum):
    TCP = auto()
    UDP = auto()


class TrafficClass(Enum):
    VOIP = "voip"
    VIDEO = "video"
    IOT = "iot"
    HTTP = "http"
    BULK = "bulk"


class PacketState(Enum):
    CREATED = auto()
    QUEUED = auto()
    SCHEDULED = auto()
    TRANSMITTED = auto()
    DROPPED = auto()
    RETRANSMITTING = auto()


@dataclass
class Packet:
    """
    Represents a network packet with full QoS metadata.
    Size is in bytes. Timestamps are wall-clock seconds (float).
    """
    flow_id: str
    seq_num: int
    size_bytes: int
    protocol: Protocol
    traffic_class: TrafficClass
    slice_id: str
    priority: int

    created_at: float = field(default_factory=time.monotonic)
    enqueued_at: float = 0.0
    dequeued_at: float = 0.0
    transmitted_at: float = 0.0

    state: PacketState = PacketState.CREATED
    drop_reason: Optional[str] = None
    retransmit_count: int = 0
    is_retransmit: bool = False

    # TCP-specific fields
    ack_expected: bool = False
    acked: bool = False
    ack_received_at: float = 0.0

    @property
    def queuing_latency_ms(self) -> float:
        if self.dequeued_at > 0 and self.enqueued_at > 0:
            return (self.dequeued_at - self.enqueued_at) * 1000.0
        return 0.0

    @property
    def end_to_end_latency_ms(self) -> float:
        if self.transmitted_at > 0:
            return (self.transmitted_at - self.created_at) * 1000.0
        return 0.0

    @property
    def rtt_ms(self) -> float:
        if self.ack_received_at > 0 and self.transmitted_at > 0:
            return (self.ack_received_at - self.transmitted_at) * 1000.0
        return 0.0

    def mark_enqueued(self) -> None:
        self.enqueued_at = time.monotonic()
        self.state = PacketState.QUEUED

    def mark_dequeued(self) -> None:
        self.dequeued_at = time.monotonic()
        self.state = PacketState.SCHEDULED

    def mark_transmitted(self) -> None:
        self.transmitted_at = time.monotonic()
        self.state = PacketState.TRANSMITTED

    def mark_dropped(self, reason: str) -> None:
        self.drop_reason = reason
        self.state = PacketState.DROPPED

    def __repr__(self) -> str:
        return (
            f"Packet(flow={self.flow_id[:8]}, seq={self.seq_num}, "
            f"sz={self.size_bytes}B, {self.protocol.name}, "
            f"class={self.traffic_class.value}, state={self.state.name})"
        )


@dataclass
class Flow:
    """
    Represents an active network flow (connection).
    Tracks transport-layer state per flow.
    """
    flow_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    protocol: Protocol = Protocol.TCP
    traffic_class: TrafficClass = TrafficClass.HTTP
    slice_id: str = "best_effort"
    priority: int = 4

    # TCP state
    cwnd: float = 10.0               # congestion window (segments)
    ssthresh: float = 65535.0        # slow start threshold
    in_slow_start: bool = True
    seq_num: int = 0
    next_expected_ack: int = 0
    unacked_packets: int = 0

    # RTT estimation (Jacobson/Karels)
    srtt_ms: float = 100.0           # smoothed RTT
    rttvar_ms: float = 50.0          # RTT variance
    rto_ms: float = 200.0            # retransmission timeout

    # Throughput tracking
    bytes_sent: int = 0
    bytes_acked: int = 0
    packets_sent: int = 0
    packets_dropped: int = 0
    retransmits: int = 0

    started_at: float = field(default_factory=time.monotonic)
    active: bool = True

    def update_rtt(self, measured_rtt_ms: float) -> None:
        """Jacobson/Karels RTT estimation algorithm."""
        from slicenet.core.constants import (
            RTT_ALPHA, RTT_BETA, TCP_MIN_RTO_MS, TCP_MAX_RTO_MS
        )
        rtt_err = measured_rtt_ms - self.srtt_ms
        self.srtt_ms += RTT_ALPHA * rtt_err
        self.rttvar_ms += RTT_BETA * (abs(rtt_err) - self.rttvar_ms)
        self.rto_ms = max(
            TCP_MIN_RTO_MS,
            min(TCP_MAX_RTO_MS, self.srtt_ms + 4.0 * self.rttvar_ms)
        )

    def on_ack(self, acked_bytes: int) -> None:
        """
        Update cwnd on ACK receipt.
        Implements slow start and congestion avoidance (RFC 5681).
        """
        from slicenet.core.constants import TCP_MAX_CWND
        self.bytes_acked += acked_bytes
        if self.unacked_packets > 0:
            self.unacked_packets -= 1

        if self.in_slow_start:
            # Slow start: cwnd grows by 1 MSS per ACK
            self.cwnd = min(self.cwnd + 1.0, TCP_MAX_CWND)
            if self.cwnd >= self.ssthresh:
                self.in_slow_start = False
        else:
            # Congestion avoidance: cwnd grows by 1/cwnd per ACK
            self.cwnd = min(self.cwnd + (1.0 / self.cwnd), TCP_MAX_CWND)

    def on_loss(self) -> None:
        """
        React to packet loss (triple duplicate ACK or timeout).
        Implements multiplicative decrease (Reno).
        """
        self.ssthresh = max(self.cwnd / 2.0, 2.0)
        self.cwnd = self.ssthresh  # Fast retransmit: go to ssthresh
        self.in_slow_start = False
        self.retransmits += 1

    def on_timeout(self) -> None:
        """
        React to RTO expiration.
        Sets cwnd=1 and re-enters slow start (Reno).
        """
        self.ssthresh = max(self.cwnd / 2.0, 2.0)
        self.cwnd = 1.0
        self.in_slow_start = True
        self.retransmits += 1

    @property
    def goodput_mbps(self) -> float:
        elapsed = time.monotonic() - self.started_at
        if elapsed <= 0:
            return 0.0
        return (self.bytes_acked * 8) / (elapsed * 1e6)

    @property
    def loss_rate(self) -> float:
        if self.packets_sent == 0:
            return 0.0
        return self.packets_dropped / self.packets_sent
