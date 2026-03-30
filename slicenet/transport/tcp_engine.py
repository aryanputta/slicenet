"""
TCP Transport Engine.

Simulates RFC 5681 (TCP Congestion Control):
- Slow start
- Congestion avoidance
- Fast retransmit / fast recovery
- RTT estimation (Jacobson/Karels)
- Bandwidth-delay product effects
- Head-of-line blocking
- Nagle's algorithm (conceptual)

This is a discrete-event simulation — not a raw socket layer.
"""

from __future__ import annotations
import logging
import random
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from slicenet.core.packet import Packet
from slicenet.core.constants import (
    TCP_MSS, TCP_INITIAL_CWND, TCP_MAX_CWND, TCP_SSTHRESH_INIT,
    TCP_MIN_RTO_MS
)

logger = logging.getLogger(__name__)


class TCPCongestionState:
    """
    Per-flow TCP congestion control state machine.
    Models the TCP sender side.
    """
    __slots__ = [
        "flow_id", "cwnd", "ssthresh", "in_slow_start",
        "seq_num", "unacked", "srtt_ms", "rttvar_ms", "rto_ms",
        "dup_ack_count", "in_fast_recovery", "retransmit_queue",
        "flight_size_bytes"
    ]

    def __init__(self, flow_id: str):
        self.flow_id = flow_id
        self.cwnd: float = float(TCP_INITIAL_CWND)
        self.ssthresh: float = float(TCP_SSTHRESH_INIT)
        self.in_slow_start: bool = True
        self.in_fast_recovery: bool = False
        self.seq_num: int = 0
        self.unacked: int = 0
        self.srtt_ms: float = 100.0
        self.rttvar_ms: float = 50.0
        self.rto_ms: float = 200.0
        self.dup_ack_count: int = 0
        self.flight_size_bytes: int = 0
        self.retransmit_queue: Deque[int] = deque()

    def can_send(self) -> bool:
        """True if congestion window allows another segment."""
        return self.unacked < int(self.cwnd)

    def on_ack(self, acked_bytes: int, measured_rtt_ms: float) -> None:
        """
        Process an ACK. Updates cwnd and RTT estimate.
        RFC 5681 Section 3.1
        """
        self._update_rtt(measured_rtt_ms)
        self.dup_ack_count = 0
        self.in_fast_recovery = False
        self.unacked = max(0, self.unacked - 1)
        self.flight_size_bytes = max(0, self.flight_size_bytes - acked_bytes)

        if self.in_slow_start:
            # Slow start: +1 MSS per ACK
            self.cwnd = min(self.cwnd + 1.0, TCP_MAX_CWND)
            if self.cwnd >= self.ssthresh:
                self.in_slow_start = False
                logger.debug("Flow %s exiting slow start, cwnd=%.1f", self.flow_id, self.cwnd)
        else:
            # Congestion avoidance: +1/cwnd per ACK (AIMD)
            self.cwnd = min(self.cwnd + (1.0 / self.cwnd), TCP_MAX_CWND)

    def on_duplicate_ack(self) -> bool:
        """
        Process duplicate ACK.
        Returns True if fast retransmit should be triggered (3 dup ACKs).
        RFC 5681 Section 3.2
        """
        self.dup_ack_count += 1

        if self.dup_ack_count == 3:
            # Fast retransmit
            self.ssthresh = max(self.cwnd / 2.0, 2.0)
            self.cwnd = self.ssthresh + 3.0  # inflate by 3 segments
            self.in_fast_recovery = True
            logger.debug(
                "Flow %s fast retransmit triggered, ssthresh=%.1f cwnd=%.1f",
                self.flow_id, self.ssthresh, self.cwnd
            )
            return True
        elif self.dup_ack_count > 3 and self.in_fast_recovery:
            # Fast recovery: inflate cwnd by 1 per dup ACK
            self.cwnd += 1.0
        return False

    def on_timeout(self) -> None:
        """
        RTO expiration: multiplicative decrease, re-enter slow start.
        RFC 5681 Section 3.1
        """
        logger.debug(
            "Flow %s RTO timeout, cwnd %.1f -> 1, ssthresh=%.1f",
            self.flow_id, self.cwnd, self.ssthresh
        )
        self.ssthresh = max(self.cwnd / 2.0, 2.0)
        self.cwnd = 1.0
        self.in_slow_start = True
        self.in_fast_recovery = False
        self.dup_ack_count = 0

    def _update_rtt(self, measured_rtt_ms: float) -> None:
        """Jacobson/Karels EWMA RTT estimator."""
        alpha, beta = 0.125, 0.25
        err = measured_rtt_ms - self.srtt_ms
        self.srtt_ms += alpha * err
        self.rttvar_ms += beta * (abs(err) - self.rttvar_ms)
        self.rto_ms = max(
            TCP_MIN_RTO_MS,
            min(120_000.0, self.srtt_ms + 4.0 * self.rttvar_ms)
        )

    @property
    def bandwidth_delay_product(self) -> float:
        """BDP in bytes: cwnd * MSS approximates the pipe fill."""
        return self.cwnd * TCP_MSS


class TCPEngine:
    """
    Manages TCP congestion state for all active TCP flows.
    Processes sends, ACKs, losses, and timeouts.
    """

    def __init__(self, base_rtt_ms: float = 20.0, loss_rate: float = 0.001):
        self._states: Dict[str, TCPCongestionState] = {}
        self.base_rtt_ms = base_rtt_ms
        self.loss_rate = loss_rate  # simulated link loss rate

    def register_flow(self, flow_id: str) -> TCPCongestionState:
        state = TCPCongestionState(flow_id)
        self._states[flow_id] = state
        logger.debug("TCP flow registered: %s", flow_id)
        return state

    def get_state(self, flow_id: str) -> Optional[TCPCongestionState]:
        return self._states.get(flow_id)

    def process_packet(self, packet: Packet) -> Tuple[bool, Optional[str]]:
        """
        Simulate TCP send decision.
        Returns (accepted: bool, drop_reason: Optional[str]).
        """
        state = self._states.get(packet.flow_id)
        if state is None:
            state = self.register_flow(packet.flow_id)

        # Simulate packet loss on link
        if random.random() < self.loss_rate:
            state.on_timeout()
            return False, "link_loss"

        if not state.can_send():
            return False, "cwnd_limit"

        state.seq_num += 1
        state.unacked += 1
        state.flight_size_bytes += packet.size_bytes
        return True, None

    def process_ack(self, flow_id: str, acked_bytes: int) -> None:
        """Simulate ACK arrival for a flow."""
        state = self._states.get(flow_id)
        if state is None:
            return
        # Simulate RTT with jitter
        measured_rtt = self.base_rtt_ms + random.gauss(0, self.base_rtt_ms * 0.1)
        state.on_ack(acked_bytes, max(1.0, measured_rtt))

    def get_all_states(self) -> Dict[str, TCPCongestionState]:
        return dict(self._states)

    def snapshot(self) -> List[dict]:
        """Return per-flow cwnd snapshot for telemetry."""
        return [
            {
                "flow_id": fid,
                "cwnd": s.cwnd,
                "ssthresh": s.ssthresh,
                "in_slow_start": s.in_slow_start,
                "unacked": s.unacked,
                "srtt_ms": round(s.srtt_ms, 2),
                "rto_ms": round(s.rto_ms, 2),
            }
            for fid, s in self._states.items()
        ]
