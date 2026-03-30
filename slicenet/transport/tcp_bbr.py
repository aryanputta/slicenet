"""
TCP BBR (Bottleneck Bandwidth and RTT) Congestion Control — BBRv1.

BBR replaces AIMD loss-based control with a model-based approach:
  - BtlBw: estimated bottleneck bandwidth (max delivery rate observed)
  - RTprop: estimated minimum RTT (path propagation delay)
  - Pacing rate = BtlBw × pacing_gain  (controls send rate)
  - cwnd = BtlBw × RTprop × cwnd_gain  (limits flight size = BDP)

Unlike Reno/CUBIC, BBR does NOT back off on packet loss — it fills the
pipe at exactly BtlBw, keeping queues near-empty (minimal bufferbloat).

Reference: Neal Cardwell et al., "BBR: Congestion-Based Congestion Control",
ACM Queue 2016; IETF draft-cardwell-iccrg-bbr-congestion-control.

State machine:
  STARTUP     → exponential BtlBw probing (gain=2/ln2 ≈ 2.89)
  DRAIN       → drain startup queue (gain=1/2.89)
  PROBE_BW    → steady-state: cycle 8-phase pacing gains [1.25, 0.75, 1×6]
  PROBE_RTT   → 4 packets in-flight for 200ms every 10s to refresh RTprop
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, Dict, Optional

# BBR constants (from the paper and Linux implementation)
BBR_HIGH_GAIN: float = 2.885   # 2 / ln(2) — startup and drain inverse gain
BBR_DRAIN_GAIN: float = 1.0 / BBR_HIGH_GAIN
BBR_CWND_GAIN: float = 2.0     # cwnd = 2 × BDP to absorb ACK compression
BBR_PROBE_BW_GAINS = (1.25, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)  # 8-phase cycle

BBR_BTLBW_FILTER_LEN = 10      # BtlBw filter window (RTT rounds)
BBR_RTPROP_FILTER_MS = 10_000.0  # RTprop filter window (10 seconds)
BBR_PROBE_RTT_INTERVAL_MS = 10_000.0  # How often to enter PROBE_RTT
BBR_PROBE_RTT_DURATION_MS = 200.0     # Duration of PROBE_RTT drain phase
BBR_MIN_CWND_PKTS = 4          # Minimum cwnd during PROBE_RTT
BBR_MSS = 1460                 # Max segment size in bytes
BBR_INITIAL_BW_BPS = 1_000_000.0  # 1 Mbps seed estimate (refined quickly)


class BBRMode(Enum):
    STARTUP = auto()
    DRAIN = auto()
    PROBE_BW = auto()
    PROBE_RTT = auto()


@dataclass
class DeliveryRate:
    """
    Tracks the delivery rate of a single ACK event.
    Mirrors Linux tcp_rate.c: delivered bytes / elapsed time.
    """
    delivered_bytes: int = 0
    elapsed_ms: float = 1.0

    @property
    def rate_bps(self) -> float:
        if self.elapsed_ms <= 0:
            return 0.0
        return (self.delivered_bytes * 8 * 1000.0) / self.elapsed_ms


@dataclass
class BBRState:
    """
    Per-connection BBR state.

    Key invariant: pacing_rate = BtlBw × pacing_gain
    BBR never sets cwnd based on loss — only BDP + headroom.
    """
    flow_id: str

    # Mode
    mode: BBRMode = BBRMode.STARTUP

    # BtlBw: windowed-max of delivery rates over last BBR_BTLBW_FILTER_LEN RTTs
    _btlbw_samples: Deque[float] = field(default_factory=deque)
    btlbw_bps: float = BBR_INITIAL_BW_BPS

    # RTprop: windowed-min of RTT samples over last BBR_RTPROP_FILTER_MS ms
    rtprop_ms: float = 100.0      # Initial estimate
    rtprop_stamp_ms: float = field(default_factory=lambda: time.monotonic() * 1000.0)

    # Gains
    pacing_gain: float = BBR_HIGH_GAIN
    cwnd_gain: float = BBR_HIGH_GAIN

    # Congestion window (packets)
    cwnd: float = 10.0

    # Delivery tracking
    delivered_bytes: int = 0
    delivered_stamp_ms: float = field(default_factory=lambda: time.monotonic() * 1000.0)
    prior_delivered_bytes: int = 0
    prior_stamp_ms: float = field(default_factory=lambda: time.monotonic() * 1000.0)

    # RTT estimation (Jacobson/Karels for RTO, BBR uses separate RTprop min)
    srtt_ms: float = 100.0
    rttvar_ms: float = 50.0
    rto_ms: float = 200.0

    # Inflight / unacked
    unacked: int = 0
    flight_size_bytes: int = 0

    # PROBE_BW phase cycling
    _probe_bw_phase: int = 0
    _phase_rtt_count: int = 0   # RTTs elapsed in current PROBE_BW phase

    # PROBE_RTT bookkeeping
    _probe_rtt_done_stamp_ms: float = 0.0
    _probe_rtt_enter_stamp_ms: float = 0.0
    _last_probe_rtt_ms: float = field(default_factory=lambda: time.monotonic() * 1000.0)

    # Startup exit detection: track consecutive RTT rounds with < 25% BtlBw growth
    _startup_rounds_no_growth: int = 0
    _prior_btlbw_bps: float = 0.0

    # Stats
    retransmits: int = 0
    dup_ack_count: int = 0
    in_fast_recovery: bool = False

    def __post_init__(self) -> None:
        now_ms = time.monotonic() * 1000.0
        self._btlbw_samples = deque(maxlen=BBR_BTLBW_FILTER_LEN)
        self.rtprop_stamp_ms = now_ms
        self._last_probe_rtt_ms = now_ms
        self.delivered_stamp_ms = now_ms
        self.prior_stamp_ms = now_ms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def can_send(self) -> bool:
        """True if flight size (unacked packets) is below cwnd."""
        return self.unacked < int(self.cwnd)

    def on_ack(self, acked_bytes: int, measured_rtt_ms: float) -> None:
        """
        Process one ACK event.
        Updates BtlBw, RTprop, gains, and cwnd.
        """
        now_ms = time.monotonic() * 1000.0

        self._update_rtt(measured_rtt_ms)
        self._update_rtprop(measured_rtt_ms, now_ms)

        # Delivery rate sample
        self.unacked = max(0, self.unacked - 1)
        self.flight_size_bytes = max(0, self.flight_size_bytes - acked_bytes)
        self.delivered_bytes += acked_bytes

        elapsed_ms = max(now_ms - self.prior_stamp_ms, 0.1)
        delta_delivered = self.delivered_bytes - self.prior_delivered_bytes
        rate = DeliveryRate(delivered_bytes=delta_delivered, elapsed_ms=elapsed_ms)

        self._update_btlbw(rate.rate_bps)
        self.prior_delivered_bytes = self.delivered_bytes
        self.prior_stamp_ms = now_ms

        self.dup_ack_count = 0
        self.in_fast_recovery = False

        self._bbr_update_model(now_ms)

    def on_duplicate_ack(self) -> bool:
        """
        Process a duplicate ACK.
        BBR does NOT reduce cwnd on dup ACKs — only triggers retransmit at 3.
        Returns True on the 3rd dup ACK (fast retransmit trigger).
        """
        self.dup_ack_count += 1
        if self.dup_ack_count == 3:
            self.retransmits += 1
            self.in_fast_recovery = True
            return True
        return False

    def on_timeout(self) -> None:
        """
        RTO expiration.
        BBR re-enters STARTUP on RTO but keeps its BtlBw/RTprop estimates.
        """
        self.retransmits += 1
        self.unacked = 0
        self.flight_size_bytes = 0
        # Re-enter STARTUP to reprofile the path
        self._enter_startup()

    def on_packet_sent(self, pkt_bytes: int) -> None:
        """Called when a packet is put into flight."""
        self.unacked += 1
        self.flight_size_bytes += pkt_bytes

    # ------------------------------------------------------------------
    # Model update (state machine)
    # ------------------------------------------------------------------

    def _bbr_update_model(self, now_ms: float) -> None:
        if self.mode == BBRMode.STARTUP:
            self._update_startup()
        elif self.mode == BBRMode.DRAIN:
            self._update_drain()
        elif self.mode == BBRMode.PROBE_BW:
            self._update_probe_bw(now_ms)
        elif self.mode == BBRMode.PROBE_RTT:
            self._update_probe_rtt(now_ms)

        self._maybe_enter_probe_rtt(now_ms)
        self._set_cwnd()

    def _enter_startup(self) -> None:
        self.mode = BBRMode.STARTUP
        self.pacing_gain = BBR_HIGH_GAIN
        self.cwnd_gain = BBR_HIGH_GAIN
        self._startup_rounds_no_growth = 0
        self._prior_btlbw_bps = self.btlbw_bps

    def _update_startup(self) -> None:
        # Exit startup when BtlBw growth < 25% for 3 consecutive RTT rounds
        if self.btlbw_bps > self._prior_btlbw_bps * 1.25:
            self._startup_rounds_no_growth = 0
        else:
            self._startup_rounds_no_growth += 1

        self._prior_btlbw_bps = self.btlbw_bps

        if self._startup_rounds_no_growth >= 3:
            self._enter_drain()

    def _enter_drain(self) -> None:
        self.mode = BBRMode.DRAIN
        self.pacing_gain = BBR_DRAIN_GAIN
        self.cwnd_gain = BBR_HIGH_GAIN

    def _update_drain(self) -> None:
        # Exit DRAIN when inflight <= BDP (queue is drained)
        bdp_bytes = self._bdp_bytes()
        if self.flight_size_bytes <= bdp_bytes:
            self._enter_probe_bw()

    def _enter_probe_bw(self) -> None:
        self.mode = BBRMode.PROBE_BW
        self._probe_bw_phase = 1  # Start at phase 1 (skip the initial 1.25 probe)
        self.pacing_gain = BBR_PROBE_BW_GAINS[self._probe_bw_phase]
        self.cwnd_gain = BBR_CWND_GAIN
        self._phase_rtt_count = 0

    def _update_probe_bw(self, now_ms: float) -> None:
        # Advance phase every RTT (approximated by each ACK)
        self._phase_rtt_count += 1
        rtts_per_phase = max(1, int(self.rtprop_ms))  # rough RTT approximation

        if self._phase_rtt_count >= rtts_per_phase:
            self._phase_rtt_count = 0
            self._probe_bw_phase = (self._probe_bw_phase + 1) % len(BBR_PROBE_BW_GAINS)
            self.pacing_gain = BBR_PROBE_BW_GAINS[self._probe_bw_phase]

    def _maybe_enter_probe_rtt(self, now_ms: float) -> None:
        """
        Enter PROBE_RTT every BBR_PROBE_RTT_INTERVAL_MS to refresh RTprop.
        Skip if we just got a fresh RTprop sample.
        """
        if self.mode == BBRMode.PROBE_RTT:
            return
        age_ms = now_ms - self._last_probe_rtt_ms
        if age_ms >= BBR_PROBE_RTT_INTERVAL_MS:
            self._enter_probe_rtt(now_ms)

    def _enter_probe_rtt(self, now_ms: float) -> None:
        self.mode = BBRMode.PROBE_RTT
        self.pacing_gain = 1.0
        self.cwnd_gain = 1.0
        self._probe_rtt_enter_stamp_ms = now_ms
        self._probe_rtt_done_stamp_ms = 0.0

    def _update_probe_rtt(self, now_ms: float) -> None:
        if self._probe_rtt_done_stamp_ms == 0.0:
            # Mark done time once inflight is reduced to minimum
            if self.unacked <= BBR_MIN_CWND_PKTS:
                self._probe_rtt_done_stamp_ms = now_ms + BBR_PROBE_RTT_DURATION_MS

        elif now_ms >= self._probe_rtt_done_stamp_ms:
            self._last_probe_rtt_ms = now_ms
            # Return to previous mode
            self._enter_probe_bw()

    # ------------------------------------------------------------------
    # BtlBw and RTprop estimation
    # ------------------------------------------------------------------

    def _update_btlbw(self, rate_bps: float) -> None:
        """Windowed max of delivery rate samples."""
        if rate_bps > 0:
            self._btlbw_samples.append(rate_bps)
        if self._btlbw_samples:
            self.btlbw_bps = max(self._btlbw_samples)

    def _update_rtprop(self, measured_rtt_ms: float, now_ms: float) -> None:
        """
        Windowed min of RTT samples (RTprop = path propagation delay).
        Expire old estimate after BBR_RTPROP_FILTER_MS.
        """
        age_ms = now_ms - self.rtprop_stamp_ms
        if measured_rtt_ms <= self.rtprop_ms or age_ms > BBR_RTPROP_FILTER_MS:
            self.rtprop_ms = measured_rtt_ms
            self.rtprop_stamp_ms = now_ms

    def _bdp_bytes(self) -> float:
        """BDP = BtlBw (bytes/s) × RTprop (s)."""
        return (self.btlbw_bps / 8.0) * (self.rtprop_ms / 1000.0)

    def _set_cwnd(self) -> None:
        """
        cwnd = cwnd_gain × BDP, in packets.
        During PROBE_RTT, cap at BBR_MIN_CWND_PKTS.
        """
        bdp_bytes = self._bdp_bytes()
        target_pkts = max(
            (bdp_bytes * self.cwnd_gain) / BBR_MSS,
            float(BBR_MIN_CWND_PKTS),
        )
        if self.mode == BBRMode.PROBE_RTT:
            self.cwnd = float(BBR_MIN_CWND_PKTS)
        else:
            self.cwnd = target_pkts

    # ------------------------------------------------------------------
    # RTT estimation (for RTO — separate from RTprop)
    # ------------------------------------------------------------------

    def _update_rtt(self, measured_rtt_ms: float) -> None:
        alpha, beta = 0.125, 0.25
        err = measured_rtt_ms - self.srtt_ms
        self.srtt_ms += alpha * err
        self.rttvar_ms += beta * (abs(err) - self.rttvar_ms)
        self.rto_ms = max(200.0, min(120_000.0, self.srtt_ms + 4.0 * self.rttvar_ms))

    # ------------------------------------------------------------------
    # Properties and diagnostics
    # ------------------------------------------------------------------

    @property
    def pacing_rate_mbps(self) -> float:
        """Current target send rate in Mbps."""
        return (self.btlbw_bps * self.pacing_gain) / 1e6

    @property
    def bandwidth_delay_product(self) -> float:
        """BDP in bytes — the pipe capacity BBR targets to fill."""
        return self._bdp_bytes()

    def snapshot(self) -> dict:
        return {
            "flow_id": self.flow_id,
            "mode": self.mode.name,
            "btlbw_mbps": round(self.btlbw_bps / 1e6, 3),
            "rtprop_ms": round(self.rtprop_ms, 3),
            "pacing_rate_mbps": round(self.pacing_rate_mbps, 3),
            "pacing_gain": round(self.pacing_gain, 4),
            "cwnd_gain": round(self.cwnd_gain, 4),
            "cwnd_pkts": round(self.cwnd, 2),
            "bdp_bytes": round(self.bandwidth_delay_product, 1),
            "inflight_pkts": self.unacked,
            "srtt_ms": round(self.srtt_ms, 2),
            "rto_ms": round(self.rto_ms, 2),
            "retransmits": self.retransmits,
            "probe_bw_phase": self._probe_bw_phase,
        }

    def __repr__(self) -> str:
        return (
            f"BBRState(flow={self.flow_id}, mode={self.mode.name}, "
            f"BtlBw={self.btlbw_bps/1e6:.2f}Mbps, RTprop={self.rtprop_ms:.1f}ms, "
            f"cwnd={self.cwnd:.1f}pkts)"
        )


class BBREngine:
    """
    BBR transport engine. Manages per-flow BBRState instances.

    Drop behaviour: BBR does not reduce cwnd on loss — it retransmits
    immediately (fast retransmit on 3 dup ACKs) and re-probes BtlBw.
    Loss only affects flight tracking, not the pacing rate.
    """

    def __init__(self, base_rtt_ms: float = 20.0, loss_rate: float = 0.001):
        self.base_rtt_ms = base_rtt_ms
        self.loss_rate = loss_rate
        self._states: Dict[str, BBRState] = {}

    def register_flow(self, flow_id: str) -> BBRState:
        state = BBRState(flow_id=flow_id)
        state.rtprop_ms = self.base_rtt_ms
        self._states[flow_id] = state
        return state

    def process_packet(self, packet) -> tuple[bool, Optional[str]]:
        """
        Admission control for an outgoing packet.
        Returns (accepted, drop_reason).
        """
        import random

        state = self._states.get(packet.flow_id)
        if state is None:
            state = self.register_flow(packet.flow_id)

        # Simulate link-level loss (loss does not reduce cwnd in BBR)
        if random.random() < self.loss_rate:
            return False, "link_loss"

        # Enforce cwnd: limit inflight packets
        if not state.can_send():
            return False, "bbr_cwnd_limit"

        state.on_packet_sent(packet.size_bytes)
        return True, None

    def process_ack(self, flow_id: str, acked_bytes: int) -> None:
        """Simulate ACK arrival with jitter-perturbed RTT."""
        import random

        state = self._states.get(flow_id)
        if state is None:
            return

        jitter = random.gauss(0, self.base_rtt_ms * 0.05)
        measured_rtt = max(1.0, self.base_rtt_ms + jitter)
        state.on_ack(acked_bytes=acked_bytes, measured_rtt_ms=measured_rtt)

    def snapshot(self) -> list[dict]:
        return [s.snapshot() for s in self._states.values()]
