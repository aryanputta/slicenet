"""
TCP CUBIC Congestion Control (RFC 8312).

CUBIC replaces the linear TCP Reno window growth with a cubic function of
time since the last congestion event, enabling faster recovery on high-BDP paths
while remaining TCP-Friendly (RFC 5033) under moderate RTTs.

Window growth function:
    W(t) = C * (t - K)^3 + W_max

where:
    C       = 0.4  (scaling constant, RFC 8312 Section 5)
    W_max   = window size at last congestion event (segments)
    K       = (W_max * beta / C)^(1/3)
    beta    = 0.7  (multiplicative decrease factor, RFC 8312)
    t       = time elapsed since last congestion event (seconds)

TCP-Friendly mode: if the standard Reno-estimated window W_est exceeds
the CUBIC window, CUBIC falls back to W_est to remain TCP-friendly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


# RFC 8312 constants
CUBIC_C: float = 0.4
CUBIC_BETA: float = 0.7
CUBIC_MSS: int = 1460       # bytes
CUBIC_MAX_CWND: float = 65535.0
CUBIC_MIN_CWND: float = 2.0
TCP_MIN_RTO_MS: float = 200.0
TCP_MAX_RTO_MS: float = 120_000.0


@dataclass
class CUBICState:
    """
    Per-connection TCP CUBIC congestion control state.

    Tracks both the CUBIC window function and the TCP-Friendly (Reno-equivalent)
    window estimate for the friendly-mode comparison.
    """
    flow_id: str

    # Congestion window (in segments)
    cwnd: float = 10.0
    ssthresh: float = CUBIC_MAX_CWND

    # CUBIC-specific
    w_max: float = 0.0          # cwnd at last congestion event (segments)
    k: float = 0.0              # K parameter for cubic root
    t_epoch: float = field(default_factory=time.monotonic)  # time of last loss

    # TCP-Friendly Reno estimate
    w_est: float = 10.0         # estimated Reno cwnd

    # Slow start flag
    in_slow_start: bool = True
    in_fast_recovery: bool = False

    # RTT estimation (Jacobson/Karels)
    srtt_ms: float = 100.0
    rttvar_ms: float = 50.0
    rto_ms: float = 200.0

    # Duplicate ACK tracking
    dup_ack_count: int = 0

    # Stats
    unacked: int = 0
    flight_size_bytes: int = 0
    retransmits: int = 0

    def __post_init__(self) -> None:
        self.t_epoch = time.monotonic()
        self.w_est = self.cwnd

    def _compute_k(self) -> float:
        """K = cbrt(W_max * beta / C)."""
        return (self.w_max * CUBIC_BETA / CUBIC_C) ** (1.0 / 3.0)

    def cubic_window(self, t: float) -> float:
        """
        CUBIC window function at elapsed time t seconds since last congestion.
        W_cubic(t) = C * (t - K)^3 + W_max
        """
        return CUBIC_C * (t - self.k) ** 3 + self.w_max

    def _reno_friendly_increment(self) -> float:
        """
        Estimated Reno cwnd increment per ACK during congestion avoidance.
        W_est grows by 3 * beta / (2 - beta) * (1 / cwnd) per ACK (RFC 8312 §4).
        Simplification: 1/cwnd per ACK (standard AIMD Reno rate).
        """
        reno_inc = 3.0 * CUBIC_BETA / (2.0 - CUBIC_BETA)
        return reno_inc / self.cwnd

    def can_send(self) -> bool:
        return self.unacked < int(self.cwnd)

    def on_ack(self, acked_bytes: int, measured_rtt_ms: float) -> None:
        """
        Process a cumulative ACK.
        Updates cwnd via CUBIC or TCP-Friendly mode, whichever is larger.
        """
        self._update_rtt(measured_rtt_ms)
        self.dup_ack_count = 0
        self.in_fast_recovery = False
        self.unacked = max(0, self.unacked - 1)
        self.flight_size_bytes = max(0, self.flight_size_bytes - acked_bytes)

        if self.in_slow_start:
            # Slow start: +1 MSS per ACK
            self.cwnd = min(self.cwnd + 1.0, CUBIC_MAX_CWND)
            self.w_est = self.cwnd
            if self.cwnd >= self.ssthresh:
                self.in_slow_start = False
                # Initialize CUBIC epoch on slow-start exit
                self.w_max = self.cwnd
                self.k = self._compute_k()
                self.t_epoch = time.monotonic()
            return

        # Congestion avoidance
        t = time.monotonic() - self.t_epoch

        w_cubic = self.cubic_window(t)

        # TCP-Friendly estimate: Reno-like linear growth
        self.w_est += self._reno_friendly_increment()

        if w_cubic < self.w_est:
            # TCP-Friendly mode: use Reno estimate
            self.cwnd = min(self.w_est, CUBIC_MAX_CWND)
        else:
            # CUBIC mode: move toward w_cubic
            # Increment per ACK to reach w_cubic from current cwnd
            target = min(w_cubic, CUBIC_MAX_CWND)
            increment = (target - self.cwnd) / self.cwnd
            self.cwnd = min(self.cwnd + max(increment, 0.0), CUBIC_MAX_CWND)

    def on_duplicate_ack(self) -> bool:
        """
        Process a duplicate ACK.
        Returns True if fast retransmit should be triggered (3 dup ACKs).
        RFC 8312 Section 5: same fast-retransmit trigger as Reno.
        """
        self.dup_ack_count += 1
        if self.dup_ack_count == 3:
            self._on_cubic_loss()
            self.in_fast_recovery = True
            # Inflate cwnd by 3 as in Reno fast recovery
            self.cwnd = self.ssthresh + 3.0
            self.retransmits += 1
            return True
        elif self.dup_ack_count > 3 and self.in_fast_recovery:
            self.cwnd += 1.0
        return False

    def on_timeout(self) -> None:
        """
        RTO expiration: reset to slow start, set ssthresh per CUBIC.
        RFC 8312 Section 5.
        """
        self.w_max = self.cwnd
        self._on_cubic_loss()
        self.cwnd = CUBIC_MIN_CWND
        self.w_est = CUBIC_MIN_CWND
        self.in_slow_start = True
        self.in_fast_recovery = False
        self.dup_ack_count = 0
        self.retransmits += 1

    def _on_cubic_loss(self) -> None:
        """
        Update CUBIC state on any loss event.
        W_max = cwnd (save peak); ssthresh = max(cwnd * beta, 2).
        """
        self.w_max = self.cwnd
        self.ssthresh = max(self.cwnd * CUBIC_BETA, CUBIC_MIN_CWND)
        self.k = self._compute_k()
        self.t_epoch = time.monotonic()

    def _update_rtt(self, measured_rtt_ms: float) -> None:
        alpha, beta = 0.125, 0.25
        err = measured_rtt_ms - self.srtt_ms
        self.srtt_ms += alpha * err
        self.rttvar_ms += beta * (abs(err) - self.rttvar_ms)
        self.rto_ms = max(
            TCP_MIN_RTO_MS,
            min(TCP_MAX_RTO_MS, self.srtt_ms + 4.0 * self.rttvar_ms),
        )

    @property
    def bandwidth_delay_product(self) -> float:
        """BDP approximation in bytes using current cwnd and SRTT."""
        return self.cwnd * CUBIC_MSS

    def snapshot(self) -> dict:
        t = time.monotonic() - self.t_epoch
        return {
            "flow_id": self.flow_id,
            "cwnd": round(self.cwnd, 2),
            "ssthresh": round(self.ssthresh, 2),
            "w_max": round(self.w_max, 2),
            "w_est": round(self.w_est, 2),
            "k": round(self.k, 4),
            "t_since_loss_s": round(t, 4),
            "cubic_window": round(self.cubic_window(t), 2),
            "in_slow_start": self.in_slow_start,
            "in_fast_recovery": self.in_fast_recovery,
            "srtt_ms": round(self.srtt_ms, 2),
            "rto_ms": round(self.rto_ms, 2),
            "unacked": self.unacked,
            "retransmits": self.retransmits,
        }

    def __repr__(self) -> str:
        return (
            f"CUBICState(flow={self.flow_id}, cwnd={self.cwnd:.1f}, "
            f"ssthresh={self.ssthresh:.1f}, w_max={self.w_max:.1f}, "
            f"ss={self.in_slow_start})"
        )


def compare_cubic_vs_reno(
    duration_rtts: int = 50,
    base_rtt_ms: float = 50.0,
    loss_at_rtt: int = 20,
) -> dict:
    """
    Simulate CUBIC and Reno over `duration_rtts` RTT rounds and compare
    congestion window evolution.

    loss_at_rtt: which RTT round to inject a single loss event.

    Returns a dict with per-round cwnd lists for both algorithms.
    """
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from slicenet.transport.tcp_engine import TCPCongestionState

    cubic = CUBICState(flow_id="cubic-test")
    reno = TCPCongestionState(flow_id="reno-test")

    cubic_cwnds: list[float] = []
    reno_cwnds: list[float] = []

    for rtt_round in range(duration_rtts):
        # Simulate one ACK per RTT for simplicity (window already open)
        if rtt_round == loss_at_rtt:
            cubic.on_timeout()
            reno.on_timeout()
        else:
            cubic.on_ack(acked_bytes=CUBIC_MSS, measured_rtt_ms=base_rtt_ms)
            reno.on_ack(acked_bytes=CUBIC_MSS, measured_rtt_ms=base_rtt_ms)

        cubic_cwnds.append(round(cubic.cwnd, 2))
        reno_cwnds.append(round(reno.cwnd, 2))

    return {
        "rtts": list(range(duration_rtts)),
        "cubic_cwnd": cubic_cwnds,
        "reno_cwnd": reno_cwnds,
        "loss_at_rtt": loss_at_rtt,
        "cubic_final_cwnd": cubic.cwnd,
        "reno_final_cwnd": reno.cwnd,
    }


if __name__ == "__main__":
    result = compare_cubic_vs_reno()
    print(f"After {len(result['rtts'])} RTT rounds (loss at RTT {result['loss_at_rtt']}):")
    print(f"  CUBIC final cwnd: {result['cubic_final_cwnd']:.2f}")
    print(f"  Reno  final cwnd: {result['reno_final_cwnd']:.2f}")
    print(f"  CUBIC advantage:  {result['cubic_final_cwnd'] / max(result['reno_final_cwnd'], 1):.2f}x")
