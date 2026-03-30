"""
Unit tests for TCP congestion control engine.

Covers:
- Slow start cwnd growth (doubles per RTT)
- Congestion avoidance linear growth after ssthresh
- Fast retransmit triggered by 3 duplicate ACKs
- RTO timeout resets cwnd to 1 and re-enters slow start
- Jacobson/Karels RTT estimation (SRTT and RTTVAR convergence)
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slicenet.transport.tcp_engine import TCPCongestionState, TCPEngine
from slicenet.core.constants import TCP_INITIAL_CWND, TCP_MSS, TCP_MIN_RTO_MS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(flow_id: str = "test-flow") -> TCPCongestionState:
    return TCPCongestionState(flow_id=flow_id)


# ---------------------------------------------------------------------------
# Slow start tests
# ---------------------------------------------------------------------------

class TestSlowStart:

    def test_cwnd_starts_at_initial_value(self):
        s = make_state()
        assert s.cwnd == float(TCP_INITIAL_CWND)
        assert s.in_slow_start is True

    def test_cwnd_increments_by_one_per_ack_in_slow_start(self):
        s = make_state()
        s.ssthresh = 100.0  # high so we stay in slow start
        initial = s.cwnd
        s.on_ack(acked_bytes=TCP_MSS, measured_rtt_ms=50.0)
        assert s.cwnd == initial + 1.0

    def test_cwnd_approximately_doubles_per_rtt(self):
        """
        Simulating one ACK per packet in a window:
        After cwnd ACKs (one full RTT), cwnd should roughly double.
        """
        s = make_state()
        s.ssthresh = 1000.0
        window = int(s.cwnd)
        cwnd_before = s.cwnd
        for _ in range(window):
            s.on_ack(acked_bytes=TCP_MSS, measured_rtt_ms=50.0)
        # cwnd should have grown by ~window (one per ACK) ≈ doubled
        assert s.cwnd >= cwnd_before * 1.5

    def test_slow_start_exits_when_cwnd_reaches_ssthresh(self):
        s = make_state()
        s.cwnd = 14.0
        s.ssthresh = 15.0
        s.in_slow_start = True
        s.on_ack(acked_bytes=TCP_MSS, measured_rtt_ms=50.0)
        # cwnd is now 15, which equals ssthresh → must exit slow start
        assert s.in_slow_start is False

    def test_remains_in_slow_start_below_ssthresh(self):
        s = make_state()
        s.cwnd = 5.0
        s.ssthresh = 100.0
        s.in_slow_start = True
        s.on_ack(acked_bytes=TCP_MSS, measured_rtt_ms=50.0)
        assert s.in_slow_start is True


# ---------------------------------------------------------------------------
# Congestion avoidance tests
# ---------------------------------------------------------------------------

class TestCongestionAvoidance:

    def _enter_ca(self, s: TCPCongestionState, cwnd: float = 32.0) -> None:
        """Force state into congestion avoidance at a safe cwnd below TCP_MAX_CWND."""
        from slicenet.core.constants import TCP_MAX_CWND
        s.ssthresh = cwnd  # ensure cwnd == ssthresh so next ACK stays in CA
        s.cwnd = cwnd
        s.in_slow_start = False
        assert s.cwnd < TCP_MAX_CWND, "Test setup: cwnd must be below max"

    def test_cwnd_grows_linearly_in_ca(self):
        """In CA, cwnd should grow by roughly 1/cwnd per ACK (much less than 1)."""
        s = make_state()
        self._enter_ca(s, cwnd=32.0)
        cwnd_before = s.cwnd
        s.on_ack(acked_bytes=TCP_MSS, measured_rtt_ms=50.0)
        increment = s.cwnd - cwnd_before
        assert 0 < increment < 1.0, f"CA increment per ACK should be 0 < x < 1, got {increment}"

    def test_cwnd_grows_by_one_per_rtt_in_ca(self):
        """
        After cwnd ACKs (one RTT) in CA, cwnd should increase by approximately 1.
        Use a moderate cwnd (32) well below TCP_MAX_CWND so increments are not capped.
        """
        s = make_state()
        self._enter_ca(s, cwnd=32.0)
        window = int(s.cwnd)
        cwnd_before = s.cwnd
        for _ in range(window):
            s.on_ack(acked_bytes=TCP_MSS, measured_rtt_ms=50.0)
        growth = s.cwnd - cwnd_before
        assert 0.8 <= growth <= 1.2, f"CA growth per RTT expected ~1, got {growth:.3f}"

    def test_cwnd_does_not_exceed_max(self):
        from slicenet.core.constants import TCP_MAX_CWND
        s = make_state()
        s.cwnd = float(TCP_MAX_CWND) - 1.0
        s.ssthresh = 1.0
        s.in_slow_start = True
        s.on_ack(acked_bytes=TCP_MSS, measured_rtt_ms=50.0)
        assert s.cwnd <= float(TCP_MAX_CWND)


# ---------------------------------------------------------------------------
# Fast retransmit tests
# ---------------------------------------------------------------------------

class TestFastRetransmit:

    def test_fast_retransmit_triggered_on_third_dup_ack(self):
        s = make_state()
        s.cwnd = 20.0
        s.ssthresh = 100.0
        s.in_slow_start = False

        assert s.on_duplicate_ack() is False  # 1st dup ACK
        assert s.on_duplicate_ack() is False  # 2nd dup ACK
        triggered = s.on_duplicate_ack()      # 3rd dup ACK
        assert triggered is True

    def test_ssthresh_halved_on_fast_retransmit(self):
        s = make_state()
        s.cwnd = 20.0
        s.ssthresh = 100.0
        s.in_slow_start = False

        for _ in range(3):
            s.on_duplicate_ack()

        assert s.ssthresh == pytest.approx(10.0)

    def test_cwnd_set_to_ssthresh_plus_three_on_fast_retransmit(self):
        s = make_state()
        s.cwnd = 20.0
        s.in_slow_start = False

        for _ in range(3):
            s.on_duplicate_ack()

        # ssthresh = 20/2 = 10, cwnd = ssthresh + 3 = 13
        assert s.cwnd == pytest.approx(13.0)

    def test_in_fast_recovery_after_three_dup_acks(self):
        s = make_state()
        s.cwnd = 20.0
        s.in_slow_start = False
        for _ in range(3):
            s.on_duplicate_ack()
        assert s.in_fast_recovery is True

    def test_cwnd_inflated_per_additional_dup_ack_in_fast_recovery(self):
        s = make_state()
        s.cwnd = 20.0
        s.in_slow_start = False
        for _ in range(3):
            s.on_duplicate_ack()
        cwnd_after_fr = s.cwnd
        s.on_duplicate_ack()  # 4th dup ACK → inflate by 1
        assert s.cwnd == pytest.approx(cwnd_after_fr + 1.0)

    def test_dup_ack_count_cleared_on_new_ack(self):
        s = make_state()
        s.cwnd = 20.0
        s.in_slow_start = False
        s.on_duplicate_ack()
        s.on_duplicate_ack()
        s.on_ack(acked_bytes=TCP_MSS, measured_rtt_ms=50.0)
        assert s.dup_ack_count == 0

    def test_fast_recovery_cleared_on_new_ack(self):
        s = make_state()
        s.cwnd = 20.0
        s.in_slow_start = False
        for _ in range(3):
            s.on_duplicate_ack()
        s.on_ack(acked_bytes=TCP_MSS, measured_rtt_ms=50.0)
        assert s.in_fast_recovery is False


# ---------------------------------------------------------------------------
# RTO / timeout tests
# ---------------------------------------------------------------------------

class TestRTOTimeout:

    def test_cwnd_reset_to_one_on_timeout(self):
        s = make_state()
        s.cwnd = 30.0
        s.ssthresh = 100.0
        s.on_timeout()
        assert s.cwnd == pytest.approx(1.0)

    def test_slow_start_reentered_on_timeout(self):
        s = make_state()
        s.cwnd = 20.0
        s.in_slow_start = False
        s.on_timeout()
        assert s.in_slow_start is True

    def test_ssthresh_halved_on_timeout(self):
        s = make_state()
        s.cwnd = 20.0
        s.ssthresh = 100.0
        s.on_timeout()
        assert s.ssthresh == pytest.approx(10.0)

    def test_ssthresh_minimum_is_two_on_timeout(self):
        s = make_state()
        s.cwnd = 2.0
        s.ssthresh = 100.0
        s.on_timeout()
        assert s.ssthresh >= 2.0

    def test_fast_recovery_cleared_on_timeout(self):
        s = make_state()
        s.in_fast_recovery = True
        s.on_timeout()
        assert s.in_fast_recovery is False

    def test_dup_ack_count_cleared_on_timeout(self):
        s = make_state()
        s.dup_ack_count = 5
        s.on_timeout()
        assert s.dup_ack_count == 0

    def test_cwnd_grows_from_one_after_timeout(self):
        """After a timeout, slow start should grow cwnd from 1."""
        s = make_state()
        s.cwnd = 30.0
        s.ssthresh = 5.0
        s.on_timeout()
        assert s.cwnd == pytest.approx(1.0)
        s.on_ack(acked_bytes=TCP_MSS, measured_rtt_ms=50.0)
        assert s.cwnd == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# RTT estimation tests (Jacobson/Karels)
# ---------------------------------------------------------------------------

class TestJacobsonRTTEstimation:

    def test_srtt_converges_toward_measured_rtt(self):
        """After many samples, SRTT should approach the true RTT."""
        s = make_state()
        true_rtt = 80.0
        for _ in range(50):
            s._update_rtt(true_rtt)
        assert abs(s.srtt_ms - true_rtt) < 5.0, (
            f"SRTT {s.srtt_ms:.2f} should converge near {true_rtt}"
        )

    def test_rttvar_shrinks_with_stable_rtt(self):
        """Variance should decrease when RTT is stable."""
        s = make_state()
        s.srtt_ms = 50.0
        s.rttvar_ms = 30.0
        for _ in range(30):
            s._update_rtt(50.0)
        assert s.rttvar_ms < 30.0

    def test_rto_at_least_min_rto(self):
        """RTO must never fall below TCP_MIN_RTO_MS."""
        s = make_state()
        for _ in range(100):
            s._update_rtt(1.0)  # very small RTT
        assert s.rto_ms >= TCP_MIN_RTO_MS

    def test_rto_increases_with_high_variance(self):
        """High RTT jitter should drive RTO higher."""
        s = make_state()
        s.srtt_ms = 50.0
        s.rttvar_ms = 0.0

        rtts_stable = [50.0] * 20
        for r in rtts_stable:
            s._update_rtt(r)
        rto_stable = s.rto_ms

        # Now inject high-variance RTTs
        import random
        random.seed(7)
        for _ in range(20):
            s._update_rtt(50.0 + random.uniform(-40, 100))
        rto_variable = s.rto_ms

        assert rto_variable >= rto_stable

    def test_rto_alpha_beta_weights(self):
        """SRTT must follow alpha=0.125 update; RTTVAR follows beta=0.25."""
        s = make_state()
        s.srtt_ms = 100.0
        s.rttvar_ms = 0.0
        measured = 120.0
        s._update_rtt(measured)
        expected_srtt = 100.0 + 0.125 * (120.0 - 100.0)  # 102.5
        assert s.srtt_ms == pytest.approx(expected_srtt, abs=0.01)

    def test_rto_capped_at_max(self):
        """RTO must not exceed TCP_MAX_RTO_MS (120 seconds)."""
        from slicenet.core.constants import TCP_MAX_RTO_MS
        s = make_state()
        s.srtt_ms = 60_000.0
        s.rttvar_ms = 60_000.0
        s._update_rtt(100_000.0)
        assert s.rto_ms <= TCP_MAX_RTO_MS


# ---------------------------------------------------------------------------
# TCPEngine integration tests
# ---------------------------------------------------------------------------

class TestTCPEngine:

    def test_register_and_retrieve_flow(self):
        engine = TCPEngine()
        state = engine.register_flow("flow-abc")
        assert engine.get_state("flow-abc") is state

    def test_process_packet_registers_flow_if_missing(self):
        engine = TCPEngine(loss_rate=0.0)
        from slicenet.core.packet import Packet, Protocol, TrafficClass
        pkt = Packet(
            flow_id="new-flow",
            seq_num=1,
            size_bytes=1000,
            protocol=Protocol.TCP,
            traffic_class=TrafficClass.HTTP,
            slice_id="best_effort",
            priority=4,
        )
        accepted, reason = engine.process_packet(pkt)
        assert engine.get_state("new-flow") is not None

    def test_process_ack_reduces_unacked(self):
        engine = TCPEngine(loss_rate=0.0)
        state = engine.register_flow("f1")
        state.unacked = 5
        engine.process_ack("f1", TCP_MSS)
        assert state.unacked == 4

    def test_snapshot_returns_all_flows(self):
        engine = TCPEngine()
        engine.register_flow("a")
        engine.register_flow("b")
        snap = engine.snapshot()
        ids = {s["flow_id"] for s in snap}
        assert "a" in ids and "b" in ids

    def test_cwnd_limited_send(self):
        """With cwnd=1 and one unacked packet, sending should fail."""
        engine = TCPEngine(loss_rate=0.0)
        state = engine.register_flow("limited")
        state.cwnd = 1.0
        state.unacked = 1  # already at limit

        from slicenet.core.packet import Packet, Protocol, TrafficClass
        pkt = Packet(
            flow_id="limited",
            seq_num=2,
            size_bytes=1000,
            protocol=Protocol.TCP,
            traffic_class=TrafficClass.HTTP,
            slice_id="best_effort",
            priority=4,
        )
        accepted, reason = engine.process_packet(pkt)
        assert accepted is False
        assert reason == "cwnd_limit"
