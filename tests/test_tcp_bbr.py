"""
Tests for TCP BBR congestion control (slicenet/transport/tcp_bbr.py).

Coverage:
  - STARTUP phase: exponential BtlBw probing, exit on bandwidth saturation
  - DRAIN phase: reduces inflight to BDP, then transitions to PROBE_BW
  - PROBE_BW phase: 8-phase pacing gain cycle
  - PROBE_RTT: cwnd floor, RTprop refresh
  - BtlBw windowed-max filter
  - RTprop windowed-min filter
  - Duplicate ACK fast-retransmit (no cwnd reduction)
  - RTO timeout (re-enters STARTUP, keeps BtlBw/RTprop)
  - BBREngine: flow registration, admission, ACK processing
"""

import time
import pytest

from slicenet.transport.tcp_bbr import (
    BBREngine,
    BBRMode,
    BBRState,
    BBR_HIGH_GAIN,
    BBR_DRAIN_GAIN,
    BBR_MIN_CWND_PKTS,
    BBR_PROBE_BW_GAINS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(rtt_ms: float = 20.0) -> BBRState:
    state = BBRState(flow_id="test-flow")
    state.rtprop_ms = rtt_ms
    state.srtt_ms = rtt_ms
    return state


def _ack_n(state: BBRState, n: int, rtt_ms: float = 20.0, acked_bytes: int = 1460) -> None:
    for _ in range(n):
        state.on_ack(acked_bytes=acked_bytes, measured_rtt_ms=rtt_ms)


# ---------------------------------------------------------------------------
# STARTUP phase
# ---------------------------------------------------------------------------

class TestStartup:
    def test_initial_mode_is_startup(self):
        s = _make_state()
        assert s.mode == BBRMode.STARTUP

    def test_startup_pacing_gain_is_high_gain(self):
        s = _make_state()
        assert s.pacing_gain == pytest.approx(BBR_HIGH_GAIN, rel=1e-3)

    def test_startup_cwnd_gain_is_high_gain(self):
        s = _make_state()
        assert s.cwnd_gain == pytest.approx(BBR_HIGH_GAIN, rel=1e-3)

    def test_cwnd_grows_during_startup(self):
        s = _make_state()
        initial_cwnd = s.cwnd
        # Feed ACKs with increasing delivery rate to simulate BW probing
        for i in range(10):
            rate = 1_000_000.0 * (2 ** i)  # Doubling rate
            s._btlbw_samples.append(rate)
            s.btlbw_bps = max(s._btlbw_samples)
            s.on_ack(acked_bytes=1460, measured_rtt_ms=20.0)
        # cwnd should have grown (set by BDP)
        assert s.cwnd >= initial_cwnd

    def test_exits_startup_after_bw_plateau(self):
        """BBR exits STARTUP when BtlBw growth < 25% for 3 consecutive rounds."""
        s = _make_state()
        # Seed a stable BtlBw — no growth
        for _ in range(5):
            s._btlbw_samples.append(10_000_000.0)  # constant 10 Mbps
        s.btlbw_bps = 10_000_000.0
        s._prior_btlbw_bps = 10_000_000.0

        # Three ACK rounds with no BtlBw growth → should exit startup
        for _ in range(3):
            s._update_startup()

        assert s.mode in (BBRMode.DRAIN, BBRMode.PROBE_BW)

    def test_does_not_exit_startup_with_growth(self):
        s = _make_state()
        s._prior_btlbw_bps = 5_000_000.0
        s.btlbw_bps = 10_000_000.0  # 100% growth — well above 25%
        s._startup_rounds_no_growth = 0
        s._update_startup()
        assert s.mode == BBRMode.STARTUP
        assert s._startup_rounds_no_growth == 0


# ---------------------------------------------------------------------------
# DRAIN phase
# ---------------------------------------------------------------------------

class TestDrain:
    def test_enters_drain_after_startup_exit(self):
        s = _make_state()
        s._enter_drain()
        assert s.mode == BBRMode.DRAIN

    def test_drain_pacing_gain_less_than_one(self):
        s = _make_state()
        s._enter_drain()
        assert s.pacing_gain == pytest.approx(BBR_DRAIN_GAIN, rel=1e-3)
        assert s.pacing_gain < 1.0

    def test_exits_drain_when_inflight_le_bdp(self):
        s = _make_state()
        s._enter_drain()
        s.btlbw_bps = 10_000_000.0   # 10 Mbps
        s.rtprop_ms = 20.0            # 20 ms
        # BDP = 10Mbps * 0.02s / 8 = 25000 bytes
        bdp = s._bdp_bytes()
        s.flight_size_bytes = int(bdp) - 1  # Just under BDP
        s._update_drain()
        assert s.mode == BBRMode.PROBE_BW

    def test_stays_in_drain_when_inflight_above_bdp(self):
        s = _make_state()
        s._enter_drain()
        s.btlbw_bps = 10_000_000.0
        s.rtprop_ms = 20.0
        bdp = s._bdp_bytes()
        s.flight_size_bytes = int(bdp) + 10_000  # Above BDP
        s._update_drain()
        assert s.mode == BBRMode.DRAIN


# ---------------------------------------------------------------------------
# PROBE_BW phase
# ---------------------------------------------------------------------------

class TestProbeBW:
    def test_enters_probe_bw_after_drain(self):
        s = _make_state()
        s._enter_probe_bw()
        assert s.mode == BBRMode.PROBE_BW

    def test_probe_bw_has_eight_phases(self):
        assert len(BBR_PROBE_BW_GAINS) == 8

    def test_probe_bw_phase_1_gain_is_0_75(self):
        """Phase 1 is 0.75 (drain phase after the 1.25 probe)."""
        assert BBR_PROBE_BW_GAINS[1] == pytest.approx(0.75, rel=1e-3)

    def test_probe_bw_phase_0_gain_is_1_25(self):
        """Phase 0 is the upward probe at 1.25x."""
        assert BBR_PROBE_BW_GAINS[0] == pytest.approx(1.25, rel=1e-3)

    def test_probe_bw_cycles_all_phases(self):
        """After enough RTTs, all 8 phases should be visited."""
        s = _make_state()
        s._enter_probe_bw()
        seen_phases = set()
        for _ in range(200):
            s._update_probe_bw(time.monotonic() * 1000.0)
            seen_phases.add(s._probe_bw_phase)
        assert len(seen_phases) == 8


# ---------------------------------------------------------------------------
# PROBE_RTT phase
# ---------------------------------------------------------------------------

class TestProbeRTT:
    def test_cwnd_floored_at_min_during_probe_rtt(self):
        s = _make_state()
        s.btlbw_bps = 100_000_000.0  # 100 Mbps — would give large cwnd
        s.rtprop_ms = 5.0
        s._enter_probe_rtt(time.monotonic() * 1000.0)
        s._set_cwnd()
        assert s.cwnd == float(BBR_MIN_CWND_PKTS)

    def test_exits_probe_rtt_after_duration(self):
        s = _make_state()
        s.btlbw_bps = 10_000_000.0
        s.rtprop_ms = 20.0

        now_ms = time.monotonic() * 1000.0
        s._enter_probe_rtt(now_ms)
        s.unacked = BBR_MIN_CWND_PKTS - 1  # Trigger done stamp
        s._update_probe_rtt(now_ms)

        # Fast-forward past the 200ms probe duration
        future_ms = now_ms + 250.0
        s._update_probe_rtt(future_ms)
        assert s.mode == BBRMode.PROBE_BW


# ---------------------------------------------------------------------------
# BtlBw and RTprop estimation
# ---------------------------------------------------------------------------

class TestEstimation:
    def test_btlbw_tracks_max_delivery_rate(self):
        s = _make_state()
        rates = [5e6, 10e6, 8e6, 15e6, 12e6]
        for r in rates:
            s._update_btlbw(r)
        assert s.btlbw_bps == pytest.approx(max(rates), rel=1e-6)

    def test_btlbw_windowed_max_evicts_old_samples(self):
        """Windowed max discards old samples after filter window."""
        s = _make_state()
        # Fill window with high values
        for _ in range(s._btlbw_samples.maxlen):  # type: ignore[arg-type]
            s._update_btlbw(50e6)
        # Now add many lower values — old high values should be evicted
        for _ in range(s._btlbw_samples.maxlen):  # type: ignore[arg-type]
            s._update_btlbw(5e6)
        assert s.btlbw_bps == pytest.approx(5e6, rel=1e-6)

    def test_rtprop_tracks_minimum_rtt(self):
        s = _make_state(rtt_ms=100.0)
        now_ms = time.monotonic() * 1000.0
        # Feed a series of RTTs — RTprop should converge to min
        for rtt in [100.0, 80.0, 60.0, 90.0, 40.0, 55.0]:
            s._update_rtprop(rtt, now_ms)
        assert s.rtprop_ms == pytest.approx(40.0, rel=1e-6)

    def test_bdp_equals_bw_times_rtt(self):
        s = _make_state()
        s.btlbw_bps = 10_000_000.0   # 10 Mbps
        s.rtprop_ms = 20.0            # 20 ms
        expected = (10_000_000.0 / 8.0) * 0.020  # bytes/s * s
        assert s._bdp_bytes() == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Fast retransmit (no cwnd reduction in BBR)
# ---------------------------------------------------------------------------

class TestFastRetransmit:
    def test_no_cwnd_reduction_on_dup_ack(self):
        """BBR does not reduce cwnd on duplicate ACKs."""
        s = _make_state()
        s.btlbw_bps = 10_000_000.0
        s.rtprop_ms = 20.0
        s._enter_probe_bw()
        s._set_cwnd()
        cwnd_before = s.cwnd

        s.on_duplicate_ack()
        s.on_duplicate_ack()
        s._set_cwnd()
        assert s.cwnd == pytest.approx(cwnd_before, rel=0.01)

    def test_fast_retransmit_triggered_at_third_dup_ack(self):
        s = _make_state()
        assert s.on_duplicate_ack() is False
        assert s.on_duplicate_ack() is False
        assert s.on_duplicate_ack() is True  # 3rd dup ACK
        assert s.retransmits == 1

    def test_in_fast_recovery_after_dup_ack(self):
        s = _make_state()
        s.on_duplicate_ack()
        s.on_duplicate_ack()
        s.on_duplicate_ack()
        assert s.in_fast_recovery is True

    def test_fast_recovery_clears_on_ack(self):
        s = _make_state()
        s.on_duplicate_ack()
        s.on_duplicate_ack()
        s.on_duplicate_ack()
        s.on_ack(acked_bytes=1460, measured_rtt_ms=20.0)
        assert s.in_fast_recovery is False


# ---------------------------------------------------------------------------
# RTO timeout
# ---------------------------------------------------------------------------

class TestTimeout:
    def test_timeout_resets_inflight(self):
        s = _make_state()
        s.unacked = 50
        s.flight_size_bytes = 50 * 1460
        s.on_timeout()
        assert s.unacked == 0
        assert s.flight_size_bytes == 0

    def test_timeout_reenters_startup(self):
        s = _make_state()
        s._enter_probe_bw()
        s.on_timeout()
        assert s.mode == BBRMode.STARTUP

    def test_btlbw_preserved_after_timeout(self):
        """BBR keeps BtlBw estimate across timeouts."""
        s = _make_state()
        s.btlbw_bps = 50_000_000.0
        s.on_timeout()
        assert s.btlbw_bps == pytest.approx(50_000_000.0, rel=1e-6)

    def test_retransmit_counter_increments(self):
        s = _make_state()
        s.on_timeout()
        assert s.retransmits == 1


# ---------------------------------------------------------------------------
# BBREngine integration
# ---------------------------------------------------------------------------

class TestBBREngine:
    def _make_packet(self, flow_id: str = "f0", size_bytes: int = 1460):
        from unittest.mock import MagicMock
        pkt = MagicMock()
        pkt.flow_id = flow_id
        pkt.size_bytes = size_bytes
        return pkt

    def test_register_flow_creates_state(self):
        engine = BBREngine(base_rtt_ms=20.0, loss_rate=0.0)
        state = engine.register_flow("flow-1")
        assert isinstance(state, BBRState)
        assert state.flow_id == "flow-1"

    def test_process_packet_accepts_when_cwnd_open(self):
        engine = BBREngine(base_rtt_ms=20.0, loss_rate=0.0)
        state = engine.register_flow("flow-a")
        state.cwnd = 100.0  # Wide open
        state.unacked = 0
        pkt = self._make_packet("flow-a")
        accepted, reason = engine.process_packet(pkt)
        assert accepted is True
        assert reason is None

    def test_process_packet_rejects_at_cwnd_limit(self):
        engine = BBREngine(base_rtt_ms=20.0, loss_rate=0.0)
        state = engine.register_flow("flow-b")
        state.cwnd = 5.0
        state.unacked = 5  # At limit
        pkt = self._make_packet("flow-b")
        accepted, reason = engine.process_packet(pkt)
        assert accepted is False
        assert reason == "bbr_cwnd_limit"

    def test_process_ack_updates_state(self):
        engine = BBREngine(base_rtt_ms=20.0, loss_rate=0.0)
        state = engine.register_flow("flow-c")
        state.unacked = 10
        state.flight_size_bytes = 10 * 1460
        engine.process_ack("flow-c", acked_bytes=1460)
        assert state.unacked == 9

    def test_snapshot_returns_all_flows(self):
        engine = BBREngine(base_rtt_ms=20.0, loss_rate=0.0)
        engine.register_flow("f1")
        engine.register_flow("f2")
        snap = engine.snapshot()
        assert len(snap) == 2
        flow_ids = {s["flow_id"] for s in snap}
        assert flow_ids == {"f1", "f2"}

    def test_snapshot_contains_expected_keys(self):
        engine = BBREngine(base_rtt_ms=20.0, loss_rate=0.0)
        engine.register_flow("f-snap")
        snap = engine.snapshot()[0]
        for key in ("flow_id", "mode", "btlbw_mbps", "rtprop_ms",
                    "pacing_rate_mbps", "cwnd_pkts", "bdp_bytes"):
            assert key in snap, f"Missing key: {key}"

    def test_pacing_rate_mbps_property(self):
        s = _make_state()
        s.btlbw_bps = 10_000_000.0
        s.pacing_gain = 1.25
        expected = (10_000_000.0 * 1.25) / 1e6
        assert s.pacing_rate_mbps == pytest.approx(expected, rel=1e-6)
