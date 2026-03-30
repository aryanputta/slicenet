"""
Tests for FaultTolerantTopology (slicenet/topology/network.py).

Coverage:
  - Link failure injection and failed_links tracking
  - Automatic Dijkstra rerouting around failed links
  - Bidirectional vs unidirectional failure
  - Link recovery and path cache invalidation
  - Failure log with timestamps
  - ECMP paths exclude failed links
  - Per-link utilization counters
  - Unreachable detection when all paths are cut
"""

import pytest
from slicenet.topology.network import FaultTolerantTopology


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _diamond() -> FaultTolerantTopology:
    """
    Diamond topology:
         A
        / \\
       B   C
        \\ /
         D

    A→B: 10ms, A→C: 10ms, B→D: 10ms, C→D: 10ms
    Two equal-cost paths from A to D: A-B-D and A-C-D.
    """
    topo = FaultTolerantTopology(name="diamond")
    for nid in ("A", "B", "C", "D"):
        topo.add_node(nid)
    topo.add_link("A", "B", propagation_delay_ms=10.0, bandwidth_bps=1e9)
    topo.add_link("A", "C", propagation_delay_ms=10.0, bandwidth_bps=1e9)
    topo.add_link("B", "D", propagation_delay_ms=10.0, bandwidth_bps=1e9)
    topo.add_link("C", "D", propagation_delay_ms=10.0, bandwidth_bps=1e9)
    return topo


def _linear() -> FaultTolerantTopology:
    """
    Linear topology:  A — B — C — D
    Delays: 5ms each, 10Gbps.
    Only one path from A to D.
    """
    topo = FaultTolerantTopology(name="linear")
    for nid in ("A", "B", "C", "D"):
        topo.add_node(nid)
    topo.add_link("A", "B", 5.0, 10e9)
    topo.add_link("B", "C", 5.0, 10e9)
    topo.add_link("C", "D", 5.0, 10e9)
    return topo


# ---------------------------------------------------------------------------
# Basic shortest path (non-failure) — sanity checks
# ---------------------------------------------------------------------------

class TestBasicRouting:
    def test_shortest_path_found(self):
        topo = _diamond()
        path = topo.shortest_path("A", "D")
        assert path is not None
        assert path.nodes[0] == "A"
        assert path.nodes[-1] == "D"

    def test_shortest_path_delay(self):
        topo = _diamond()
        path = topo.shortest_path("A", "D")
        assert path is not None
        assert path.total_propagation_delay_ms == pytest.approx(20.0)

    def test_ecmp_diamond_has_two_paths(self):
        topo = _diamond()
        paths = topo.ecmp_paths("A", "D")
        assert len(paths) == 2

    def test_ecmp_both_paths_same_delay(self):
        topo = _diamond()
        for path in topo.ecmp_paths("A", "D"):
            assert path.total_propagation_delay_ms == pytest.approx(20.0)

    def test_same_node_path(self):
        topo = _diamond()
        path = topo.shortest_path("A", "A")
        assert path is not None
        assert path.nodes == ["A"]
        assert path.links == []


# ---------------------------------------------------------------------------
# Link failure injection
# ---------------------------------------------------------------------------

class TestLinkFailure:
    def test_fail_link_marks_as_failed(self):
        topo = _diamond()
        topo.fail_link("A", "B", bidirectional=False)
        assert topo.is_failed("A", "B")
        assert not topo.is_failed("B", "A")  # Unidirectional

    def test_fail_link_bidirectional_marks_both(self):
        topo = _diamond()
        topo.fail_link("A", "B", bidirectional=True)
        assert topo.is_failed("A", "B")
        assert topo.is_failed("B", "A")

    def test_failed_links_list(self):
        topo = _diamond()
        topo.fail_link("A", "B", bidirectional=False)
        assert ("A", "B") in topo.failed_links()

    def test_rerouting_around_failed_link(self):
        """Failing A-B forces traffic via A-C-D."""
        topo = _diamond()
        topo.fail_link("A", "B")
        path = topo.shortest_path("A", "D")
        assert path is not None
        assert "B" not in path.nodes  # Must avoid B

    def test_rerouted_path_uses_alternate_nodes(self):
        topo = _diamond()
        topo.fail_link("A", "B")
        path = topo.shortest_path("A", "D")
        assert path is not None
        assert "C" in path.nodes  # Goes via C

    def test_ecmp_excludes_failed_path(self):
        """After failure of A-B, ECMP from A to D should have only one path."""
        topo = _diamond()
        topo.fail_link("A", "B")
        paths = topo.ecmp_paths("A", "D")
        assert len(paths) == 1
        assert all("B" not in p.nodes for p in paths)

    def test_unreachable_when_all_paths_cut(self):
        """Cut all paths from A to D → returns None."""
        topo = _diamond()
        topo.fail_link("A", "B")
        topo.fail_link("A", "C")
        path = topo.shortest_path("A", "D")
        assert path is None

    def test_linear_unreachable_on_single_link_cut(self):
        """Linear topology: cutting one link disconnects A from D."""
        topo = _linear()
        topo.fail_link("B", "C")
        path = topo.shortest_path("A", "D")
        assert path is None

    def test_path_cache_invalidated_after_failure(self):
        topo = _diamond()
        # Warm cache
        path_before = topo.shortest_path("A", "D")
        assert path_before is not None

        topo.fail_link("A", "B")
        path_after = topo.shortest_path("A", "D")
        assert path_after is not None
        assert "B" not in path_after.nodes  # Cache was invalidated


# ---------------------------------------------------------------------------
# Link recovery
# ---------------------------------------------------------------------------

class TestLinkRecovery:
    def test_recover_link_clears_failure(self):
        topo = _diamond()
        topo.fail_link("A", "B")
        assert topo.is_failed("A", "B")
        topo.recover_link("A", "B")
        assert not topo.is_failed("A", "B")

    def test_path_restored_after_recovery(self):
        topo = _diamond()
        topo.fail_link("A", "B")
        topo.recover_link("A", "B")
        paths = topo.ecmp_paths("A", "D")
        assert len(paths) == 2  # Both paths back

    def test_failure_log_records_recovery_time(self):
        topo = _diamond()
        topo.fail_link("A", "B", bidirectional=False)
        topo.recover_link("A", "B", bidirectional=False)
        log = topo.failure_log()
        matching = [e for e in log if e["src"] == "A" and e["dst"] == "B"]
        assert len(matching) == 1
        assert matching[0]["recovered_at_ms"] is not None

    def test_failure_log_duration_positive(self):
        import time
        topo = _diamond()
        topo.fail_link("A", "B", bidirectional=False)
        time.sleep(0.01)  # 10ms real time
        topo.recover_link("A", "B", bidirectional=False)
        log = topo.failure_log()
        entry = next(e for e in log if e["src"] == "A")
        assert entry["duration_ms"] is not None
        assert entry["duration_ms"] > 0

    def test_path_cache_invalidated_after_recovery(self):
        topo = _diamond()
        topo.fail_link("A", "B")
        _ = topo.shortest_path("A", "D")  # Cache miss-path
        topo.recover_link("A", "B")
        paths = topo.ecmp_paths("A", "D")
        assert len(paths) == 2  # Cache re-populated with full topology


# ---------------------------------------------------------------------------
# Failure log
# ---------------------------------------------------------------------------

class TestFailureLog:
    def test_failure_log_empty_initially(self):
        topo = _diamond()
        assert topo.failure_log() == []

    def test_failure_log_grows_on_failures(self):
        topo = _diamond()
        topo.fail_link("A", "B", bidirectional=False)
        topo.fail_link("A", "C", bidirectional=False)
        assert len(topo.failure_log()) == 2

    def test_failure_log_entries_have_required_keys(self):
        topo = _diamond()
        topo.fail_link("A", "B", bidirectional=False)
        entry = topo.failure_log()[0]
        for key in ("src", "dst", "failed_at_ms", "recovered_at_ms", "duration_ms"):
            assert key in entry

    def test_failure_log_recovered_at_none_while_failed(self):
        topo = _diamond()
        topo.fail_link("A", "B", bidirectional=False)
        entry = topo.failure_log()[0]
        assert entry["recovered_at_ms"] is None

    def test_summary_shows_failed_links_count(self):
        topo = _diamond()
        topo.fail_link("A", "B", bidirectional=False)
        summary = topo.summary()
        assert summary["failed_links"] == 1


# ---------------------------------------------------------------------------
# Utilization tracking
# ---------------------------------------------------------------------------

class TestUtilization:
    def test_record_transmission_increments_counters(self):
        topo = _diamond()
        topo.record_transmission("A", "B", pkt_bytes=1460)
        topo.record_transmission("A", "B", pkt_bytes=1460)
        util = topo.link_utilization()
        assert util["A->B"]["packets"] == 2
        assert util["A->B"]["bytes"] == 2 * 1460

    def test_record_drop_increments_drops(self):
        topo = _diamond()
        topo.record_drop("A", "C")
        util = topo.link_utilization()
        assert util["A->C"]["drops"] == 1

    def test_utilization_pct_between_zero_and_hundred(self):
        import time
        topo = _linear()
        topo.add_link("X", "Y", 1.0, 1_000_000.0)  # 1 Mbps link
        topo.record_transmission("X", "Y", pkt_bytes=1000)
        time.sleep(0.01)
        util = topo.link_utilization()
        pct = util["X->Y"]["utilization_pct"]
        assert 0.0 <= pct <= 100.0

    def test_failed_link_shown_in_utilization(self):
        topo = _diamond()
        topo.record_transmission("A", "B", pkt_bytes=500)
        topo.fail_link("A", "B", bidirectional=False)
        util = topo.link_utilization()
        assert util["A->B"]["failed"] is True
