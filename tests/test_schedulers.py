"""
Unit tests for all QoS schedulers:
- FIFO
- Priority Queue (strict ordering)
- WFQ (proportional service)
- DRR (deficit credit accumulation)
- Token Bucket (rate limiting)
- RED (probabilistic drop in threshold zone)
"""

import sys
import os
import time
import random

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slicenet.core.packet import Packet, Protocol, TrafficClass, PacketState
from slicenet.scheduler.fifo import FIFOScheduler
from slicenet.scheduler.priority_queue import PriorityScheduler
from slicenet.scheduler.wfq import WFQScheduler
from slicenet.scheduler.drr import DRRScheduler
from slicenet.congestion.token_bucket import TokenBucket
from slicenet.congestion.red import REDQueueManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_packet(
    flow_id: str = "flow-1",
    seq_num: int = 0,
    size_bytes: int = 1000,
    priority: int = 2,
    slice_id: str = "best_effort",
    traffic_class: TrafficClass = TrafficClass.HTTP,
) -> Packet:
    return Packet(
        flow_id=flow_id,
        seq_num=seq_num,
        size_bytes=size_bytes,
        protocol=Protocol.TCP,
        traffic_class=traffic_class,
        slice_id=slice_id,
        priority=priority,
    )


# ---------------------------------------------------------------------------
# FIFO tests
# ---------------------------------------------------------------------------

class TestFIFOScheduler:

    def test_enqueue_returns_true_when_space_available(self):
        sched = FIFOScheduler(max_size=10)
        pkt = make_packet(seq_num=1)
        assert sched.enqueue(pkt) is True

    def test_dequeue_order_is_fifo(self):
        """Packets must come out in the exact order they went in."""
        sched = FIFOScheduler(max_size=100)
        seq_nums = list(range(10))
        pkts = [make_packet(seq_num=i) for i in seq_nums]
        for p in pkts:
            sched.enqueue(p)

        out = []
        while sched.size > 0:
            p = sched.dequeue()
            out.append(p.seq_num)

        assert out == seq_nums

    def test_dequeue_empty_returns_none(self):
        sched = FIFOScheduler()
        assert sched.dequeue() is None

    def test_overflow_drops_packet(self):
        sched = FIFOScheduler(max_size=2)
        for i in range(2):
            sched.enqueue(make_packet(seq_num=i))
        overflow_pkt = make_packet(seq_num=99)
        accepted = sched.enqueue(overflow_pkt)
        assert accepted is False
        assert overflow_pkt.state == PacketState.DROPPED

    def test_drain_returns_up_to_n_packets(self):
        sched = FIFOScheduler(max_size=100)
        for i in range(10):
            sched.enqueue(make_packet(seq_num=i))
        batch = sched.drain(4)
        assert len(batch) == 4

    def test_stats_counters_are_accurate(self):
        sched = FIFOScheduler(max_size=5)
        for i in range(5):
            sched.enqueue(make_packet(seq_num=i))
        # One overflow drop
        sched.enqueue(make_packet(seq_num=99))
        sched.dequeue()
        s = sched.stats()
        assert s["total_enqueued"] == 5
        assert s["total_dropped"] == 1
        assert s["total_dequeued"] == 1


# ---------------------------------------------------------------------------
# Priority Queue tests
# ---------------------------------------------------------------------------

class TestPriorityScheduler:

    def test_strict_priority_ordering(self):
        """Lower priority value (higher urgency) must be dequeued first."""
        sched = PriorityScheduler(max_size_per_level=100)
        low_prio = make_packet(seq_num=1, priority=4)
        high_prio = make_packet(seq_num=2, priority=0)
        med_prio = make_packet(seq_num=3, priority=2)

        # Enqueue in reverse urgency order
        sched.enqueue(low_prio)
        sched.enqueue(med_prio)
        sched.enqueue(high_prio)

        first = sched.dequeue()
        second = sched.dequeue()
        third = sched.dequeue()

        assert first.priority == 0
        assert second.priority == 2
        assert third.priority == 4

    def test_fifo_within_same_priority_level(self):
        """Packets at the same priority must come out FIFO."""
        sched = PriorityScheduler(max_size_per_level=100)
        pkts = [make_packet(seq_num=i, priority=1) for i in range(5)]
        for p in pkts:
            sched.enqueue(p)
        out_seq = [sched.dequeue().seq_num for _ in range(5)]
        assert out_seq == list(range(5))

    def test_high_priority_drains_before_low(self):
        """All high-priority packets before any low-priority packet."""
        sched = PriorityScheduler(max_size_per_level=100)
        for i in range(3):
            sched.enqueue(make_packet(seq_num=100 + i, priority=0))
        for i in range(3):
            sched.enqueue(make_packet(seq_num=200 + i, priority=4))

        results = [sched.dequeue() for _ in range(6)]
        high_seq = [p.seq_num for p in results[:3]]
        low_seq = [p.seq_num for p in results[3:]]
        assert all(s >= 100 and s < 200 for s in high_seq)
        assert all(s >= 200 for s in low_seq)

    def test_overflow_per_level_drops_packet(self):
        sched = PriorityScheduler(max_size_per_level=2)
        sched.enqueue(make_packet(priority=0))
        sched.enqueue(make_packet(priority=0))
        overflow = make_packet(priority=0)
        assert sched.enqueue(overflow) is False
        assert overflow.state == PacketState.DROPPED

    def test_size_reflects_all_levels(self):
        sched = PriorityScheduler(max_size_per_level=100)
        sched.enqueue(make_packet(priority=0))
        sched.enqueue(make_packet(priority=2))
        sched.enqueue(make_packet(priority=4))
        assert sched.size == 3


# ---------------------------------------------------------------------------
# WFQ tests
# ---------------------------------------------------------------------------

class TestWFQScheduler:

    def test_heavier_weight_gets_more_service(self):
        """
        WFQ virtual-finish-time ordering: with weight_heavy=80 and weight_light=20,
        the VFT increment per byte for heavy is 4x smaller than for light, so heavy
        packets are scheduled 4x more often.  We enqueue a large backlog of both
        flows and drain only a subset; the ratio of dequeued counts must be ~4:1.
        Allow ±25% tolerance.
        """
        weights = {"heavy": 80, "light": 20}
        sched = WFQScheduler(weights=weights, max_total_size=20000)

        n = 5000
        # Bulk-enqueue heavy first so its VFTs are all computed relative to virtual_time=0.
        # Then enqueue light. WFQ will serve heavy ~4x more per unit virtual time.
        for i in range(n):
            sched.enqueue(make_packet(
                flow_id="heavy", seq_num=i, size_bytes=100, slice_id="heavy"
            ))
        for i in range(n):
            sched.enqueue(make_packet(
                flow_id="light", seq_num=i, size_bytes=100, slice_id="light"
            ))

        # Drain 1000 packets (well within total) so both backlogs remain non-empty.
        counts: dict[str, int] = {"heavy": 0, "light": 0}
        for _ in range(1000):
            pkt = sched.dequeue()
            if pkt:
                counts[pkt.flow_id] += 1

        ratio = counts["heavy"] / max(counts["light"], 1)
        # Expected ratio ~4; allow ±25%
        assert 3.0 <= ratio <= 5.0, f"WFQ ratio {ratio:.2f} out of expected [3.0, 5.0]"

    def test_enqueue_dequeue_basic(self):
        sched = WFQScheduler(weights={"s1": 50, "s2": 50})
        p = make_packet(slice_id="s1")
        assert sched.enqueue(p) is True
        out = sched.dequeue()
        assert out is not None
        assert out.flow_id == p.flow_id

    def test_overflow_drops(self):
        sched = WFQScheduler(weights={"s1": 100}, max_total_size=1)
        sched.enqueue(make_packet(slice_id="s1"))
        overflow = make_packet(slice_id="s1")
        assert sched.enqueue(overflow) is False
        assert overflow.state == PacketState.DROPPED

    def test_dequeue_empty_returns_none(self):
        sched = WFQScheduler()
        assert sched.dequeue() is None


# ---------------------------------------------------------------------------
# DRR tests
# ---------------------------------------------------------------------------

class TestDRRScheduler:

    def _make_drr(self, weights=None, base_quantum=1500):
        return DRRScheduler(
            weights=weights or {"slice_a": 20, "slice_b": 10},
            base_quantum=base_quantum,
            max_per_queue=1000,
        )

    def test_deficit_credit_accumulates_until_packet_fits(self):
        """
        If a packet is larger than one quantum, it should still be served
        after enough rounds have accumulated sufficient deficit.
        """
        sched = self._make_drr(weights={"slice_a": 10}, base_quantum=500)
        # Enqueue a packet bigger than one quantum (500 bytes)
        big_pkt = make_packet(slice_id="slice_a", size_bytes=1200)
        sched.enqueue(big_pkt)

        served = None
        for _ in range(10):
            served = sched.dequeue()
            if served is not None:
                break

        assert served is not None, "DRR must eventually serve a packet larger than one quantum"
        assert served.size_bytes == 1200

    def test_heavier_slice_served_more_bytes(self):
        """
        DRR proportionality in bytes: slice_a quantum=3000 (weight=20) vs
        slice_b quantum=1500 (weight=10).  With slice_a packets=1000B and
        slice_b packets=500B, per round slice_a serves 3 pkts (3000B) and
        slice_b serves 3 pkts (1500B) → byte ratio 2:1.
        Drain a partial window so both queues remain non-empty.
        """
        sched = self._make_drr(weights={"slice_a": 20, "slice_b": 10}, base_quantum=1500)
        n = 5000
        # Different packet sizes: 1000B for a, 500B for b
        for i in range(n):
            sched.enqueue(make_packet(slice_id="slice_a", seq_num=i, size_bytes=1000))
        for i in range(n):
            sched.enqueue(make_packet(slice_id="slice_b", seq_num=i, size_bytes=500))

        # Drain 1200 packets (both queues still have backlog)
        for _ in range(1200):
            sched.dequeue()

        st = sched.stats()
        bytes_a = st["per_slice_bytes_served"]["slice_a"]
        bytes_b = st["per_slice_bytes_served"]["slice_b"]
        assert bytes_a > bytes_b, (
            f"Heavier slice_a ({bytes_a}B) should serve more bytes than slice_b ({bytes_b}B)"
        )
        # Ratio should be ~2:1 (quantum ratio); allow generous tolerance
        ratio = bytes_a / max(bytes_b, 1)
        assert 1.5 <= ratio <= 2.5, f"DRR byte ratio {ratio:.2f} expected near 2.0"

    def test_enqueue_to_unknown_slice_drops(self):
        sched = self._make_drr()
        pkt = make_packet(slice_id="nonexistent_slice")
        result = sched.enqueue(pkt)
        assert result is False

    def test_per_slice_sizes(self):
        sched = self._make_drr()
        sched.enqueue(make_packet(slice_id="slice_a", seq_num=0))
        sched.enqueue(make_packet(slice_id="slice_a", seq_num=1))
        sched.enqueue(make_packet(slice_id="slice_b", seq_num=0))
        sizes = sched.per_slice_sizes()
        assert sizes["slice_a"] == 2
        assert sizes["slice_b"] == 1


# ---------------------------------------------------------------------------
# Token Bucket tests
# ---------------------------------------------------------------------------

class TestTokenBucket:

    def test_allows_packet_when_tokens_available(self):
        tb = TokenBucket(rate_bps=1_000_000, burst_bytes=10_000)
        assert tb.consume(1000) is True

    def test_drops_when_tokens_exhausted(self):
        tb = TokenBucket(rate_bps=1, burst_bytes=100)
        # Drain the bucket
        tb._tokens = 0.0
        assert tb.consume(1) is False

    def test_tokens_refill_over_time(self):
        """After sleeping, the bucket should have new tokens."""
        rate_bps = 800_000  # 100 KB/s
        tb = TokenBucket(rate_bps=rate_bps, burst_bytes=500)
        # Drain completely
        tb._tokens = 0.0
        tb._last_refill = time.monotonic()
        time.sleep(0.01)  # 10ms → ~1000 bytes at 100KB/s
        assert tb.consume(500) is True

    def test_burst_capped_at_burst_bytes(self):
        tb = TokenBucket(rate_bps=1_000_000, burst_bytes=500)
        # Even after a long wait, tokens should not exceed burst_bytes
        tb._last_refill = time.monotonic() - 3600.0
        tb._refill()
        assert tb._tokens <= 500.0

    def test_rate_limiting_enforces_throughput(self):
        """
        Submit a burst of packets; only packets that fit within burst should pass.
        """
        burst_bytes = 3000
        tb = TokenBucket(rate_bps=8, burst_bytes=burst_bytes)  # tiny rate
        tb._tokens = float(burst_bytes)  # full bucket

        passed = 0
        # Each packet = 1000 bytes; burst allows 3 packets
        for _ in range(10):
            if tb.consume(1000):
                passed += 1

        assert passed == 3, f"Expected exactly 3 packets to pass, got {passed}"

    def test_update_rate_takes_effect(self):
        tb = TokenBucket(rate_bps=1_000_000, burst_bytes=10_000)
        tb.update_rate(2_000_000)
        assert tb.rate_bps == 2_000_000


# ---------------------------------------------------------------------------
# RED tests
# ---------------------------------------------------------------------------

class TestREDQueueManager:

    def test_no_drop_below_min_threshold(self):
        red = REDQueueManager(min_thresh=200, max_thresh=600, max_prob=0.1)
        # Queue well below min_thresh → never drop
        drops = sum(1 for _ in range(100) if red.should_drop(50))
        assert drops == 0

    def test_always_drop_above_max_threshold(self):
        """
        Once the EWMA average exceeds max_thresh, every packet must be dropped.
        We pre-warm the EWMA by feeding a high weight so avg_queue converges quickly,
        then verify that all subsequent arrivals are hard-dropped.
        """
        red = REDQueueManager(min_thresh=200, max_thresh=600, max_prob=0.1, weight=1.0)
        # weight=1.0 → avg_queue == current_queue_len immediately
        drops = sum(1 for _ in range(100) if red.should_drop(700))
        assert drops == 100

    def test_probabilistic_drop_in_threshold_zone(self):
        """
        With avg queue between min and max threshold, some packets should be
        dropped and some should pass (probabilistic, not deterministic).
        We force avg_queue into the zone by setting it directly.
        """
        red = REDQueueManager(min_thresh=200, max_thresh=600, max_prob=0.1, weight=1.0)
        # weight=1.0 → avg_queue = current_queue_len immediately
        n = 1000
        random.seed(42)
        drops = sum(1 for _ in range(n) if red.should_drop(400))
        # Should drop some but not all
        assert 0 < drops < n, f"Expected probabilistic drops, got {drops}/{n}"

    def test_drop_rate_increases_with_queue_depth(self):
        """Packets reported at higher queue depth should yield higher drop rates."""
        red_low = REDQueueManager(min_thresh=200, max_thresh=600, max_prob=0.1, weight=1.0)
        red_high = REDQueueManager(min_thresh=200, max_thresh=600, max_prob=0.1, weight=1.0)

        n = 500
        random.seed(0)
        drops_low = sum(1 for _ in range(n) if red_low.should_drop(250))
        random.seed(0)
        drops_high = sum(1 for _ in range(n) if red_high.should_drop(550))

        assert drops_high >= drops_low, (
            f"Higher queue depth should yield >= drops: low={drops_low}, high={drops_high}"
        )

    def test_stats_tracking(self):
        red = REDQueueManager(min_thresh=200, max_thresh=600)
        for _ in range(10):
            red.should_drop(50)   # below threshold, no drops
        s = red.stats()
        assert s["total_arrivals"] == 10
        assert s["total_drops"] == 0

    def test_invalid_thresholds_raise(self):
        with pytest.raises(ValueError):
            REDQueueManager(min_thresh=600, max_thresh=200)
