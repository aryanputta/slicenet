"""
Multi-hop network topology model.

Models a directed graph of nodes and links with:
- Per-link propagation delay and bandwidth capacity
- Shortest-path and ECMP path selection
- Propagation delay accumulation across hops
- Bandwidth-delay product calculation per path
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class Node:
    """A network node (router/switch)."""
    node_id: str
    label: str = ""

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return self.node_id == other.node_id

    def __repr__(self) -> str:
        return f"Node({self.node_id})"


@dataclass
class Link:
    """
    A directed network link between two nodes.

    propagation_delay_ms: one-way propagation delay in milliseconds.
    bandwidth_bps: link capacity in bits per second.
    """
    src: str
    dst: str
    propagation_delay_ms: float
    bandwidth_bps: float
    link_id: str = ""

    def __post_init__(self) -> None:
        if not self.link_id:
            self.link_id = f"{self.src}->{self.dst}"

    @property
    def propagation_delay_s(self) -> float:
        return self.propagation_delay_ms / 1000.0

    @property
    def bandwidth_bytes_per_sec(self) -> float:
        return self.bandwidth_bps / 8.0

    def transmission_delay_ms(self, packet_bytes: int) -> float:
        """Serialization delay for a packet of given size."""
        return (packet_bytes * 8 / self.bandwidth_bps) * 1000.0

    def __repr__(self) -> str:
        return (
            f"Link({self.src}->{self.dst}, "
            f"delay={self.propagation_delay_ms:.2f}ms, "
            f"bw={self.bandwidth_bps / 1e6:.1f}Mbps)"
        )


@dataclass
class Path:
    """A sequence of nodes and links forming an end-to-end path."""
    nodes: List[str]
    links: List[Link]

    @property
    def total_propagation_delay_ms(self) -> float:
        """Sum of one-way propagation delays across all links."""
        return sum(link.propagation_delay_ms for link in self.links)

    @property
    def rtt_propagation_ms(self) -> float:
        """Round-trip propagation delay (2x one-way)."""
        return self.total_propagation_delay_ms * 2.0

    @property
    def bottleneck_bandwidth_bps(self) -> float:
        """Min bandwidth along the path (bottleneck link)."""
        if not self.links:
            return 0.0
        return min(link.bandwidth_bps for link in self.links)

    def bdp_bytes(self) -> float:
        """
        Bandwidth-delay product for this path.
        BDP = bottleneck_bandwidth (bytes/s) * RTT (seconds)
        Represents the optimal TCP window size to fill the pipe.
        """
        bw_bytes_s = self.bottleneck_bandwidth_bps / 8.0
        rtt_s = self.rtt_propagation_ms / 1000.0
        return bw_bytes_s * rtt_s

    def hop_count(self) -> int:
        return len(self.links)

    def __repr__(self) -> str:
        hops = " -> ".join(self.nodes)
        return (
            f"Path([{hops}], delay={self.total_propagation_delay_ms:.2f}ms, "
            f"bw={self.bottleneck_bandwidth_bps / 1e6:.1f}Mbps, "
            f"BDP={self.bdp_bytes():.0f}B)"
        )


class NetworkTopology:
    """
    Directed graph network topology.

    Supports:
    - Node and link registration
    - Dijkstra shortest-path (by propagation delay)
    - ECMP: all equal-cost shortest paths
    - Path delay and BDP queries
    """

    def __init__(self, name: str = "topology"):
        self.name = name
        self._nodes: Dict[str, Node] = {}
        self._links: Dict[str, List[Link]] = {}  # src -> list of outgoing links
        self._link_index: Dict[Tuple[str, str], Link] = {}

    def add_node(self, node_id: str, label: str = "") -> Node:
        node = Node(node_id=node_id, label=label or node_id)
        self._nodes[node_id] = node
        if node_id not in self._links:
            self._links[node_id] = []
        return node

    def add_link(
        self,
        src: str,
        dst: str,
        propagation_delay_ms: float,
        bandwidth_bps: float,
        bidirectional: bool = True,
    ) -> Link:
        """Add a link (and optionally its reverse) to the topology."""
        for node_id in (src, dst):
            if node_id not in self._nodes:
                self.add_node(node_id)

        link = Link(
            src=src,
            dst=dst,
            propagation_delay_ms=propagation_delay_ms,
            bandwidth_bps=bandwidth_bps,
        )
        self._links[src].append(link)
        self._link_index[(src, dst)] = link

        if bidirectional:
            rev = Link(
                src=dst,
                dst=src,
                propagation_delay_ms=propagation_delay_ms,
                bandwidth_bps=bandwidth_bps,
            )
            self._links[dst].append(rev)
            self._link_index[(dst, src)] = rev

        return link

    def get_link(self, src: str, dst: str) -> Optional[Link]:
        return self._link_index.get((src, dst))

    def nodes(self) -> List[str]:
        return list(self._nodes.keys())

    def shortest_path(self, src: str, dst: str) -> Optional[Path]:
        """
        Dijkstra shortest path by propagation delay.
        Returns None if no path exists.
        """
        if src not in self._nodes or dst not in self._nodes:
            return None
        if src == dst:
            return Path(nodes=[src], links=[])

        dist: Dict[str, float] = {n: math.inf for n in self._nodes}
        prev_link: Dict[str, Optional[Link]] = {n: None for n in self._nodes}
        dist[src] = 0.0

        # (distance, node_id)
        heap: List[Tuple[float, str]] = [(0.0, src)]

        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            if u == dst:
                break
            for link in self._links.get(u, []):
                v = link.dst
                alt = dist[u] + link.propagation_delay_ms
                if alt < dist[v]:
                    dist[v] = alt
                    prev_link[v] = link
                    heapq.heappush(heap, (alt, v))

        if math.isinf(dist[dst]):
            return None

        links: List[Link] = []
        cur = dst
        while prev_link[cur] is not None:
            lnk = prev_link[cur]
            links.append(lnk)
            cur = lnk.src
        links.reverse()

        node_ids = [src] + [lnk.dst for lnk in links]
        return Path(nodes=node_ids, links=links)

    def ecmp_paths(self, src: str, dst: str) -> List[Path]:
        """
        Return all equal-cost shortest paths from src to dst.
        Uses DFS over the Dijkstra shortest-path DAG.
        """
        sp = self.shortest_path(src, dst)
        if sp is None:
            return []

        target_delay = sp.total_propagation_delay_ms

        # Build shortest-path DAG: node -> allowed outgoing links
        dist: Dict[str, float] = {n: math.inf for n in self._nodes}
        dist[src] = 0.0
        heap: List[Tuple[float, str]] = [(0.0, src)]

        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for link in self._links.get(u, []):
                alt = dist[u] + link.propagation_delay_ms
                if alt < dist[link.dst]:
                    dist[link.dst] = alt
                    heapq.heappush(heap, (alt, link.dst))

        # DFS over the DAG
        all_paths: List[Path] = []

        def dfs(
            cur: str,
            path_nodes: List[str],
            path_links: List[Link],
            visited: Set[str],
        ) -> None:
            if cur == dst:
                total = sum(lk.propagation_delay_ms for lk in path_links)
                if abs(total - target_delay) < 1e-9:
                    all_paths.append(
                        Path(nodes=list(path_nodes), links=list(path_links))
                    )
                return
            for link in self._links.get(cur, []):
                nxt = link.dst
                if nxt in visited:
                    continue
                # Only traverse along the shortest-path DAG
                if abs(dist[cur] + link.propagation_delay_ms - dist[nxt]) < 1e-9:
                    visited.add(nxt)
                    path_nodes.append(nxt)
                    path_links.append(link)
                    dfs(nxt, path_nodes, path_links, visited)
                    path_nodes.pop()
                    path_links.pop()
                    visited.discard(nxt)

        dfs(src, [src], [], {src})
        return all_paths

    def path_delay_ms(self, src: str, dst: str) -> float:
        """One-way propagation delay of the shortest path, or inf if unreachable."""
        path = self.shortest_path(src, dst)
        return path.total_propagation_delay_ms if path else math.inf

    def path_bdp_bytes(self, src: str, dst: str) -> float:
        """BDP of the shortest path in bytes."""
        path = self.shortest_path(src, dst)
        return path.bdp_bytes() if path else 0.0

    def summary(self) -> dict:
        return {
            "name": self.name,
            "nodes": len(self._nodes),
            "links": sum(len(v) for v in self._links.values()),
            "node_ids": sorted(self._nodes.keys()),
        }
