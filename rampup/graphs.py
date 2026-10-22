"""Canonical node alignment and sparse graph operations."""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.io import mmread


@dataclass(frozen=True)
class GraphData:
    adjacency: sparse.csr_matrix
    node_ids: np.ndarray

    @property
    def n_nodes(self):
        return len(self.node_ids)

    @property
    def n_edges(self):
        return self.adjacency.nnz // 2

    @cached_property
    def degrees(self):
        return np.asarray(self.adjacency.sum(axis=1)).ravel()

    @cached_property
    def walk(self):
        return sparse.diags(1.0 / self.degrees) @ self.adjacency

    @cached_property
    def return_probability(self):
        # diag(W @ W), without creating W^2 (which can be nearly dense).
        return np.asarray(self.walk.multiply(self.walk.T).sum(axis=1)).ravel()


def _graph(adjacency, node_ids):
    adjacency = sparse.csr_matrix(adjacency, dtype=np.float64)
    adjacency.eliminate_zeros()
    if adjacency.shape != (len(node_ids), len(node_ids)) or len(node_ids) < 2:
        raise ValueError("graph must be square with at least two nodes")
    if adjacency.diagonal().any():
        raise ValueError("self-loops are not supported by these outcome models")
    if not np.isfinite(adjacency.data).all() or not np.all(adjacency.data == 1):
        raise ValueError("expected an unweighted simple graph")
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.sort_indices()
    result = GraphData(adjacency, np.asarray(node_ids, dtype=np.int64))
    if np.any(result.degrees == 0):
        raise ValueError("isolated nodes have undefined degree-normalized exposure")
    return result


def from_networkx(graph):
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("expected a simple undirected graph")
    if any(not isinstance(n, (int, np.integer)) for n in graph.nodes):
        raise ValueError("node IDs must be integers")
    nodes = sorted(graph.nodes)
    adjacency = nx.to_scipy_sparse_array(graph, nodelist=nodes, dtype=float, format="csr")
    return _graph(adjacency, nodes)


def _read_edge_list(path):
    """Read and validate repository-format endpoints without reordering them."""
    path = Path(path)
    with path.open() as stream:
        first = stream.readline().strip()
    if first.startswith("%%MatrixMarket"):
        raise ValueError("legacy_edgelist requires the repository edge-list format, not Matrix Market")
    try:
        n, columns, rows = (int(v) for v in first.split())
        edges = np.loadtxt(path, skiprows=1, dtype=float, ndmin=2)
    except (ValueError, TypeError) as exc:
        raise ValueError("expected graph dimensions followed by endpoint pairs") from exc
    if n != columns or n < 2 or rows < 1 or edges.shape != (rows, 2):
        raise ValueError("graph dimensions or edge row count do not match header")
    if not np.isfinite(edges).all() or np.any(edges != np.floor(edges)) or np.any(edges < 1) or np.any(edges > n):
        raise ValueError("edge endpoints must be integer IDs within 1..n")
    return n, edges.astype(np.int64) - 1


def load_graph(path):
    """Read Matrix Market or repository format: `n n edge_rows`, then 1-based edges."""
    path = Path(path)
    with path.open() as stream:
        first = stream.readline().strip()
    if first.startswith("%%MatrixMarket"):
        matrix = sparse.csr_matrix(mmread(path))
        return _graph(matrix, np.arange(1, matrix.shape[0] + 1))
    n, endpoints = _read_edge_list(path)
    matrix = sparse.coo_matrix((np.ones(len(endpoints)), (endpoints[:, 0], endpoints[:, 1])), shape=(n, n)).tocsr()
    return _graph(matrix, np.arange(1, n + 1))


def demo_graph(seed=0):
    return from_networkx(nx.connected_watts_strogatz_graph(60, 6, 0.2, seed=seed))


def cluster_membership(graph, resolution=10.0, seed=10, *, legacy_edgelist=None):
    """Partition into canonical rows, optionally preserving archived edge insertion.

    The default is unchanged. Legacy mode preserves both the source graph's
    insertion order and the notebooks' stable descending-size cluster labels;
    it does not change treatment counts, RNG streams or outcome generation.
    """
    if legacy_edgelist is None:
        source = nx.from_scipy_sparse_array(graph.adjacency)
    else:
        n, endpoints = _read_edge_list(legacy_edgelist)
        if n != graph.n_nodes or not np.array_equal(graph.node_ids, np.arange(1, n + 1)):
            raise ValueError("legacy edge-list node IDs do not match the canonical graph")
        source = nx.Graph()
        # Do not pre-insert sorted nodes: first occurrence in this edge stream
        # determines the node order shuffled by Louvain's local random state.
        source.add_edges_from((int(u) + 1, int(v) + 1) for u, v in endpoints)
        if set(source) != set(graph.node_ids):
            raise ValueError("legacy edge-list nodes do not match the canonical graph")
        adjacency = nx.to_scipy_sparse_array(source, nodelist=graph.node_ids, dtype=float, format="csr")
        if (adjacency != graph.adjacency).nnz:
            raise ValueError("legacy edge-list edges do not match the canonical graph")
    clusters = nx.community.louvain_communities(source, resolution=resolution, seed=seed)
    if legacy_edgelist is None:
        clusters.sort(key=lambda nodes: (-len(nodes), min(nodes)))
    else:
        clusters.sort(key=len, reverse=True)
    membership = np.empty(graph.n_nodes, dtype=np.int64)
    for index, nodes in enumerate(clusters):
        rows = np.asarray(list(nodes), dtype=np.int64)
        if legacy_edgelist is not None:
            rows = rows - 1  # Source IDs are 1..n; canonical matrix rows are 0..n-1.
        membership[rows] = index
    return membership


def evolve_graphs(graph, count, seed=0):
    """Fixed, treatment-independent preferential edge additions across stages."""
    rng = np.random.default_rng(seed)
    current = nx.from_scipy_sparse_array(graph.adjacency)
    result = []
    for _ in range(count):
        degrees = np.array([current.degree[i] for i in range(graph.n_nodes)], dtype=float)
        selected = np.flatnonzero(degrees >= np.median(degrees))
        probability = degrees / degrees.sum()
        candidates = rng.choice(graph.n_nodes, size=(len(selected), 5), replace=True, p=probability)
        for node, neighbors in zip(selected, candidates):
            current.add_edges_from((int(node), int(other)) for other in neighbors if node != other)
        evolved = nx.to_scipy_sparse_array(current, nodelist=range(graph.n_nodes), format="csr", dtype=float)
        result.append(_graph(evolved, graph.node_ids.copy()))
    return result
