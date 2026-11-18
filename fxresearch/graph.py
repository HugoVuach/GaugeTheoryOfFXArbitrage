"""Polynomial graph diagnostics, independent of the legacy four-currency code.

All weights are log basis points, directed from the currency sold to the currency
received. Missing edges are -inf. The diagonal is absent, never a free trade.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsqr

BP = 10_000.0


def validate_weights(weights):
    a = np.asarray(weights, dtype=float).copy()
    if a.ndim != 2 or a.shape[0] != a.shape[1] or np.isnan(a).any():
        raise ValueError("weights must be a square matrix without NaN")
    if np.isposinf(a).any():
        raise ValueError("positive infinite exchange rate")
    np.fill_diagonal(a, -np.inf)
    return a


def strong_components(a):
    """Return every SCC, including isolated vertices; no completeness assumption."""
    a = validate_weights(a)
    _, labels = connected_components(csr_matrix(np.isfinite(a)), directed=True,
                                     connection="strong")
    return [np.flatnonzero(labels == k) for k in np.unique(labels)]


def positive_cycle(weights, tolerance=1e-8):
    """Bellman--Ford witness, O(nm), O(n+m) auxiliary storage.

    A returned cycle has mean greater than tolerance, to within roundoff.
    An empty tuple means no such witness was found, not an exhaustive list.
    """
    a = validate_weights(weights)
    n = len(a)
    src, dst = np.where(np.isfinite(a))
    if not len(src):
        return ()
    w = a[src, dst] - tolerance
    distance = np.zeros(n)
    parent = np.full(n, -1, dtype=int)
    changed = -1
    for _ in range(n):
        changed = -1
        for i, j, value in zip(src, dst, w):
            candidate = distance[i] + value
            if candidate > distance[j] + 1e-12:
                distance[j] = candidate
                parent[j] = i
                changed = int(j)
        if changed == -1:
            return ()
    vertex = changed
    for _ in range(n):
        vertex = int(parent[vertex])
        if vertex < 0:
            return ()
    reverse = [vertex]
    current = int(parent[vertex])
    while current != vertex and len(reverse) <= n:
        reverse.append(current)
        current = int(parent[current])
    cycle = tuple(reversed(reverse))
    if current != vertex or cycle_mean(a, cycle) <= tolerance - 1e-10:
        return ()
    start = cycle.index(min(cycle))
    return cycle[start:] + cycle[:start]


def cycle_mean(a, cycle):
    if len(cycle) < 2:
        return -np.inf
    return float(np.mean([a[i, j] for i, j in zip(cycle, cycle[1:] + cycle[:1])]))


def maximum_cycle_mean(weights):
    """Karp on each SCC, O(nm), O(n²) DP storage, valid on sparse graphs."""
    a = validate_weights(weights)
    best = -np.inf
    for vertices in strong_components(a):
        if len(vertices) < 2:
            continue
        sub = a[np.ix_(vertices, vertices)]
        n = len(sub)
        history = np.full((n + 1, n), -np.inf)
        # A virtual source connects to every vertex: covers all cyclic classes.
        history[0] = 0.0
        for k in range(1, n + 1):
            history[k] = np.max(history[k - 1, :, None] + sub, axis=0)
        ratios = np.full((n, n), np.inf)
        for k in range(n):
            valid = np.isfinite(history[n]) & np.isfinite(history[k])
            ratios[k, valid] = (history[n, valid] - history[k, valid]) / (n - k)
        vertex_values = ratios.min(axis=0)
        valid = np.isfinite(history[n])
        if valid.any():
            best = max(best, float(vertex_values[valid].max()))
    return best


def best_witness(weights, tolerance=1e-8):
    a = validate_weights(weights)
    value = maximum_cycle_mean(a)
    if not np.isfinite(value):
        return value, ()
    # The critical graph may have zero cycles. A small downward perturbation
    # yields a witness whose mean is within epsilon of the optimum.
    epsilon = max(tolerance, 1e-10 * max(1, np.max(np.abs(a[np.isfinite(a)]))))
    if value > tolerance:
        epsilon = min(epsilon, value / 4)
    cycle = positive_cycle(a - value + 2 * epsilon, tolerance=epsilon / 2)
    return value, cycle


def triangle_maximum(weights):
    a = validate_weights(weights)
    best = -np.inf
    for k in range(len(a)):
        gains = a[:, k, None] + a[k, None, :] + a.T
        np.fill_diagonal(gains, -np.inf)
        gains[k, :] = gains[:, k] = -np.inf
        best = max(best, float(np.max(gains)) / 3)
    return best


def enumerate_cycles(n, max_length=None, anchor=None):
    """Research/reference only: exponential. Production detection never calls it.

    With anchor, each cycle begins at the chosen funding currency. This is a
    restricted candidate family, not a global maximum-cycle algorithm.
    """
    limit = min(n, max_length or n)
    if anchor is not None:
        others = [i for i in range(n) if i != anchor]
        for size in range(2, limit + 1):
            for rest in permutations(others, size - 1):
                yield (anchor,) + rest
    else:
        for size in range(2, limit + 1):
            for vertices in combinations(range(n), size):
                for rest in permutations(vertices[1:]):
                    yield (vertices[0],) + rest


@dataclass
class HodgeSummary:
    rms: float
    potential: np.ndarray
    residual: np.ndarray
    edge_count: int
    cycle_rank: int
    component_count: int


def potential_diagnostic(mid):
    """Least-squares exact-form residual on the available undirected graph.

    It does not equate the residual with a harmonic component. Faces are not
    supplied here; the first Betti number of a filled complex is not inferred.
    """
    a = validate_weights(mid)
    edges = [(i, j) for i in range(len(a)) for j in range(i + 1, len(a))
             if np.isfinite(a[i, j]) and np.isfinite(a[j, i])]
    if not edges:
        return HodgeSummary(0.0, np.zeros(len(a)), np.zeros(0), 0, 0, len(a))
    row_index=np.repeat(np.arange(len(edges)),2)
    column_index=np.array(edges).ravel()
    incidence=coo_matrix((np.tile([-1.,1.],len(edges)),(row_index,column_index)),shape=(len(edges),len(a))).tocsr()
    values = np.empty(len(edges))
    adjacency = np.zeros_like(a, dtype=bool)
    for k, (i, j) in enumerate(edges):
        values[k] = (a[i, j] - a[j, i]) / 2
        adjacency[i, j] = adjacency[j, i] = True
    count, _ = connected_components(csr_matrix(adjacency), directed=False)
    potential = (np.linalg.lstsq(incidence.toarray(), values, rcond=None)[0]
                 if len(a)<=24 else lsqr(incidence,values,atol=1e-12,btol=1e-12,iter_lim=max(100,5*len(a)))[0])
    residual = values - incidence @ potential
    return HodgeSummary(float(np.sqrt(np.mean(residual ** 2))), potential,
                        residual, len(edges), len(edges) - len(a) + count, count)
