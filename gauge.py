"""Transports logarithmiques, cycles dirigés et certificats sur un graphe."""
from itertools import combinations, permutations

import numpy as np
from scipy.optimize import linprog

import config as cfg

N = len(cfg.CURRENCIES)
CURRENCY_INDEX = {symbol: i for i, symbol in enumerate(cfg.CURRENCIES)}
PAIR_EDGES = tuple((CURRENCY_INDEX[x], CURRENCY_INDEX[y]) for x, y in cfg.PAIRS)
PAIR_INDEX = {edge: p for p, (i, j) in enumerate(PAIR_EDGES) for edge in ((i, j), (j, i))}
PAIR_FROM, PAIR_TO = zip(*PAIR_EDGES)


def coboundaries(n, edges, faces):
    """Matrices de d_0 et d_1 dans les orientations choisies des cellules."""
    d0 = np.zeros((len(edges), n))
    signed_index = {}
    for k, (i, j) in enumerate(edges):
        d0[k, i], d0[k, j] = -1, 1
        signed_index[i, j] = k, 1
        signed_index[j, i] = k, -1

    d1 = np.zeros((len(faces), len(edges)))
    for row, face in enumerate(faces):
        for edge in zip(face, face[1:] + face[:1]):
            col, orientation = signed_index[edge]
            d1[row, col] += orientation
    return d0, d1


# Complexe complet avec toutes les faces triangulaires du réseau étudié.
D0, D1 = coboundaries(N, PAIR_EDGES, tuple(combinations(range(N), 3)))


def simple_cycles(n):
    """Un représentant de chaque cycle, sans éliminer son sens inverse."""
    for size in range(2, n + 1):
        for nodes in combinations(range(n), size):
            for tail in permutations(nodes[1:]):
                yield (nodes[0],) + tail


CYCLES = tuple(simple_cycles(N))
EDGES = tuple(tuple(zip(cycle, cycle[1:] + cycle[:1])) for cycle in CYCLES)
LENGTHS = np.array([len(cycle) for cycle in CYCLES])
TRIANGLE = LENGTHS == 3


def log_graphs(bid, ask):
    """Poids du bid/ask, du médian géométrique et du demi-spread, en pb log."""
    forward = cfg.BP * np.log(bid)
    inverse = -cfg.BP * np.log(ask)
    shape = bid.shape[:-1] + (N, N)
    net = np.full(shape, -np.inf)
    mid = np.full(shape, -np.inf)
    spread = np.zeros(shape)
    for p, (i, j) in enumerate(PAIR_EDGES):
        net[..., i, j], net[..., j, i] = forward[..., p], inverse[..., p]
        mid[..., i, j] = (forward[..., p] - inverse[..., p]) / 2
        mid[..., j, i] = -mid[..., i, j]
        spread[..., i, j] = spread[..., j, i] = -(forward[..., p] + inverse[..., p]) / 2
    return net, mid, spread


def cycle_scores(weights, cycles=CYCLES):
    result = []
    for cycle in cycles:
        index = np.array(cycle)
        result.append(weights[..., index, np.roll(index, -1)].sum(axis=-1))
    return np.stack(result, axis=-1)


def best_cycle(weights):
    means = cycle_scores(weights) / LENGTHS
    choice = means.argmax(axis=-1)
    return choice, np.max(means, axis=-1)


def hodge_diagnostics(mid):
    """Défaut quadratique moyen au potentiel et flux absolu maximal, en pb log."""
    edge_values = mid[..., PAIR_FROM, PAIR_TO]
    flux = edge_values @ D1.T
    residual = flux @ D1 / N
    rms = np.sqrt(np.mean(residual**2, axis=-1))
    return rms, np.max(np.abs(flux), axis=-1)


def spread_dilation(mid, spread):
    """Facteur minimal de dilatation des demi-spreads pour admettre un potentiel."""
    periods = np.abs(cycle_scores(mid))
    budgets = cycle_scores(spread)
    ratios = np.zeros_like(periods)
    np.divide(periods, budgets, out=ratios, where=budgets > 0)
    ratios[(budgets == 0) & (periods > cfg.EPS)] = np.inf
    return ratios.max(axis=-1)


def bellman_ford_positive(weights):
    """Relaxation en poids maximaux depuis une source virtuelle de poids nul."""
    n = weights.shape[-1]
    weights = weights - cfg.EPS
    distance = np.zeros(weights.shape[:-2] + (n,))
    for _ in range(n):
        previous = distance
        distance = np.maximum(previous, np.max(previous[..., :, None] + weights, axis=-2))
    return (distance > previous + 1e-10).any(axis=-1)


def karp_max_mean(weights):
    """Moyenne maximale sur un graphe dirigé fortement connexe."""
    n = weights.shape[-1]
    history = np.full((n + 1,) + weights.shape[:-1], -np.inf)
    history[0, ..., 0] = 0
    for k in range(1, n + 1):
        history[k] = np.max(history[k - 1, ..., :, None] + weights, axis=-2)
    candidates = np.full(history[:-1].shape, np.inf)
    for k in range(n):
        valid = np.isfinite(history[k]) & np.isfinite(history[n])
        np.subtract(history[n], history[k], where=valid, out=candidates[k])
        candidates[k] /= n - k
    vertex_values = candidates.min(axis=0)
    return np.max(np.where(np.isfinite(history[n]), vertex_values, -np.inf), axis=-1)


def common_value_null(mid, spread):
    """Simultanément, les taux médians reconstruits par l'étoile USD sont exacts."""
    ref = CURRENCY_INDEX["USD"]
    values = mid[..., :, ref].copy()
    values[..., ref] = 0
    model = values[..., :, None] - values[..., None, :] - spread
    indices = np.arange(N)
    model[..., indices, indices] = -np.inf
    return model


def quadrilateral():
    """Exemple en pb log sur le graphe complet à quatre devises."""
    weights = np.full((N, N), -3.0)
    np.fill_diagonal(weights, -np.inf)
    for i in range(N):
        weights[i, (i + 1) % N] = 1
    return weights


def weighted_lp(weights, costs):
    """Dual du rapport maximal de cycle, avec h_0 fixé à zéro."""
    n = len(weights)
    rows = []
    rhs = []
    for i, j in zip(*np.where(np.isfinite(weights))):
        row = np.zeros(n + 1)
        row[i], row[j], row[-1] = 1, -1, -costs[i, j]
        rows.append(row)
        rhs.append(-weights[i, j])
    obj = np.zeros(n + 1)
    obj[-1] = 1
    result = linprog(
        obj, A_ub=rows, b_ub=rhs,
        bounds=[(0, 0)] + [(None, None)] * n, method="highs",
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.fun


def verify_geometry(samples, gauge_shift, rng):
    np.testing.assert_array_equal(D1 @ D0, 0)
    np.testing.assert_array_equal(D0 @ D0.T + D1.T @ D1, N * np.eye(len(PAIR_EDGES)))
    assert np.linalg.matrix_rank(D0) + np.linalg.matrix_rank(D1) == len(PAIR_EDGES)

    # Deux anneaux reliés par quatre faces carrées.
    lower = ((0, 1), (1, 2), (2, 3), (3, 0))
    upper = tuple((i + 4, j + 4) for i, j in lower)
    vertical = tuple((i, i + 4) for i in range(4))
    faces = tuple((i, j, j + 4, i + 4) for i, j in lower)
    cylinder_d0, cylinder_d1 = coboundaries(8, lower + upper + vertical, faces)
    cylinder_form = np.r_[np.full(8, 0.25), np.zeros(4)]
    np.testing.assert_array_equal(cylinder_d1 @ cylinder_form, 0)
    np.testing.assert_array_equal(cylinder_d0.T @ cylinder_form, 0)
    assert np.linalg.matrix_rank(cylinder_d0) + np.linalg.matrix_rank(cylinder_d1) == 11
    assert cylinder_form[:4].sum() == 1

    mid = samples.copy()
    mid[:, PAIR_TO, PAIR_FROM] = -mid[:, PAIR_FROM, PAIR_TO]
    spreads = np.zeros_like(mid)
    spreads[:, PAIR_FROM, PAIR_TO] = rng.uniform(0.1, 3, (len(mid), len(PAIR_EDGES)))
    spreads[:, PAIR_TO, PAIR_FROM] = spreads[:, PAIR_FROM, PAIR_TO]
    rms, flux = hodge_diagnostics(mid)
    shifted_rms, shifted_flux = hodge_diagnostics(mid + gauge_shift)
    np.testing.assert_allclose(shifted_rms, rms, rtol=0, atol=1e-11)
    np.testing.assert_allclose(shifted_flux, flux, rtol=0, atol=1e-11)
    dilation = spread_dilation(mid, spreads)
    np.testing.assert_allclose(spread_dilation(mid + gauge_shift, spreads), dilation,
                               rtol=0, atol=1e-11)
    lp = np.array([weighted_lp(matrix, cost) for matrix, cost in zip(mid, spreads)])
    np.testing.assert_allclose(dilation, lp, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(dilation > 1, best_cycle(mid - spreads)[1] > cfg.EPS)

    flat = np.zeros((N, N))
    np.fill_diagonal(flat, -np.inf)
    np.testing.assert_allclose(spread_dilation(flat, np.zeros_like(flat)), 0)
    assert np.isinf(spread_dilation(mid[0], np.zeros_like(flat)))
    return float(np.max(np.abs(dilation - lp)))


def verify_theory():
    weights = quadrilateral()
    gains = cycle_scores(weights)
    assert gains[TRIANGLE].max() == -1
    assert gains[LENGTHS == 4].max() == 4
    assert gains[LENGTHS == 2].max() == -2
    assert karp_max_mean(weights) == 1
    assert bellman_ford_positive(weights)
    assert not bellman_ford_positive(weights - 1)

    rng = np.random.default_rng(30971)
    samples = rng.normal(0, 3, (48, N, N))
    samples[:, np.arange(N), np.arange(N)] = -np.inf
    gauges = rng.normal(0, 12, (len(samples), N))
    gauge_shift = gauges[:, None, :] - gauges[:, :, None]
    base_scores = cycle_scores(samples)
    error = float(np.max(np.abs(cycle_scores(samples + gauge_shift) - base_scores)))
    np.testing.assert_allclose(error, 0, atol=1e-11)
    means = (base_scores / LENGTHS).max(axis=1)
    np.testing.assert_allclose(karp_max_mean(samples), means, atol=1e-11)
    np.testing.assert_array_equal(bellman_ford_positive(samples), means > cfg.EPS)
    np.testing.assert_allclose(karp_max_mean(samples - 0.35), means - 0.35, atol=1e-11)
    ratios = []
    for sample in samples:
        costs = rng.uniform(0.1, 3, (N, N))
        exact = np.max(cycle_scores(sample) / cycle_scores(costs))
        ratios.append(abs(exact - weighted_lp(sample, costs)))
    lp_error = float(max(ratios))
    np.testing.assert_allclose(lp_error, 0, atol=1e-10)
    return {
        "cycles": len(CYCLES), "matrices_aléatoires": len(samples),
        "erreur_jauge_pb": error, "erreur_dual_programme_linéaire": lp_error,
        "erreur_dual_dilatation_spread": verify_geometry(samples, gauge_shift, rng),
    }


if __name__ == "__main__":
    print(verify_theory())
