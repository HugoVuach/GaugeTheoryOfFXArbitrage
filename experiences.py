"""Trois expériences géométriques. Calculs déterministes à graines fixées."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyse
import config as cfg
import gauge
import market_data as market

EPS = cfg.EPS
QUANTILES = (0.90, 0.95, 0.99)


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cylinder():
    lower = ((0, 1), (1, 2), (2, 3), (3, 0))
    upper = tuple((i + 4, j + 4) for i, j in lower)
    vertical = tuple((i, i + 4) for i in range(4))
    edges = lower + upper + vertical
    faces = tuple((i, j, j + 4, i + 4) for i, j in lower)
    d0, d1 = gauge.coboundaries(8, edges, faces)
    oriented = {edge: (k, 1) for k, edge in enumerate(edges)}
    oriented.update({(j, i): (k, -1) for k, (i, j) in enumerate(edges)})
    cycles = []
    for cycle in gauge.simple_cycles(8):
        pairs = tuple(zip(cycle, cycle[1:] + cycle[:1]))
        if all(edge in oriented for edge in pairs):
            row = np.zeros(len(edges))
            for edge in pairs:
                k, sign = oriented[edge]
                row[k] += sign
            cycles.append((cycle, row))
    return edges, d0, d1, cycles


def experiment_one(output):
    edges, d0, d1, cycles = cylinder()
    lengths = np.array([len(c) for c, _ in cycles])
    cycle_rows = np.array([row for _, row in cycles])
    harmonic = np.r_[np.full(8, .25), np.zeros(4)]
    rng = np.random.default_rng(110601)
    potential = rng.normal(0, 100, 8)
    np.testing.assert_allclose(d1 @ harmonic, 0, atol=1e-13)
    np.testing.assert_allclose(d0.T @ harmonic, 0, atol=1e-13)
    assert 12 - np.linalg.matrix_rank(d0) - np.linalg.matrix_rank(d1) == 1

    # Coordonnées en pb logarithmiques. Le changement de repère est exact.
    alphas = np.linspace(-2, 2, 161)
    spreads = np.linspace(0, .60, 121)
    weights = potential @ d0.T + alphas[:, None] * harmonic
    periods = weights @ cycle_rows.T
    mean_gain = periods[:, None, :] / lengths - spreads[None, :, None]
    best = mean_gain.max(axis=-1)
    expected = np.abs(alphas[:, None]) > 4 * spreads[None, :] + 1e-8
    np.testing.assert_array_equal(best > EPS, expected)
    np.testing.assert_allclose(weights @ d1.T, 0, rtol=0, atol=1e-11)
    recovered = (weights @ harmonic) / (harmonic @ harmonic)
    np.testing.assert_allclose(recovered, alphas, rtol=0, atol=1e-11)
    shifted = weights + rng.normal(0, 30, 8) @ d0.T
    gauge_error = float(np.max(np.abs(shifted @ cycle_rows.T - periods)))
    assert gauge_error < 1e-10
    aa, ss = np.meshgrid(alphas, spreads, indexing="ij")
    pd.DataFrame({
        "alpha_pb": aa.ravel(), "demi_spread_pb": ss.ravel(),
        "meilleure_moyenne_pb": best.ravel(),
        "cycle_positif": (best > EPS).ravel(),
    }).to_csv(output / "exp1_phase.csv", index=False)
    rows = []
    for alpha, spread in ((0, .1), (.3, .1), (1, .1), (1, .3)):
        rows.append({"alpha_pb": alpha, "demi_spread_pb": spread,
                     "periode_tour_pb": alpha, "cout_tour_pb": 4 * spread,
                     "gain_tour_favorable_pb": abs(alpha) - 4 * spread,
                     "dilatation_spread": abs(alpha) / (4 * spread)})
    pd.DataFrame(rows).to_csv(output / "exp1_cas.csv", index=False)
    save_json(output / "exp1_validation.json", {
        "sommets": 8, "aretes": len(edges), "faces": len(d1),
        "cycles_orientes_simples": len(cycles), "b1": 1,
        "points_phase": int(best.size), "points_avec_gain": int((best > EPS).sum()),
        "erreur_periodes_sous_jauge_pb": gauge_error,
        "erreur_courbure_pb": float(np.abs(weights @ d1.T).max()),
        "erreur_alpha_pb": float(np.abs(recovered - alphas).max()),
    })


def build_panels(raw, cache):
    reference = pd.read_csv("results/sources_sha256.csv").set_index("archive")
    for name, row in reference.iterrows():
        with (raw / name).open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == row.sha256, name
    buckets, _ = market.extract_archives(raw)
    arrays = {}
    for day, data in buckets.items():
        print(f"Panneaux de cycles : {day}", flush=True)
        grid = market.decision_grid(day)
        weights, _, _, ages = analyse.snapshot(data, grid)
        fresh = analyse.valid_snapshot(weights, ages, cfg.MAX_AGE_MS)
        nominal = gauge.cycle_scores(weights[fresh]) / gauge.LENGTHS
        date_key = day.strftime("%Y%m%d")
        arrays[date_key + "_nominal"] = nominal
        for delay in cfg.DELAYS_MS:
            future = analyse.future_quotes(data, grid, delay, weights, ages)
            moves, available = analyse.envelopes_sample(weights, fresh, future)
            realized = np.zeros_like(nominal)
            complete = np.ones_like(nominal, dtype=bool)
            for col, edges in enumerate(gauge.EDGES):
                for k, (i, j) in enumerate(edges):
                    later, age = future[k]
                    value = later[fresh, i, j]
                    realized[:, col] += value
                    complete[:, col] &= np.isfinite(value) & (age[fresh, gauge.PAIR_INDEX[i, j]] <= cfg.MAX_AGE_MS)
            realized /= gauge.LENGTHS
            key = f"{date_key}_{delay}_"
            arrays[key + "realized"] = realized
            arrays[key + "complete"] = complete
            arrays[key + "available"] = available[fresh]
            arrays[key + "moves"] = moves[fresh]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **arrays)


def cycle_radii(edge_radius):
    linear_edges = tuple(zip(*np.where(analyse.OFF_DIAGONAL)))
    index = {edge: i for i, edge in enumerate(linear_edges)}
    return np.array([sum(edge_radius[index[e]] for e in edges) / len(edges) for edges in gauge.EDGES])


def experiment_two(cache, output):
    panels = np.load(cache)
    rows, limits = [], []
    for delay in cfg.DELAYS_MS:
        errors, moves = [], []
        for day in cfg.CALIBRATION:
            key = f"{day:%Y%m%d}_{delay}_"
            valid = panels[key + "available"]
            nominal = panels[f"{day:%Y%m%d}_nominal"]
            errors.append(np.abs(panels[key + "realized"][valid] - nominal[valid]))
            moves.append(panels[key + "moves"][valid])
        cal_errors, cal_moves = np.concatenate(errors), np.concatenate(moves)
        for quantile in QUANTILES:
            edge_marginal = np.quantile(cal_moves, quantile, axis=0, method="higher")
            edge_joint = np.quantile(cal_moves.max(axis=1), quantile, method="higher")
            cycle_joint = np.quantile(cal_errors.max(axis=1), quantile, method="higher")
            assert cycle_joint <= edge_joint + EPS
            methods = {
                "arcs_marginaux": cycle_radii(edge_marginal),
                "arcs_simultanes": np.full(len(gauge.CYCLES), edge_joint),
                "cycles_simultanes": np.full(len(gauge.CYCLES), cycle_joint),
            }
            for method, radii in methods.items():
                limits.append({"delai_ms": delay, "quantile": quantile, "methode": method,
                               "n_calibration": len(cal_errors), "rayon_min_pb_conversion": radii.min(),
                               "rayon_max_pb_conversion": radii.max()})
                for day in cfg.EVALUATION:
                    key = f"{day:%Y%m%d}_{delay}_"
                    nominal = panels[f"{day:%Y%m%d}_nominal"]
                    realized = panels[key + "realized"]
                    complete = panels[key + "complete"]
                    valid = panels[key + "available"]
                    lower = nominal - radii
                    upper = nominal + radii
                    choice = lower.argmax(axis=1)
                    ix = np.arange(len(choice))
                    accepted = lower[ix, choice] > EPS
                    rejected = upper.max(axis=1) <= EPS
                    selected_complete = complete[ix, choice]
                    selected_gain = realized[ix, choice]
                    positive = selected_gain > EPS
                    covered = (np.abs(realized - nominal) <= radii + EPS).all(axis=1)
                    assert not (accepted & selected_complete & valid & covered & ~positive).any()
                    counterexample = (complete & (realized > EPS)).any(axis=1)
                    assert not (rejected & valid & covered & counterexample).any()
                    edge_bound = edge_marginal if method == "arcs_marginaux" else np.full(12, edge_joint)
                    edge_covered = (panels[key + "moves"] <= edge_bound + EPS).all(axis=1)
                    raw_alert = nominal.max(axis=1) > EPS
                    rows.append({
                        "date": str(day), "delai_ms": delay, "quantile": quantile, "methode": method,
                        "n": len(choice), "alertes_nominales": int(raw_alert.sum()),
                        "acceptes": int(accepted.sum()), "rejetes": int(rejected.sum()),
                        "indetermines": int((~accepted & ~rejected).sum()),
                        "acceptes_evaluables": int((accepted & selected_complete).sum()),
                        "acceptes_positifs": int((accepted & selected_complete & positive).sum()),
                        "acceptes_nonpositifs": int((accepted & selected_complete & ~positive).sum()),
                        "rejets_avec_cycle_positif_rejouable": int((rejected & counterexample).sum()),
                        "couverture_evaluable": int(valid.sum()),
                        "couverture_cycles_ok": int((valid & covered).sum()),
                        "couverture_arcs_ok": int((valid & edge_covered).sum()) if method != "cycles_simultanes" else np.nan,
                        "couverture_alertes_n": int((valid & raw_alert).sum()),
                        "couverture_alertes_ok": int((valid & raw_alert & covered).sum()),
                        "meilleure_borne_max_pb": float(lower.max()),
                    })
    pd.DataFrame(rows).to_csv(output / "exp2_resultats.csv", index=False)
    pd.DataFrame(limits).to_csv(output / "exp2_rayons.csv", index=False)


def sampled_quotes(true_mid, decisions, average_ms, extra_delay_ms, outage, rng):
    n_times = len(true_mid)
    latest = np.empty((n_times, len(cfg.PAIRS)), dtype=np.int64)
    for col in range(len(cfg.PAIRS)):
        updates = np.ones(n_times, dtype=bool) if average_ms == 0 else rng.random(n_times) < -np.expm1(-10 / average_ms)
        updates[0] = True
        if outage and col == 0:
            updates[4000:4500] = False
        latest[:, col] = np.maximum.accumulate(np.where(updates, np.arange(n_times), 0))
    query = np.broadcast_to(decisions[:, None], (len(decisions), len(cfg.PAIRS))).copy()
    eurusd = cfg.PAIRS.index(("EUR", "USD"))
    query[:, eurusd] -= extra_delay_ms // 10
    stamp = latest[query, np.arange(len(cfg.PAIRS))]
    observed = true_mid[stamp, np.arange(len(cfg.PAIRS))]
    return observed, 10 * (decisions[:, None] - stamp)


def pair_graph(edge_values, half_spread):
    matrix = np.full((len(edge_values), gauge.N, gauge.N), -np.inf)
    matrix[:, gauge.PAIR_FROM, gauge.PAIR_TO] = edge_values - half_spread
    matrix[:, gauge.PAIR_TO, gauge.PAIR_FROM] = -edge_values - half_spread
    return matrix


def execute_synthetic(true_mid, decisions, choices, delay_ms, half_spread):
    gain = np.zeros(len(choices))
    # Une jauge dépendante du temps, avec détentions transformées.
    times = np.arange(len(true_mid)) / 100
    chi = 3 * np.sin(times[:, None] / 7 + np.arange(gauge.N)[None, :]) + np.arange(gauge.N)[None, :] * 11
    changed = np.zeros(len(choices))
    for col, edges in enumerate(gauge.EDGES):
        selected = choices == col
        start_currency = edges[0][0]
        previous = decisions[selected]
        end = previous
        for k, (i, j) in enumerate(edges, 1):
            end = decisions[selected] + k * (delay_ms // 10)
            p = gauge.PAIR_INDEX[i, j]
            orientation = 1 if (i, j) == gauge.PAIR_EDGES[p] else -1
            value = orientation * true_mid[end, p] - half_spread
            gain[selected] += value
            changed[selected] += value + chi[end, j] - chi[end, i]
            changed[selected] += chi[end, i] - chi[previous, i]
            previous = end
        changed[selected] -= chi[end, start_currency] - chi[decisions[selected], start_currency]
    np.testing.assert_allclose(changed, gain, atol=1e-10, rtol=0)
    return gain / gauge.LENGTHS[choices], float(np.abs(changed - gain).max())


def experiment_three(output, replicates=40):
    # Unité de temps seconde, volatilités en pb / racine(seconde).
    times = np.arange(12501) / 100
    decisions = np.arange(500, 11500, 10)
    vols = np.array([.45, .55, 0, .60])
    spread = .1
    scenarios = [(0, 0, False), (0, 500, False)]
    scenarios += [(update, lag, False) for update in (20, 100, 500) for lag in (0, 100, 500, 1000)]
    scenarios += [(100, 500, True)]
    rows, gauge_errors = [], []
    for rep in range(replicates):
        rng = np.random.default_rng(110603 + rep)
        common = rng.normal(size=(len(times) - 1, 1))
        idio = rng.normal(size=(len(times) - 1, gauge.N))
        noise = (.6 * common + .8 * idio) * vols * .1
        drift = -.5 * vols**2 / cfg.BP * .01
        log_values = np.vstack((np.zeros(gauge.N), np.cumsum(noise + drift, axis=0)))
        mid = log_values[:, gauge.PAIR_FROM] - log_values[:, gauge.PAIR_TO]
        np.testing.assert_allclose(gauge.best_cycle(pair_graph(mid[decisions], spread))[1], -spread, atol=1e-11, rtol=0)
        for update, lag, outage in scenarios:
            quote_rng = np.random.default_rng(220000 + 1000 * rep + update)
            observed, age = sampled_quotes(mid, decisions, update, lag, outage, quote_rng)
            graph = pair_graph(observed, spread)
            choices, means = gauge.best_cycle(graph)
            flags = means > EPS
            fresh = (age <= 1000).all(axis=1)
            if update == 0 and lag == 0:
                assert not flags.any()
            for delay in (0, 100, 500, 1000):
                gain, error = execute_synthetic(mid, decisions, choices, delay, spread)
                gauge_errors.append(error)
                positive = gain > EPS
                rows.append({
                    "replique": rep, "mise_a_jour_ms": update, "retard_eurusd_ms": lag,
                    "panne": outage, "delai_conversion_ms": delay, "n": len(flags),
                    "n_frais": int(fresh.sum()), "fausses_alertes": int(flags.sum()),
                    "fausses_alertes_fraiches": int((flags & fresh).sum()),
                    "rejeu_n": int((flags & fresh).sum()),
                    "rejeu_positifs": int((flags & fresh & positive).sum()),
                    "gain_moyen_pb_conversion": float(gain[flags & fresh].mean()) if (flags & fresh).any() else np.nan,
                })
            if rep == 0 and update == 100 and lag == 500 and not outage:
                pd.DataFrame({
                    "temps_s": times[decisions], "marge_observee_pb": means,
                    "marge_synchrone_pb": np.full(len(means), -spread),
                    "age_max_ms": age.max(axis=1),
                }).to_csv(output / "exp3_trace.csv", index=False)
        print(f"Marché simulé : {rep + 1}/{replicates}", flush=True)
    pd.DataFrame(rows).to_csv(output / "exp3_repliques.csv", index=False)
    save_json(output / "exp3_validation.json", {
        "repliques": replicates, "scenarios": len(scenarios), "decisions_par_replique": len(decisions),
        "pas_latent_ms": 10, "demi_spread_pb": spread,
        "erreur_max_jauge_temporelle_pb": max(gauge_errors),
        "volatilites_pb_racine_s": vols.tolist(),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("raw"))
    parser.add_argument("--cache", type=Path, default=Path("data/exp2_cycle_panels.npz"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--replicates", type=int, default=40)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    experiment_one(args.output)
    if args.download:
        market.fetch_all(args.raw)
    if args.refresh_data or not args.cache.exists():
        build_panels(args.raw, args.cache)
    experiment_two(args.cache, args.output)
    experiment_three(args.output, args.replicates)


if __name__ == "__main__":
    main()
