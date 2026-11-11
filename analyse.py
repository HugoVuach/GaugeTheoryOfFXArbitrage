"""Rejoue les cotations, estime les enveloppes et produit les résultats de la note."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
import gauge
import market_data as market

OFF_DIAGONAL = ~np.eye(gauge.N, dtype=bool)
EURUSD = (gauge.CURRENCY_INDEX["EUR"], gauge.CURRENCY_INDEX["USD"])
AGE_STRICT = 250


def snapshot(data, grid):
    bid, ask, age = market.quotes_at(data, grid)
    net, mid, spread = gauge.log_graphs(bid, ask)
    return net, mid, spread, age


def valid_snapshot(weights, age, cap):
    return np.isfinite(weights[:, OFF_DIAGONAL]).all(axis=1) & (age <= cap).all(axis=1)


def count(mask):
    return int(np.count_nonzero(mask))


def fmt_time(t):
    return datetime.fromtimestamp(int(t) / 1000, timezone.utc).isoformat(timespec="milliseconds")


def name_cycle(index):
    nodes = gauge.CYCLES[index]
    return "–".join(cfg.CURRENCIES[i] for i in nodes + nodes[:1])


def future_quotes(data, grid, delay, weights, age):
    """Taux aux quatre instants d'exécution, lus par dernière cotation connue."""
    if delay == 0:
        return [(weights, age)] * gauge.N
    future = []
    for k in range(1, gauge.N + 1):
        bid, ask, quote_age = market.quotes_at(data, grid + k * delay)
        later, _, _ = gauge.log_graphs(bid, ask)
        future.append((later, quote_age))
    return future


def envelopes_sample(weights, fresh, future):
    valid = fresh.copy()
    movement = np.zeros((len(weights), int(OFF_DIAGONAL.sum())))
    base = weights[:, OFF_DIAGONAL]
    for later, age in future:
        valid &= valid_snapshot(later, age, cfg.COMMON_MAX_AGE_MS)
        movement = np.maximum(movement, np.abs(later[:, OFF_DIAGONAL] - base))
    return movement, valid


def fit_envelopes(buckets):
    radii = {}
    rows = []
    for delay in cfg.DELAYS_MS:
        per_day = []
        for day in cfg.CALIBRATION:
            data = buckets[day]
            grid = market.decision_grid(day)
            weights, _, _, age = snapshot(data, grid)
            fresh = valid_snapshot(weights, age, cfg.MAX_AGE_MS)
            moves, available = envelopes_sample(
                weights, fresh, future_quotes(data, grid, delay, weights, age)
            )
            per_day.append(moves[available])
        pooled = np.concatenate(per_day)
        if not len(pooled):
            raise ValueError("Aucune observation de calibration valide")
        matrix = np.zeros((gauge.N, gauge.N))
        matrix[OFF_DIAGONAL] = np.quantile(pooled, cfg.COVERAGE_Q, axis=0, method="higher")
        radii[delay] = matrix
        for i, j in zip(*np.where(OFF_DIAGONAL)):
            rows.append({
                "delai_ms": delay, "origine": cfg.CURRENCIES[i], "destination": cfg.CURRENCIES[j],
                "rayon_pb": matrix[i, j], "n_calibration": len(pooled),
            })
    return radii, pd.DataFrame(rows)


def replay(future, choices):
    """Rejoue le cycle fixé à t. Le point de départ est fixé par l'ordre de cfg."""
    realized = np.zeros(len(choices))
    complete = np.ones(len(choices), dtype=bool)
    for c, edges in enumerate(gauge.EDGES):
        selected = np.flatnonzero(choices == c)
        if not len(selected):
            continue
        for k, (i, j) in enumerate(edges):
            weights, age = future[k]
            leg = weights[selected, i, j]
            realized[selected] += leg
            complete[selected] &= np.isfinite(leg) & (age[selected, gauge.PAIR_INDEX[i, j]] <= cfg.MAX_AGE_MS)
    return realized / gauge.LENGTHS[choices], complete


def flags_before_replay(weights, fresh):
    # Les instantanés invalides restent dans les sorties de qualité de données,
    # mais ne participent jamais au choix d'un cycle.
    placeholders = gauge.quadrilateral() - 1000
    valid_weights = np.where(fresh[:, None, None], weights, placeholders)
    choice, mean = gauge.best_cycle(valid_weights)
    return valid_weights, choice, (mean > cfg.EPS) & fresh


def compare_replays(day, grid, data, weights, age, fresh, radii):
    rows = []
    valid_weights, choice, nominal = flags_before_replay(weights, fresh)
    strict = (age <= AGE_STRICT).all(axis=1)
    for delay, radius in radii.items():
        future = future_quotes(data, grid, delay, weights, age)
        moves, available = envelopes_sample(weights, fresh, future)
        covered = available & (moves <= radius[OFF_DIAGONAL] + cfg.EPS).all(axis=1)
        lower_choice, lower = gauge.best_cycle(valid_weights - radius)
        _, upper = gauge.best_cycle(valid_weights + radius)
        robust = (lower > cfg.EPS) & fresh
        reject = (upper <= cfg.EPS) & fresh
        assert not (robust & ~nominal).any()
        assert not (robust & reject).any()

        realized, complete = replay(future, choice)
        realized_robust, complete_robust = replay(future, lower_choice)
        yes = realized > cfg.EPS
        yes_robust = realized_robust > cfg.EPS
        assert not (covered & robust & complete_robust & ~yes_robust).any()
        assert not (covered & reject & complete & yes).any()
        if delay == 0:
            assert np.array_equal(nominal, robust)
            assert np.array_equal(nominal, nominal & complete & yes)
        rows.append({
            "date": str(day), "delai_ms": delay, "eligibles": count(fresh),
            "alertes_brutes": count(nominal),
            "brutes_evaluables": count(nominal & complete),
            "brutes_positives": count(nominal & complete & yes),
            "alertes_age_250": count(nominal & strict),
            "age_250_evaluables": count(nominal & strict & complete),
            "age_250_positives": count(nominal & strict & complete & yes),
            "alertes_robustes": count(robust),
            "robustes_evaluables": count(robust & complete_robust),
            "robustes_positives": count(robust & complete_robust & yes_robust),
            "rejets_borne_haute": count(reject), "indetermines": count(fresh & ~robust & ~reject),
            "enveloppe_evaluable": count(available), "enveloppe_jointe_ok": count(covered),
            "enveloppe_alertes_evaluable": count(available & nominal),
            "enveloppe_alertes_ok": count(covered & nominal),
        })
    return rows


def stress_delayed(day, grid, data, weights, mid, spread, fresh):
    rows = []
    null = gauge.common_value_null(mid, spread)
    i, j = EURUSD
    quote_times, bid_series, ask_series = data["EURUSD"]
    cohort = np.ones(count(fresh), dtype=bool)
    shifted = {}
    for delay in cfg.STALE_MS:
        bid, ask, own_age = market.asof(quote_times, bid_series, ask_series, grid[fresh] - delay)
        delayed_age = own_age + delay
        cohort &= np.isfinite(bid) & np.isfinite(ask) & (delayed_age <= cfg.COMMON_MAX_AGE_MS)
        shifted[delay] = bid, ask, delayed_age
    real_reference = weights[fresh][cohort]
    null_reference = null[fresh][cohort]
    base_real = gauge.best_cycle(real_reference)[1] > cfg.EPS
    base_null = gauge.best_cycle(null_reference)[1] > cfg.EPS
    assert not base_null.any()
    for delay, quotes in shifted.items():
        bid, ask, delayed_age = (values[cohort] for values in quotes)
        current = real_reference.copy()
        control = null_reference.copy()
        current[:, i, j] = control[:, i, j] = cfg.BP * np.log(bid)
        current[:, j, i] = control[:, j, i] = -cfg.BP * np.log(ask)
        delayed_real = gauge.best_cycle(current)[1] > cfg.EPS
        delayed_null = gauge.best_cycle(control)[1] > cfg.EPS
        still_fresh = delayed_age <= cfg.MAX_AGE_MS
        induced = delayed_real & ~base_real
        if delay == 0:
            assert np.array_equal(delayed_real, base_real)
            assert np.array_equal(delayed_null, base_null)
        rows.append({
            "date": str(day), "retard_ms": delay, "cohorte": len(bid), "eligibles_initiaux": len(cohort),
            "alertes_reel_reference": count(base_real), "alertes_reel_retarde": count(delayed_real),
            "reel_creees": count(induced), "reel_disparues": count(base_real & ~delayed_real),
            "reel_creees_age_1s": count(induced & still_fresh),
            "fausses_alertes_nul": count(delayed_null), "fausses_alertes_nul_age_1s": count(delayed_null & still_fresh),
        })
    return rows


def geometry_frame(times, mid, spread, net_mean):
    rms, flux = gauge.hodge_diagnostics(mid)
    dilation = gauge.spread_dilation(mid, spread)
    # La comparaison de signe a une bande de tolérance près de l'égalité.
    comparable = np.abs(net_mean) > cfg.EPS
    np.testing.assert_array_equal(dilation[comparable] > 1, net_mean[comparable] > 0)
    return pd.DataFrame({
        "instant_utc": [fmt_time(t) for t in times],
        "defaut_hodge_pb": rms,
        "flux_triangle_max_pb": flux,
        "dilatation_spread": dilation,
        "signal_bidask": net_mean > cfg.EPS,
    })


def daily_scan(day, grid, weights, mid, spread, age, common, fresh):
    times = grid[common]
    net = weights[common]
    valid_ages = age[common].max(axis=1)
    scores = gauge.cycle_scores(net)
    means = scores / gauge.LENGTHS
    best = means.argmax(axis=1)
    best_mean = means[np.arange(len(best)), best]
    mid_mean = gauge.best_cycle(mid[common])[1]
    positive = best_mean > cfg.EPS
    mid_positive = mid_mean > cfg.EPS
    triangle_mean = means[:, gauge.TRIANGLE].max(axis=1)
    quad_mean = means[:, gauge.LENGTHS == 4].max(axis=1)
    triangle = triangle_mean > cfg.EPS
    quadrilateral = quad_mean > cfg.EPS
    eligible = fresh[common]
    quad_only = positive & ~triangle
    np.testing.assert_array_equal(gauge.bellman_ford_positive(net), positive)
    karp_mean = gauge.karp_max_mean(net)
    np.testing.assert_allclose(karp_mean, best_mean, atol=5e-11, rtol=0)
    idx = np.flatnonzero(eligible & positive)
    example = None
    if len(idx):
        strong = idx[np.argmax(best_mean[idx])]
        example = {
            "instant": fmt_time(times[strong]), "cycle": name_cycle(best[strong]),
            "gain_pb": float(scores[strong, best[strong]]), "mean_pb": float(best_mean[strong]),
            "age_ms": int(valid_ages[strong]), "triangle_possible": bool(triangle[strong]),
        }
    ages = []
    for cap in cfg.AGE_CUTS_MS:
        keep = valid_ages <= cap
        ages.append({
            "date": str(day), "age_max_ms": cap, "n": count(keep),
            "alertes": count(keep & positive), "quadrilateres_seuls": count(keep & quad_only),
        })
    fees = []
    for fee in cfg.FEES_BP:
        fees.append({
            "date": str(day), "frais_pb_conversion": fee, "n": count(eligible),
            "alertes": count(eligible & (best_mean > fee + cfg.EPS)),
        })
    eligible_flags = np.zeros(len(grid), dtype=bool)
    eligible_flags[common] = eligible & positive
    new_episode = eligible_flags & ~np.r_[False, eligible_flags[:-1]]
    daily = {
        "date": str(day), "phase": "juin" if day in cfg.CALIBRATION else "juillet",
        "grille": len(grid), "n_30s": count(common), "n_1s": count(eligible),
        "mid_1s": count(eligible & mid_positive), "bidask_30s": count(positive),
        "bidask_1s": count(eligible & positive), "triangle_1s": count(eligible & triangle),
        "quad_positif_1s": count(eligible & quadrilateral),
        "quad_marge_superieure_1s": count(eligible & quadrilateral & (quad_mean > triangle_mean + cfg.EPS)),
        "quad_seul_1s": count(eligible & quad_only), "episodes_1s": count(new_episode),
        "marge_max_pb": float(best_mean[eligible].max()) if eligible.any() else None,
        "erreur_karp_pb": float(np.max(np.abs(karp_mean - best_mean))),
    }
    release = market.timestamp_ms(day, "12:30:00")
    seconds = (times - release) / 1000
    event = (-30 <= seconds) & (seconds <= 120)
    frame = pd.DataFrame({
        "secondes": seconds[event], "marge_mid_pb": mid_mean[event], "marge_bidask_pb": best_mean[event],
        "age_max_ms": valid_ages[event], "eligible": eligible[event],
    })
    geometry = geometry_frame(grid[fresh], mid[fresh], spread[fresh], best_mean[eligible])
    return daily, ages, fees, example, frame, geometry


def run(raw, output):
    output.mkdir(parents=True, exist_ok=True)
    theory = gauge.verify_theory()
    market.manifest(raw, output / "sources_sha256.csv")
    buckets, audit = market.extract_archives(raw)
    audit.to_csv(output / "qualite_donnees.csv", index=False)
    radii, radii_frame = fit_envelopes(buckets)
    radii_frame.to_csv(output / "rayons_calibres.csv", index=False)
    summaries, ages, fees, examples, latency, stale = [], [], [], [], [], []
    geometry = []
    for day, data in buckets.items():
        print(f"Analyse : {day}", flush=True)
        grid = market.decision_grid(day)
        weights, mid, spread, age = snapshot(data, grid)
        common = valid_snapshot(weights, age, cfg.COMMON_MAX_AGE_MS)
        fresh = valid_snapshot(weights, age, cfg.MAX_AGE_MS)
        daily, daily_ages, daily_fees, example, event, daily_geometry = daily_scan(
            day, grid, weights, mid, spread, age, common, fresh
        )
        summaries.append(daily)
        ages.extend(daily_ages)
        fees.extend(daily_fees)
        geometry.append(daily_geometry)
        if example:
            examples.append(example)
        if day in cfg.ANNOUNCEMENT:
            event.to_csv(output / f"profil_ipc_{day}.csv", index=False)
        if day in cfg.EVALUATION:
            latency.extend(compare_replays(day, grid, data, weights, age, fresh, radii))
            stale.extend(stress_delayed(day, grid, data, weights, mid, spread, fresh))

    outputs = {
        "par_jour": pd.DataFrame(summaries), "fraicheur": pd.DataFrame(ages),
        "frais": pd.DataFrame(fees), "latence": pd.DataFrame(latency),
        "retard_eurusd": pd.DataFrame(stale),
        "geometrie": pd.concat(geometry, ignore_index=True),
    }
    for name, frame in outputs.items():
        frame.to_csv(output / f"{name}.csv", index=False)

    summary = {
        "verifications": theory,
        "lignes_dans_fenetres_avec_marges": int(audit["lignes"].sum()),
        "ticks_uniques_avec_marges": int(audit["ticks_uniques"].sum()),
        "prix_invalides": int(audit["prix_invalides"].sum()),
        "volume_non_nul": int(audit["volume_non_nul"].sum()),
        "pics_par_jour": examples,
    }
    with (output / "synthese.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Résultats : {output.resolve()}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("raw"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        market.fetch_all(args.raw)
    run(args.raw, args.output)
