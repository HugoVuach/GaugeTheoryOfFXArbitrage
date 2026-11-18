"""Full-session quote diagnostics, separate venue accounts, streaming outputs."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import csv
import json
import math
import sqlite3
import time
import numpy as np
from .data import NS, dataset_summary, iso_utc, iter_quotes, timestamp_ns
from .graph import positive_cycle, best_witness, potential_diagnostic, triangle_maximum, cycle_mean


def run_diagnostics(database, config, output):
    """Grid sampling from receive-time state; no future tick is consulted.

    Nodes are (venue,currency). Cross-venue transfers are deliberately absent:
    a quote from venue B is not free inventory in an account on venue A.
    The default edge policy drops stale pairs individually. 'all' reproduces
    a strict common-cohort experiment. These populations must not be mixed.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = dataset_summary(database)
    orientations = [(v, tuple(sorted((a, b)))) for v, a, b, _ in summary["instruments"]]
    if len(set(orientations)) != len(orientations):
        raise ValueError("both pair orientations occur on one venue; choose one canonical instrument orientation")
    nodes = sorted({(v, c) for v, a, b, _ in summary["instruments"] for c in (a, b)})
    if not nodes:
        raise ValueError("empty quote dataset")
    index = {node: k for k, node in enumerate(nodes)}
    currencies = sorted({c for _, c in nodes})
    if config.get("currencies") and not set(currencies) <= set(config["currencies"]):
        raise ValueError("data contain currencies absent from the declared universe")
    start = timestamp_ns(config["start_utc"]) if config.get("start_utc") else timestamp_ns(summary["start_market_utc"])
    end = timestamp_ns(config["end_utc"]) if config.get("end_utc") else timestamp_ns(summary["end_market_utc"]) + 1
    step = int(config.get("step_ms", 1000) * 1_000_000)
    max_age = int(config.get("max_age_ms", 1000) * 1_000_000)
    if not start < end or step <= 0 or max_age < 0:
        raise ValueError("invalid sampling interval")
    if config.get("freshness_policy", "edge") not in ("edge", "all"):
        raise ValueError("freshness_policy must be edge or all")
    fee = float(config.get("fee_bp_per_conversion", 0))
    tolerance = float(config.get("positive_tolerance_bp", 1e-8))
    iterator = iter_quotes(database, start=max(0, start - max_age - NS), end=end, clock="receive")
    incoming = next(iterator, None)
    book = {}
    daily = defaultdict(lambda: defaultdict(float))
    count, episode, previous_signature = 0, 0, None
    stale_arrivals = 0
    cached_a = cached_result = None
    cached_computations = actual_computations = 0
    began = time.perf_counter()
    with (output / "snapshots.csv").open("w", newline="") as sf, (output / "alerts.csv").open("w", newline="") as af:
        snapshots = csv.DictWriter(sf, fieldnames=["timestamp_utc", "active_pairs", "total_pairs", "vertices", "cycle_rank", "rms_bp", "positive", "max_mean_bp", "triangle_mean_bp", "cohort_admissible"])
        alerts = csv.DictWriter(af, fieldnames=["decision_id", "timestamp_ns", "timestamp_utc", "episode_id", "cycle_json", "mean_bp", "maximum_mean_bp", "gain_bp", "max_age_ms", "quote_ids_json", "receive_assumed", "sizes_known", "firm_known_true", "interpretation"])
        snapshots.writeheader(); alerts.writeheader()
        for current in range(start, end, step):
            while incoming is not None and incoming.receive_ns <= current:
                key = (incoming.venue, incoming.base, incoming.quote)
                if key not in book or incoming.market_ns >= book[key].market_ns:
                    book[key] = incoming
                else:
                    stale_arrivals += 1
                incoming = next(iterator, None)
            date = iso_utc(current)[:10]
            daily[date]["grid_points"] += 1
            active = [q for q in book.values() if 0 <= current - q.market_ns <= max_age]
            admissible = (len(active) == len(summary["instruments"])) if config.get("freshness_policy", "edge") == "all" else bool(active)
            a = np.full((len(nodes), len(nodes)), -np.inf)
            mid = np.full_like(a, -np.inf)
            directed = {}
            if admissible:
                for q in active:
                    i, j = index[q.venue, q.base], index[q.venue, q.quote]
                    a[i, j] = 10000 * math.log(q.bid) - fee
                    a[j, i] = -10000 * math.log(q.ask) - fee
                    mid[i, j] = 5000 * math.log(q.bid * q.ask)
                    mid[j, i] = -mid[i, j]
                    directed[i, j] = directed[j, i] = q
            if cached_a is not None and np.array_equal(cached_a, a):
                result = cached_result
                cached_computations += 1
            else:
                cycle = positive_cycle(a, tolerance)
                if cycle and config.get("optimize_mean_on_alert", True):
                    maximum, optimum = best_witness(a, tolerance)
                    cycle = optimum if optimum and cycle_mean(a,optimum)>tolerance else cycle
                else:
                    maximum = float(np.mean([a[i, j] for i, j in zip(cycle, cycle[1:] + cycle[:1])])) if cycle else math.nan
                diagnostic = potential_diagnostic(mid)
                triangle = triangle_maximum(a) if config.get("compute_triangles", True) else math.nan
                result = cycle, maximum, diagnostic, triangle
                cached_a, cached_result = a.copy(), result
                actual_computations += 1
            cycle, maximum, diagnostic, triangle = result
            snapshots.writerow({"timestamp_utc": iso_utc(current), "active_pairs": len(active), "total_pairs": len(summary["instruments"]), "vertices": len(nodes), "cycle_rank": diagnostic.cycle_rank, "rms_bp": diagnostic.rms, "positive": int(bool(cycle)), "max_mean_bp": maximum, "triangle_mean_bp": triangle, "cohort_admissible": int(admissible)})
            daily[date]["admissible_points"] += int(admissible)
            daily[date]["active_pair_sum"] += len(active)
            daily[date]["positive_points"] += int(bool(cycle))
            if cycle:
                quotes = [directed[i, j] for i, j in zip(cycle, cycle[1:] + cycle[:1])]
                signature = tuple(cycle)
                if previous_signature != signature:
                    episode += 1
                    daily[date]["episodes"] += 1
                alerts.writerow({"decision_id": count, "timestamp_ns": current, "timestamp_utc": iso_utc(current), "episode_id": episode,
                                 "cycle_json": json.dumps([nodes[i] for i in cycle]), "mean_bp": cycle_mean(a,cycle), "maximum_mean_bp":maximum,
                                 "gain_bp": sum(a[i, j] for i, j in zip(cycle, cycle[1:] + cycle[:1])),
                                 "max_age_ms": max(current - q.market_ns for q in quotes) / 1e6,
                                 "quote_ids_json": json.dumps([q.quote_id for q in quotes]),
                                 "receive_assumed": int(any(q.receive_assumed for q in quotes)),
                                 "sizes_known": int(all(q.bid_size is not None and q.ask_size is not None for q in quotes)),
                                 "firm_known_true": int(all(q.firm is True or q.firm == 1 for q in quotes)),
                                 "interpretation": "quote_signal_not_executed_profit"})
                previous_signature = signature
            else:
                previous_signature = None
            count += 1
            if config.get("max_decisions") and count >= int(config["max_decisions"]):
                break
    with (output / "daily.csv").open("w", newline="") as stream:
        fields = ["date_utc", "grid_points", "admissible_points", "positive_points", "episodes", "mean_active_pair_fraction"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for day, stats in sorted(daily.items()):
            writer.writerow({"date_utc": day, **{k: int(stats[k]) for k in fields[1:-1]},
                             "mean_active_pair_fraction": stats["active_pair_sum"] / (stats["grid_points"] * len(summary["instruments"]))})
    manifest = {"config": config, "data": summary, "nodes": nodes, "currency_count": len(currencies),
                "grid_points": count, "episodes": episode, "stale_received_updates_ignored": stale_arrivals,
                "graph_computations": actual_computations, "unchanged_graph_reuses": cached_computations,
                "elapsed_seconds": time.perf_counter() - began,
                "profit_interpretation": "quotes only; no fill or depth guarantee",
                "hodge_interpretation": "graph gradient residual; no filled faces assumed"}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def audit_fills(path, output, base_currency="USD"):
    """Realized cash ledger from supplied fills; no marking of residual positions.

    source_amount and destination_amount are actual cash amounts from order logs;
    fees are additionally charged in fee_currency. Rejected rows must have zero
    cash amounts. A nonclosed episode has realized_pnl=None, never a mid-price PnL.
    """
    ledgers, counts, initial = defaultdict(lambda: defaultdict(float)), defaultdict(int), {}
    with open(path, newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            key = row["episode_id"]
            if key not in initial:
                initial[key] = float(row["initial_amount"])
                ledgers[key][base_currency] = initial[key]
            src, dst = float(row["source_amount"]), float(row["destination_amount"])
            if src < 0 or dst < 0:
                raise ValueError("negative fill cash amounts")
            if row["status"].lower() in ("rejected", "cancelled") and (src != 0 or dst != 0):
                raise ValueError("rejected order has nonzero filled amounts")
            if ledgers[key][row["source_currency"]] + 1e-8 < src:
                raise ValueError("fill exceeds available cash; borrowing/credit logs must be reconciled first")
            ledgers[key][row["source_currency"]] -= src
            ledgers[key][row["destination_currency"]] += dst
            fee = float(row.get("fee_amount") or 0)
            if fee < 0:
                raise ValueError("negative fees are not supported")
            fee_currency=row.get("fee_currency") or base_currency
            if ledgers[key][fee_currency]+1e-8<fee:
                raise ValueError('fee exceeds available cash in its currency')
            ledgers[key][fee_currency] -= fee
            counts[key] += 1
    result = []
    for key, cash in ledgers.items():
        residual = {c: x for c, x in cash.items() if c != base_currency and abs(x) > 1e-8}
        result.append({"episode_id": key, "closed": not bool(residual), "fills": counts[key],
                       "realized_pnl": None if residual else cash[base_currency] - initial[key],
                       "terminal_balances": dict(cash), "residual_positions": residual})
    Path(output).write_text(json.dumps(result, indent=2))
    return result
