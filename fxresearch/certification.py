"""Deterministic, selection-aware and online uncertainty calculations.

The statistical results require the assumptions stated in the report. Merely
calling a quantile does not verify exchangeability of market days.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from decimal import Decimal, ROUND_CEILING
import numpy as np
from scipy.stats import beta
import csv
import hashlib
import json
from pathlib import Path


def conformal_quantile(scores, alpha):
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite nonempty vector of calibration block scores required")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    rank = int((Decimal(len(values) + 1) * (1 - Decimal(str(alpha)))).to_integral_value(rounding=ROUND_CEILING))
    return float(np.partition(values, rank - 1)[rank - 1]) if rank <= len(values) else math.inf


def minimum_blocks(alpha):
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    return int((1 / Decimal(str(alpha)) - 1).to_integral_value(rounding=ROUND_CEILING))


def block_scores(predicted, realized, scale=1.0):
    """One-sided maximum over all times/actions in each independent block.

    Inputs have shape (blocks, ...). The action family and scale must be fixed
    before calibration; all labels include the same costs and execution rules.
    """
    p, y = np.broadcast_arrays(np.asarray(predicted), np.asarray(realized))
    s = np.broadcast_to(scale, p.shape)
    if p.ndim < 2 or not np.isfinite(p).all() or not np.isfinite(y).all() or np.any(s <= 0):
        raise ValueError("invalid predictions, labels or scales")
    return np.maximum(0.0, ((p - y) / s).reshape(len(p), -1).max(axis=1))


def structured_radius(c, factors, remainder):
    """Exact support of {D phi + U z + eps : ||z||₂<=1, |eps|<=r}.

    c must represent the difference of two paths with identical endpoints;
    callers are responsible for checking Bc=0. Potentials then telescope.
    """
    c = np.asarray(c, dtype=float)
    u = np.asarray(factors, dtype=float)
    r = np.asarray(remainder, dtype=float)
    if u.shape[0] != len(c) or r.shape != c.shape or np.any(r < 0):
        raise ValueError("incompatible uncertainty dimensions")
    return float(np.linalg.norm(u.T @ c) + np.abs(c) @ r)


def box_radius(c, factors, remainder):
    c, factors, remainder = map(np.asarray, (c, factors, remainder))
    return float(np.abs(c) @ (np.linalg.norm(factors, axis=1) + remainder))


@dataclass(frozen=True)
class ReservedLeg:
    """Monotone affine net conversion envelopes, source-unit capacity.

    Firm fill and validity of the envelopes until the deadline are explicit
    contractual/model assumptions, not inferred from a quote timestamp.
    """
    lower_rate: float
    upper_rate: float
    capacity_lower: float
    fee_destination: float = 0.0


def reserved_path_bound(amount, legs, reference_upper=1.0):
    if amount < 0 or reference_upper < 0:
        raise ValueError("nonnegative initial amount and benchmark required")
    lower = upper = float(amount)
    for leg in legs:
        if not 0 < leg.lower_rate <= leg.upper_rate or leg.fee_destination < 0:
            raise ValueError("invalid conversion envelope")
        if upper > leg.capacity_lower + 1e-10:
            return {"capacity_valid": False, "lower_terminal": lower,
                    "upper_terminal": upper, "lower_profit": -math.inf}
        lower = max(0.0, leg.lower_rate * lower - leg.fee_destination)
        upper = max(0.0, leg.upper_rate * upper - leg.fee_destination)
    return {"capacity_valid": True, "lower_terminal": lower, "upper_terminal": upper,
            "lower_profit": lower - amount * reference_upper}


def clopper_pearson(successes, trials, alpha=0.05):
    if not 0 <= successes <= trials or trials < 1:
        return (math.nan, math.nan)
    lo = 0.0 if successes == 0 else beta.ppf(alpha / 2, successes, trials - successes + 1)
    hi = 1.0 if successes == trials else beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
    return float(lo), float(hi)


def conditional_risk_upper(losses, accepted, delta=0.05, family_size=1):
    """One-sided exact binomial upper bound, Bonferroni over frozen policies.

    Episodes must be independent and identically distributed. Acceptance and
    route rules are frozen before these calibration labels are read. Optimising
    over policies after computing all bounds is allowed; inventing another
    threshold after looking at calibration outcomes is not.
    """
    if not 0 <= losses <= accepted or family_size < 1 or not 0 < delta < 1:
        raise ValueError("invalid binomial counts or confidence level")
    if accepted == 0 or losses == accepted:
        return 1.0
    return float(np.nextafter(beta.ppf(1 - delta / family_size, losses + 1, accepted - losses), 1.0))


def validate_policy_blocks(input_path, registry_path, output_path, alpha=.05, delta=.05):
    """One Bernoulli loss per independent block and frozen policy.

    On real full days, loss must mean 'at least one loss during this traded
    day', not a pooled tick/trade count. That is a different estimand from the
    conditional fraction of losing individual trades within a day.
    """
    registry_bytes=Path(registry_path).read_bytes()
    registry=json.loads(registry_bytes)
    policies=registry['policies']
    names={p['policy_id'] for p in policies}
    if len(names)!=len(policies) or not names:
        raise ValueError('nonempty unique frozen policy registry required')
    counts={name:[0,0,0.] for name in names};seen=set();blocks={name:set() for name in names}
    with open(input_path,newline='') as stream:
        for row in csv.DictReader(stream):
            name=row['policy_id'];block=row['block_id'];key=(name,block)
            if name not in names or key in seen:raise ValueError('unknown policy or duplicate policy/block; aggregate by block first')
            accepted=int(row['accepted']);loss=int(row['loss'])
            if accepted not in (0,1) or loss not in (0,1) or loss>accepted:raise ValueError('Bernoulli block summaries required')
            seen.add(key);blocks[name].add(block)
            counts[name][0]+=accepted;counts[name][1]+=loss;counts[name][2]+=float(row.get('pnl') or 0)
    if len({tuple(sorted(v)) for v in blocks.values()})!=1:
        raise ValueError('every frozen policy must be evaluated on the same blocks')
    rows=[]
    for name,(n,k,pnl) in sorted(counts.items()):
        upper=conditional_risk_upper(k,n,delta,len(names))
        rows.append({'policy_id':name,'accepted_blocks':n,'losing_blocks':k,'risk_upper':upper,
                     'validated':upper<=alpha,'calibration_pnl':pnl})
    feasible=[r for r in rows if r['validated']]
    chosen=max(feasible,key=lambda r:r['calibration_pnl'])['policy_id'] if feasible else None
    result={'alpha':alpha,'delta':delta,'family_size':len(names),'policies':rows,'selected_policy':chosen,
            'registry_sha256':hashlib.sha256(registry_bytes).hexdigest(),'target':registry.get('target','unspecified'),
            'estimand':'P(at least one loss in block | policy accepts the block)',
            'assumptions':['policies and target frozen independently of calibration labels','independent identically distributed blocks',
                           'known complete labels for each evaluated policy and block'],
            'not_verified_by_software':['independence and stationarity of market days','availability of counterfactual fills on real data']}
    Path(output_path).write_text(json.dumps(result,indent=2))
    return result


def adaptive_coverage(scores, calibration, alpha=0.05, gamma=0.005, window=199):
    """Basic ACI with one fully matured block label per update.

    q=+inf for beta<=0 and q=-inf for beta>=1. Empty intervals (q<0)
    and infinite intervals are abstentions for trading. This controls a
    long-run empirical error identity, not conditional success probabilities.
    """
    history = list(map(float, calibration))
    level = float(alpha)
    rows = []
    for score in np.asarray(scores, dtype=float):
        before = level
        if level <= 0:
            radius = math.inf
        elif level >= 1:
            radius = -math.inf
        else:
            radius = conformal_quantile(history[-window:], level)
        error = int(score > radius)
        level += gamma * (alpha - error)
        rows.append((radius, error, before, level))
        history.append(float(score))
    return np.asarray(rows)
