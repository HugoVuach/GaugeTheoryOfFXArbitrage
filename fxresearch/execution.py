"""Sequential IOC execution against a synthetic multi-level currency book.

The market is exogenous. Depth is consumed until the next quote version. This
does not model the endogenous response of other traders. Failure cleanup is an
explicit market-order rule, not an assumption that the remaining cycle fills.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np

BP = 10000.0


@dataclass
class MarketTape:
    log_values_bp: np.ndarray
    step_ms: int = 10
    half_spread_bp: float = 0.10
    pulse_edge: tuple = (0, 1)
    pulse_start_ms: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    pulse_end_ms: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    pulse_amplitudes_bp: np.ndarray = field(default_factory=lambda: np.array([]))
    level_usd: tuple = (20000.0, 180000.0)
    impact_bp: tuple = (0.0, 0.08)

    @property
    def n(self):
        return self.log_values_bp.shape[1]

    def index(self, time_ms):
        k = int(time_ms // self.step_ms)
        if not 0 <= k < len(self.log_values_bp):
            raise IndexError("execution reaches outside the available tape")
        return k

    def amplitude(self, time_ms):
        active = (self.pulse_start_ms <= time_ms) & (time_ms < self.pulse_end_ms)
        return float(np.sum(self.pulse_amplitudes_bp[active]))

    def mid_bp(self, i, j, time_ms):
        values = self.log_values_bp[self.index(time_ms)]
        shift = self.amplitude(time_ms)
        if (i, j) == self.pulse_edge:
            pulse = shift
        elif (j, i) == self.pulse_edge:
            pulse = -shift
        else:
            pulse = 0.0
        return float(values[i] - values[j] + pulse)

    def directed_quote(self, i, j, time_ms):
        """Rates and source-currency capacities for two price levels."""
        k = self.index(time_ms)
        value_usd = math.exp(self.log_values_bp[k, i] / BP)
        mid = self.mid_bp(i, j, time_ms)
        rates = [math.exp((mid - self.half_spread_bp - impact) / BP)
                 for impact in self.impact_bp]
        capacities = [cash / value_usd for cash in self.level_usd]
        return rates, capacities, (i, j, k)


@dataclass(frozen=True)
class ExecutionConfig:
    initial_amount: float = 100000.0
    fee_bp: float = 0.02
    latency_ms: int = 50
    latency_jitter_ms: int = 30
    cleanup_latency_ms: int = 50
    rejection_probability: float = 0.015
    partial_probability: float = 0.04
    validity_threshold_bp: float = 0.35
    validity_reject_probability: float = 0.25
    depth_stress_probability: float = 0.04
    mode: str = "unreserved"


def execute_route(tape, route, decision_ms, observed_times, config, seed,
                  keep_trace=False):
    """One funded attempt; no overlapping attempt shares this hypothetical book.

    In 'reserved' mode, a complete route has firm rates and enough depth fixed
    at decision time. In 'unreserved', each leg sees its own arrival-time book.
    Both modes charge spread, depth slippage and explicit fees.
    Rejection linked to adverse quote change is a stylised validity gate, not
    an empirical model of a particular dealer's last-look contract.
    """
    if len(route) < 2 or config.initial_amount <= 0:
        raise ValueError("nonempty funded cycle required")
    if config.mode not in ("reserved", "unreserved"):
        raise ValueError("unknown execution mode")
    rng = np.random.default_rng(seed)
    latency_rng = np.random.default_rng(seed+17)
    base = route[0]
    cash = np.zeros(tape.n)
    cash[base] = config.initial_amount
    consumed = {}
    now = int(decision_ms)
    trace = []
    complete = True
    rejected = partial = 0
    fee_factor = math.exp(-config.fee_bp / BP)
    reserved = config.mode == "reserved"

    def fill(i, j, amount, time_ms, limit=1.0, cleanup=False):
        rates, capacities, key = tape.directed_quote(i, j, decision_ms if reserved and not cleanup else time_ms)
        prior = consumed.setdefault(key, np.zeros(len(rates)))
        allowed = amount * limit
        destination = used = 0.0
        for k, (rate, capacity) in enumerate(zip(rates, capacities)):
            available = max(0.0, capacity - prior[k])
            if cleanup and k == len(rates) - 1:
                # Declared recovery assumption: direct-to-funding liquidity is
                # replenished sufficiently for cleanup at the second-level rate.
                available = max(available, allowed - used)
            take = min(allowed - used, available)
            if take > 0:
                destination += take * rate * fee_factor
                prior[k] += take
                used += take
            if used >= allowed - 1e-9:
                break
        return used, destination

    for leg, (i, j) in enumerate(zip(route, route[1:] + route[:1])):
        latency = config.latency_ms + int(latency_rng.integers(0, config.latency_jitter_ms + 1))
        now += latency
        amount = float(cash[i])
        old_time = int(observed_times[min(i, j), max(i, j)])
        adverse = (tape.mid_bp(i, j, old_time) - tape.mid_bp(i, j, now))
        reject_p = config.rejection_probability
        if adverse > config.validity_threshold_bp:
            reject_p = max(reject_p, config.validity_reject_probability)
        is_rejected = (not reserved) and rng.random() < reject_p
        fraction = 1.0
        if not reserved and rng.random() < config.partial_probability:
            fraction = float(rng.uniform(0.2, 0.8))
        if not reserved and rng.random() < config.depth_stress_probability:
            fraction = min(fraction, float(rng.uniform(0.1, 0.6)))
        if is_rejected:
            used = received = 0.0
            rejected += 1
        else:
            used, received = fill(i, j, amount, now, fraction)
            cash[i] -= used
            cash[j] += received
        status = "rejected" if is_rejected else "filled" if used >= amount - 1e-7 else "partial"
        trace.append({"leg": leg, "time_ms": now, "source": i, "destination": j,
                      "source_amount": used, "destination_amount_net": received, "status": status})
        if status != "filled":
            complete = False
            partial += status == "partial"
            break
    if not complete:
        for i in range(tape.n):
            if i == base or cash[i] <= 1e-8:
                continue
            now += config.cleanup_latency_ms
            used, received = fill(i, base, float(cash[i]), now, cleanup=True)
            cash[i] -= used
            cash[base] += received
            trace.append({"leg": "cleanup", "time_ms": now, "source": i, "destination": base,
                          "source_amount": used, "destination_amount_net": received, "status": "liquidated"})
    residual = float(np.sum(np.abs(np.delete(cash, base))))
    if residual > 1e-5:
        raise RuntimeError("unclosed synthetic episode; no realized PnL may be reported")
    pnl = float(cash[base] - config.initial_amount)
    return {"pnl": pnl, "return_bp": BP * pnl / config.initial_amount,
            "complete": complete, "rejected": rejected, "partial": int(partial),
            "terminal_ms": now, "residual": residual,
            "trace": trace if keep_trace else []}


def price_only_return(tape, route, decision_ms, config, seed=None):
    """Ideal all-filled, top-of-book sequential return; explicitly ignores depth/failures."""
    log_gain = 0.0
    now=decision_ms
    latency_rng=np.random.default_rng(seed+17) if seed is not None else None
    for k, (i, j) in enumerate(zip(route, route[1:] + route[:1]), 1):
        now+=config.latency_ms+(int(latency_rng.integers(0,config.latency_jitter_ms+1)) if latency_rng is not None else 0)
        time_ms = decision_ms if config.mode == "reserved" else now
        log_gain += tape.mid_bp(i, j, time_ms) - tape.half_spread_bp - config.fee_bp
    return BP * math.expm1(log_gain / BP)
