"""Finite generalized-flow LP and a dual bound for reserved exchange legs.

This is a deterministic benchmark, not a prediction of future order fills.
The standard LP duality result is not claimed as a new theorem.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class CashEdge:
    source: int
    destination: int
    rate: float
    capacity: float


def funded_flow(n, edges, initial_node, terminal_node, initial_amount, repayment=1.0):
    if initial_node == terminal_node or initial_amount < 0:
        raise ValueError("distinct space-time funding and terminal vertices required")
    matrix = np.zeros((n, len(edges)))
    capacities = np.empty(len(edges))
    for k, edge in enumerate(edges):
        if edge.rate <= 0 or edge.capacity < 0 or not np.isfinite(edge.capacity):
            raise ValueError("finite source-unit capacity and positive net rate required")
        matrix[edge.source, k] -= 1.0
        matrix[edge.destination, k] += edge.rate
        capacities[k] = edge.capacity
    cash = np.zeros(n); cash[initial_node] = initial_amount
    objective = matrix[terminal_node]
    primal = linprog(-objective, A_ub=-matrix, b_ub=cash,
                     bounds=list(zip(np.zeros(len(edges)), capacities)), method="highs")
    # p_v>=0, p_terminal=1; p_source-r*p_dest+s_e >=0.
    dual_rows = np.zeros((len(edges), n + len(edges)))
    for k, edge in enumerate(edges):
        dual_rows[k, edge.source] = -1
        dual_rows[k, edge.destination] = edge.rate
        dual_rows[k, n + k] = -1
    dual_objective = np.r_[cash, capacities]
    bounds = [(0, None)] * (n + len(edges)); bounds[terminal_node] = (1, 1)
    dual = linprog(dual_objective, A_ub=dual_rows, b_ub=np.zeros(len(edges)),
                   bounds=bounds, method="highs")
    if not primal.success or not dual.success:
        raise RuntimeError((primal.message, dual.message))
    terminal = float(-primal.fun + cash[terminal_node])
    bound = float(dual.fun)
    return {"terminal_cash": terminal, "net_profit": terminal - initial_amount * repayment,
            "dual_bound": bound, "duality_gap": bound - terminal,
            "edge_source_amounts": primal.x, "dual_potentials": dual.x[:n],
            "capacity_shadow_prices": dual.x[n:]}
