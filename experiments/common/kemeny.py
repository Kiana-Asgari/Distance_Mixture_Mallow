"""Exact and heuristic Kemeny consensus rankings.

Exact Kemeny minimises sum_pi K(pi, sigma) where K is the Kendall-tau
distance.  We solve the equivalent minimum-feedback-arc-tournament problem
on the precedence graph using an ILP (PuLP + CBC).

Heuristic for n=100: greedy adjacent-swap improvement starting from Borda.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False


def precedence_matrix(rankings: np.ndarray) -> np.ndarray:
    """W[i,j] = number of rankings that put item i before item j.

    Rankings are 0-based positional vectors (rankings[k, p] = item at position p).
    """
    rankings = np.asarray(rankings, dtype=np.int64)
    if rankings.min() == 1:
        rankings = rankings - 1
    m, n = rankings.shape
    pos = np.empty_like(rankings)
    rows = np.arange(m)[:, None]
    pos[rows, rankings] = np.arange(n)[None, :]  # pos[k, item] = its rank
    W = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            W[i, j] = int((pos[:, i] < pos[:, j]).sum())
    return W


def kemeny_objective(order: np.ndarray, W: np.ndarray) -> int:
    """Total Kendall-tau distance of `order` (0-based) to the dataset summarised by W."""
    n = len(order)
    pos = np.empty(n, dtype=np.int64)
    pos[order] = np.arange(n)
    cost = 0
    for i in range(n):
        for j in range(n):
            if i != j and pos[i] < pos[j]:
                # i is ranked before j by the consensus -> incurs W[j, i] disagreements
                cost += int(W[j, i])
    return cost


def borda_order(rankings: np.ndarray) -> np.ndarray:
    """Return Borda-count consensus (0-based)."""
    rankings = np.asarray(rankings, dtype=np.int64)
    if rankings.min() == 1:
        rankings = rankings - 1
    n = rankings.shape[1]
    scores = np.zeros(n)
    for rank in rankings:
        for pos, item in enumerate(rank):
            scores[item] += pos  # lower position score -> better
    return np.argsort(scores)


def kemeny_local_search(
    rankings: np.ndarray,
    init: Optional[np.ndarray] = None,
    max_passes: int = 50,
):
    """Greedy adjacent-swap local search. Used as a heuristic for n>=20."""
    W = precedence_matrix(rankings)
    if init is None:
        order = borda_order(rankings)
    else:
        order = np.array(init, dtype=np.int64)
    if order.min() == 1:
        order = order - 1
    n = len(order)
    obj = kemeny_objective(order, W)
    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            a, b = order[i], order[i + 1]
            # swap delta = W[a,b] - W[b,a]  (consensus change)
            delta = int(W[a, b]) - int(W[b, a])
            if delta < 0:
                order[i], order[i + 1] = b, a
                obj += delta
                improved = True
        if not improved:
            break
    return order, obj, W


def kemeny_exact_ilp(
    rankings: np.ndarray, time_limit_s: int = 600
) -> tuple[np.ndarray, int, np.ndarray]:
    """Solve Kemeny exactly via ILP.  Returns (order, objective, W)."""
    if not HAS_PULP:
        raise RuntimeError("PuLP not installed; cannot run exact Kemeny ILP")
    W = precedence_matrix(rankings)
    n = W.shape[0]
    prob = pulp.LpProblem("kemeny", pulp.LpMinimize)
    # x[i,j] = 1 iff i is ranked before j in the consensus
    x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", cat=pulp.LpBinary)
         for i in range(n) for j in range(n) if i != j}

    # antisymmetry
    for i in range(n):
        for j in range(i + 1, n):
            prob += x[(i, j)] + x[(j, i)] == 1

    # transitivity
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                prob += x[(i, j)] + x[(j, k)] - x[(i, k)] <= 1

    # objective: sum over (i,j), i!=j, x[i,j] * W[j,i]
    prob += pulp.lpSum(int(W[j, i]) * x[(i, j)] for (i, j) in x)

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_s)
    prob.solve(solver)

    # decode order
    n_before = np.zeros(n, dtype=np.int64)
    for (i, j), var in x.items():
        if pulp.value(var) > 0.5:
            n_before[j] += 1
    order = np.argsort(n_before)
    obj = kemeny_objective(order, W)
    return order, obj, W
