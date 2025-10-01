import numpy as np
from scipy.optimize import linear_sum_assignment   # Hungarian / Kuhn–Munkres


def consensus_ranking_estimation(pis: np.ndarray, alpha_fixed: bool = False, alpha_fixed_value: float = 1) -> np.ndarray:
    pis = np.asarray(pis)
    m, n = pis.shape
    C = np.zeros((n, n), dtype=float)

    positions = np.arange(1, n + 1)

    # Compute cost incrementally
    if not alpha_fixed:
        alpha_fixed_value = 1
    for i in range(n):
        _score = (pis[:, i][:, None] - positions)
        C[i] = np.linalg.norm(_score, ord=alpha_fixed_value, axis=0).sum(axis=0)

    row_ind, col_ind = linear_sum_assignment(C)
    sigma_hat = col_ind + 1
    return sigma_hat

