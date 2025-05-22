import numpy as np
from scipy.optimize import linear_sum_assignment   # Hungarian / Kuhn–Munkres


def consensus_ranking_estimation(pis: np.ndarray) -> np.ndarray:
    pis = np.asarray(pis)
    m, n = pis.shape
    C = np.zeros((n, n), dtype=float)

    positions = np.arange(1, n + 1)

    # Compute cost incrementally
    for i in range(n):
        C[i] = np.abs(pis[:, i][:, None] - positions).sum(axis=0)

    row_ind, col_ind = linear_sum_assignment(C)
    sigma_hat = col_ind + 1
    return sigma_hat

