import numpy as np
from scipy.optimize import linear_sum_assignment   # Hungarian / Kuhn–Munkres

def consensus_ranking_estimation(pis: np.ndarray) -> np.ndarray:
    """
    Given an (m × n) array `pis`, whose rows are m sampled permutations of {1,…,n},
    return the permutation σ̂ₘ that minimises
        Σ_{i=1}^{n} Σ_{l=1}^{m} |π^{(l)}(i) − σ(i)|.
    The result is returned as a 1-indexed NumPy array of length n.
    """
    pis = np.asarray(pis)
    m, n = pis.shape
    # Build cost matrix C[i, j] = Σ_l |π^{(l)}(i) − (j+1)|
    # Shape tricks:  pis[..., None] is (m, n, 1); positions is (1, 1, n)
    positions = np.arange(1, n + 1)
    C = np.abs(pis[:, :, None] - positions).sum(axis=0)   # (n, n)

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(C)
    sigma_hat = col_ind + 1    # back to 1-based positions
    return sigma_hat