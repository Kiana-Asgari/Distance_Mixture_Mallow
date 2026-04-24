"""Log-likelihood computation for the five competing models.

All models work with 1-based permutations (rankings of items 1..n).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from GMM_diagonalized.sampling import dp_wrapper
from experiments.common.distances import (
    bridge_sample_logZ,
    d_alpha_batch,
    ti_logZ,
)


# ----------------------------------------------------------------------
# Mallows-tau (closed form)
# ----------------------------------------------------------------------
def kendall_tau_distance(pi: np.ndarray, sigma: np.ndarray) -> int:
    """Number of inversions between two equal-length rankings."""
    pi = np.asarray(pi)
    sigma = np.asarray(sigma)
    pos = {item: i for i, item in enumerate(sigma)}
    pi_in_sigma = [pos[item] for item in pi]
    inv = 0
    for i in range(len(pi_in_sigma)):
        for j in range(i + 1, len(pi_in_sigma)):
            if pi_in_sigma[i] > pi_in_sigma[j]:
                inv += 1
    return inv


def log_Z_kendall(theta: float, n: int) -> float:
    if theta <= 0:
        return math.lgamma(n + 1)  # uniform limit
    log_z = 0.0
    for j in range(1, n + 1):
        log_z += math.log1p(-math.exp(-j * theta)) - math.log1p(-math.exp(-theta))
    return log_z


def loglik_kendall(rankings: np.ndarray, sigma: np.ndarray, theta: float) -> np.ndarray:
    """log P(pi) for Mallows-tau, vectorised over rankings."""
    n = len(sigma)
    log_z = log_Z_kendall(theta, n)
    distances = np.array([kendall_tau_distance(pi, sigma) for pi in rankings])
    return -theta * distances - log_z


# ----------------------------------------------------------------------
# Plackett-Luce (closed form)
# ----------------------------------------------------------------------
def loglik_PL(rankings: np.ndarray, utilities: np.ndarray) -> np.ndarray:
    """Compute log P(pi) under Plackett-Luce. Rankings are 0-based for indexing."""
    rankings = np.asarray(rankings, dtype=np.int64)
    if rankings.min() == 1:  # convert to 0-based
        rankings = rankings - 1
    exp_u = np.exp(utilities - utilities.max())  # numerical stability
    chosen_exp = exp_u[rankings]
    # cumulative sum from the right gives denominator at each rank position
    denom = np.flip(np.cumsum(np.flip(chosen_exp, axis=1), axis=1), axis=1)
    log_terms = np.log(np.maximum(chosen_exp[:, :-1], 1e-300)) - np.log(
        np.maximum(denom[:, :-1], 1e-300)
    )
    return log_terms.sum(axis=1)


# ----------------------------------------------------------------------
# Distance Mallows (alpha >= 1) using banded DP for log Z
# ----------------------------------------------------------------------
def log_Z_distance_dp(n: int, alpha: float, beta: float, Delta: int) -> float:
    Z = dp_wrapper(n, alpha, beta, Delta)
    return math.log(max(Z, 1e-300))


def truncation_error_bound(
    n: int, alpha: float, beta: float, Delta: int
) -> tuple[float, float]:
    """Empirical truncation-error proxy: relative gap between Z(Delta) and Z(Delta+1).

    Returns (relative_gap, log_Z_used).
    """
    log_z = log_Z_distance_dp(n, alpha, beta, Delta)
    log_z_plus = log_Z_distance_dp(n, alpha, beta, Delta + 1)
    rel_gap = abs(math.exp(log_z_plus - log_z) - 1.0)
    return rel_gap, log_z


def choose_truncation(
    n: int,
    alpha: float,
    beta: float,
    target_tv: float = 1e-4,
    delta_min: int = 3,
    delta_max: int = 10,
) -> tuple[int, float]:
    """Pick the smallest Delta such that the empirical relative-gap proxy <= target_tv.

    Returns (Delta_chosen, achieved_gap).
    """
    delta_max = min(delta_max, n - 1)
    last_gap = float("inf")
    for D in range(delta_min, delta_max + 1):
        gap, _ = truncation_error_bound(n, alpha, beta, D)
        last_gap = gap
        if gap <= target_tv:
            return D, gap
    return delta_max, last_gap


def loglik_distance(
    rankings: np.ndarray,
    sigma: np.ndarray,
    alpha: float,
    beta: float,
    Delta: int,
    log_z: float | None = None,
) -> np.ndarray:
    """log P(pi) under L_alpha Mallows centered at sigma. Uses banded DP for log Z."""
    n = len(sigma)
    if log_z is None:
        log_z = log_Z_distance_dp(n, alpha, beta, Delta)
    distances = d_alpha_batch(rankings, sigma, alpha)
    return -beta * distances - log_z


def loglik_distance_mcmc(
    rankings: np.ndarray,
    sigma: np.ndarray,
    alpha: float,
    beta: float,
    n_samples_logZ: int = 5000,
    rng_seed: int = 0,
    method: str = "ti",
    n_chains: int = 5,
) -> tuple[np.ndarray, float, float]:
    """log P(pi) when alpha < 1 using TI (default) or bridge sampling for log Z.

    Returns (loglik_per_ranking, log_Z_estimate, log_Z_se_proxy).
    """
    n = len(sigma)
    estimates = np.empty(n_chains)
    for k in range(n_chains):
        if method == "bridge":
            estimates[k] = bridge_sample_logZ(
                n=n, alpha=alpha, beta=beta, sigma=sigma,
                n_samples=n_samples_logZ, rng_seed=rng_seed + k,
            )
        elif method == "ti":
            estimates[k] = ti_logZ(
                n=n, alpha=alpha, beta=beta, sigma=sigma,
                n_steps=16,
                n_samples_per_step=max(500, n_samples_logZ // 16),
                burn_in=max(200, n_samples_logZ // 32),
                rng_seed=rng_seed + k,
            )
        else:
            raise ValueError(f"unknown method {method}")
    log_z_mean = float(estimates.mean())
    log_z_se = float(estimates.std(ddof=1) / math.sqrt(n_chains))
    distances = d_alpha_batch(rankings, sigma, alpha)
    return -beta * distances - log_z_mean, log_z_mean, log_z_se
