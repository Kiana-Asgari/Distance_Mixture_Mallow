import numpy as np
from scipy.optimize import linear_sum_assignment, differential_evolution

from MLE.score_function import psi_m, load_lookup_tables
from GMM_diagonalized.sampling import sample_truncated_mallow

import numpy as np
import multiprocessing as mp
from functools import partial

from scipy.optimize import (
    brute,                 # grid search, has workers>=1
    differential_evolution
)


def learn_spearman(permutations_train):
    sigma_hat = consensus_ranking_estimation_L2(permutations_train)
    beta_hat = solve_alpha_beta(permutations_train, sigma_hat, beta_bounds=(1e-4, 3.0))
    return sigma_hat, beta_hat

def sample_spearman(beta, sigma, num_samples=1000):
    return sample_truncated_mallow(n=len(sigma), alpha=2, beta=beta, sigma=sigma, Delta=8, num_samples=num_samples)

def psi_m_wrapper_L2(x, pis, sigma):
    lookup_data = load_lookup_tables(len(sigma))
    ψ = psi_m(pis, sigma, 2, x[0], lookup_data)
    return np.dot(ψ, ψ)

def solve_alpha_beta(pis, sigma,
                     beta_bounds =(1e-4, 3.0),
                     *,
                     num_mc      = 300,   # high-precision MC only for the final polish
                     maxiter     = 10,      # few generations are enough
                     popsize     = 50,     # small population → fast
                     mutation    = (0.5, 1),  # Smaller mutation range for finer steps
                     recombination = 0.9,    # Higher recombination for better local search
                     rng_seed    = None):  # Added finish_method parameter

    """
    Zeroth-order search for (α̂, β̂) such that Ψ_m(α̂,β̂;σ)=0.
    np.ndarray([α̂, β̂])
    """
    bounds = [beta_bounds]
    max_workers = 32


    # Use a function with args instead of a lambda
    res = differential_evolution(
        psi_m_wrapper_L2,
        bounds,
        args=(pis, sigma),  
        seed          = 42,
        #strategy      = "rand1bin",
        tol           = 5*1e-1,
        maxiter       = maxiter,
        #popsize       = popsize,
        #mutation      = mutation,
        #recombination = 0.9,
        #polish        = True,        # we'll replace polish with LS below
        #updating      = "deferred",
        workers       = 1,
        strategy='best1exp'
    )

    β_hat = res.x  
    # α_hat, β_hat = find_root_from_approx(pis, sigma, np.array([α_hat, β_hat]))
    return β_hat


def consensus_ranking_estimation_L2(pis: np.ndarray) -> np.ndarray:
    pis = np.asarray(pis)
    m, n = pis.shape
    C = np.zeros((n, n), dtype=float)

    positions = np.arange(1, n + 1)

    # Compute cost incrementally
    for i in range(n):
        C[i] = (((pis[:, i][:, None] - positions) ** 2).sum(axis=0))

    row_ind, col_ind = linear_sum_assignment(C)
    sigma_hat = col_ind + 1
    return sigma_hat