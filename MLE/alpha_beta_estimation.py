import numpy as np
from scipy.optimize import differential_evolution

from MLE.score_function import psi_m_wrapper



def solve_alpha_beta(pis, sigma, Delta,
                     alpha_bounds=(1e-4, 2),
                     beta_bounds =(1e-4, 2),
                     *,
                     maxiter     = 300,       # Increased from 100
                     popsize     = 150,       # Increased from 50
                     mutation    = (0.1, 0.5), # Wider range for better exploration
                     recombination = 0.9,
                     tol         = 1e-4,      # Decreased from 5*1e-1 for higher accuracy
                     rng_seed    = None,
                     fixed_alpha = False,
                     fixed_alpha_value = 1,
                     ord         = 1/2):  # Added finish_method parameter

    """
    Zeroth-order search for (α̂, β̂) such that Ψ_m(α̂,β̂;σ)=0.
    np.ndarray([α̂, β̂])
    """
    print('fixed_alpha', fixed_alpha)
    if fixed_alpha:
        alpha_bounds = (fixed_alpha_value-1e-3, fixed_alpha_value+1e-3)
    bounds = [alpha_bounds, beta_bounds]
    max_workers = 4

    res = differential_evolution(
        psi_m_wrapper,
        bounds,
        args=(pis, sigma, Delta, ord),
        seed          = 42,
        strategy      = 'best1bin',
        tol           = tol,  # Even tighter tolerance
        maxiter       = maxiter,
        popsize       = popsize,
        mutation      = mutation,
        recombination = recombination,
        polish        = False,
        updating      = "deferred",
        workers       = max_workers,
        init          = 'latinhypercube',  # Better initial population sampling
        disp          = False  # Display progress during optimization
    )

    α_hat, β_hat = res.x  
    #if fixed_alpha:
   #     α_hat = 1
    # α_hat, β_hat = find_root_from_approx(pis, sigma, np.array([α_hat, β_hat]))
    return np.array([α_hat, β_hat])





