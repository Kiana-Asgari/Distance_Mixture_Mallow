import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from MLE.score_function import create_interpolators
from MLE.score_function import psi_m_wrapper, psi_m_wrapper_no_alpha, psi_m, load_lookup_tables











def solve_alpha_beta(pis, sigma, Delta,
                     alpha_bounds=(1e-2, 3),
                     beta_bounds =(1e-2, 2),
                     *,
                     maxiter     = 100,       # Increased from 100
                     popsize     = 100,       # Increased from 50
                     mutation    = (0.1, 0.5), # Wider range for better exploration
                     recombination = 0.9,
                     tol         = 1e-2,      # Decreased from 5*1e-1 for higher accuracy
                     rng_seed    = None,
                     fixed_alpha = False,
                     fixed_alpha_value = 1,
                     ord         = 1/2):  # Added finish_method parameter
    if 0 in pis[0]:
        pis = pis + 1
    max_workers = 5
    if fixed_alpha:
        res = differential_evolution(
        psi_m_wrapper_no_alpha,
        bounds=[beta_bounds],
        args=(pis, sigma, Delta, fixed_alpha_value, ord),
        seed          = 42,
        strategy      = 'best1bin',
        tol           = tol,  # Even tighter tolerance
        maxiter       = maxiter,
        popsize       = popsize,
        mutation      = mutation,
        recombination = recombination,
        polish        = True,
        updating      = "deferred",
        workers       = max_workers,
        init          = 'latinhypercube',  # Better initial population sampling
        disp          = False  # Display progress during optimization
        )

        β_hat = res.x[0]
        α_hat = fixed_alpha_value
    else:
        bounds = [alpha_bounds, beta_bounds]


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
            polish        = True,
            updating      = "deferred",
            workers       = max_workers,
            init          = 'latinhypercube',  # Better initial population sampling
            disp          = False  # Display progress during optimization
        )

        α_hat, β_hat = res.x  

    return np.array([α_hat, β_hat])



