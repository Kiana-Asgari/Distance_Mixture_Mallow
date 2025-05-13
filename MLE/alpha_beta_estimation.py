# ===========================================================
#  Additional bounded solvers for  Ψm(α,β;σ)=0
#  -----------------------------------------------------------
#  • solve_alpha_beta_bisection  – 1-D bisection on Ψ₁, Ψ₂   (root_scalar, "bisect")
#  • solve_alpha_beta_brent      – 1-D Brent on Ψ₁, Ψ₂        (root_scalar, "brentq")
#  • solve_alpha_beta_fixed_point– Picard iteration           (fixed_point)
#  • solve_alpha_beta_grid       – exhaustive grid            (optimize.brute, workers)
#
#  All solvers need the two helpers already in your package:
#  psi_m          (returns np.r_[Ψ₁,Ψ₂])
#  psi_m_wrapper  (returns ‖Ψ‖²  – same as used by DE)
# ===========================================================
from MLE.score_function import psi_m, psi_m_wrapper

import numpy as np
import multiprocessing as mp
from functools import partial

from scipy.optimize import (
    brute,                 # grid search, has workers>=1
    differential_evolution
)



# ==============================================================
#  GRID SEARCH – SciPy brute with multiprocessing
# ==============================================================
def solve_alpha_beta_grid(pis, sigma,
                          alpha_bounds=(1., 3.),
                          beta_bounds=(1e-3, 3.),
                          *,
                          num_mc   = 2000,    # Increased from 500
                          grid_N   = 64,      # Increased from 30
                          tol_root = 1e-1,    # Tighter tolerance
                          rng_seed = None,
                          workers  = 1,
                          finish_method = 'BFGS'):  # Add local optimization
    """
    Exhaustive grid search with SciPy.optimize.brute (supports workers≥1).
    """
    ranges = (
        slice(alpha_bounds[0], alpha_bounds[1], complex(grid_N)),
        slice(beta_bounds[0],  beta_bounds[1],  complex(grid_N))
    )

    obj = partial(psi_m_wrapper, pis=pis, sigma=sigma,
                  num_mc=num_mc, rng_seed=rng_seed)

    # Use finish=True to enable local optimization after grid search
    result = brute(obj,
                   ranges, 
                   full_output=True,  # Get full output for diagnostics
                   finish=finish_method,  # Add local refinement
                   workers=workers)
    
    if finish_method:
        α_hat, β_hat = result[0]  # When finish is used, result is (x, fval, gri

    if np.max(np.abs(psi_m(pis,sigma,α_hat,β_hat,
                           num_mc=num_mc, rng_seed=rng_seed))) > tol_root:
        print("[Warning] Grid search did not locate a root within tolerance", tol_root)
    return np.array([α_hat, β_hat])

import sys
def solve_alpha_beta(pis, sigma,
                     alpha_bounds=(1.0, 3.0),
                     beta_bounds =(1e-3, 3.0),
                     *,
                     num_mc      = 2500,   # high-precision MC only for the final polish
                     maxiter     = 15,      # few generations are enough
                     popsize     = 50,     # small population → fast
                     mutation    = (0.5, 1),  # Smaller mutation range for finer steps
                     recombination = 0.9,    # Higher recombination for better local search
                     rng_seed    = None):  # Added finish_method parameter

    """
    Zeroth-order search for (α̂, β̂) such that Ψ_m(α̂,β̂;σ)=0.
    np.ndarray([α̂, β̂])
    """
    bounds = [alpha_bounds, beta_bounds]
    max_workers = 16
    if max_workers > 16:
        sys.exit('max_workers must be less than 8')

    # Use a function with args instead of a lambda
    res = differential_evolution(
        psi_m_wrapper,
        bounds,
        args=(pis, sigma, num_mc, rng_seed),   # cheap MC here
        seed          = rng_seed,
        #strategy      = "rand1bin",
        tol           = 5*1e-1,
        maxiter       = maxiter,
        #popsize       = popsize,
        #mutation      = mutation,
        #recombination = 0.9,
        polish        = True,        # we'll replace polish with LS below
        updating      = "deferred",
        workers       = max_workers            # all CPU cores
    )

    α_hat, β_hat = res.x  
    return np.array([α_hat, β_hat])