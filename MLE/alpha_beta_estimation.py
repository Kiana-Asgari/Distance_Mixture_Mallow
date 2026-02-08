from re import U
import numpy as np
from scipy.optimize import differential_evolution, least_squares
import matplotlib.pyplot as plt
import torch
from MLE.score_function import create_interpolators
from MLE.score_function import psi_m_wrapper, psi_m_wrapper_no_alpha, psi_m, load_lookup_tables


def solve_alpha_beta(pis, sigma, Delta, 
                        alpha_bounds=(1e-1, 3),
                        beta_bounds =(1e-3, 2),
                        rng_seed    = None,
                        fixed_alpha = False,
                        fixed_alpha_value = 1,
                        ord         = 1/2,
                        n_trials   = 1,
                        tol_GD     = 1e-3,
                        tol = 1e-4, 
                        use_lookup_tables = False):

    if len(pis[0]) <= 20:   
        use_lookup_tables = False
        return solve_alpha_beta_least_squares(pis, sigma, Delta, alpha_bounds, beta_bounds, rng_seed, fixed_alpha, fixed_alpha_value, ord, n_trials, tol_GD, tol, use_lookup_tables)
    else:
        return solve_alpha_beta_differential_evolution(pis=pis, 
        sigma=sigma, 
        Delta=Delta, 
        alpha_bounds=alpha_bounds, 
        beta_bounds=beta_bounds, 
        rng_seed=rng_seed, 
        fixed_alpha=fixed_alpha, 
        fixed_alpha_value=fixed_alpha_value, 
        use_lookup_tables=True)






def solve_alpha_beta_least_squares(pis, sigma, Delta, 
                        alpha_bounds=(1e-1, 3),
                        beta_bounds =(1e-3, 2),
                        rng_seed    = None,
                        fixed_alpha = False,
                        fixed_alpha_value = 1,
                        ord         = 1/2,
                        n_trials   = 1,
                        tol_GD     = 1e-3,
                        tol = 1e-4, 
                        use_lookup_tables = False):
    if len(pis[0]) <= 20:
        use_lookup_tables = False
    else:
        use_lookup_tables = True
    
    def residual(params):
        alpha, beta = params if not fixed_alpha else (fixed_alpha_value, params[0])
        residuals = psi_m(pis, sigma, alpha, beta, Delta, use_lookup_tables=use_lookup_tables)
        return residuals
    
    rng = np.random.default_rng(rng_seed)
    best_result, best_cost = None, np.inf
    
    for _ in range(n_trials):
        x0 = [rng.uniform(*beta_bounds)] if fixed_alpha else [rng.uniform(*alpha_bounds), rng.uniform(*beta_bounds)]
        bounds = ([beta_bounds[0]], [beta_bounds[1]]) if fixed_alpha else ([alpha_bounds[0], beta_bounds[0]], [alpha_bounds[1], beta_bounds[1]])
        
        result = least_squares(residual,
                               x0,
                               bounds=bounds,
                               ftol=tol,
                               xtol=tol,
                               gtol=tol_GD,
                               verbose=1,
                               max_nfev=1000)
        
        if result.cost < best_cost:
            best_cost = result.cost
            best_result = result

    α_hat, β_hat = (fixed_alpha_value, best_result.x[0]) if fixed_alpha else best_result.x
    print(f'Converged: alpha={α_hat:.4f}, beta={β_hat:.4f}, cost={best_cost:.4f}')
    return np.array([α_hat, β_hat])





def solve_alpha_beta_differential_evolution(pis, sigma, Delta,
                     alpha_bounds=(1e-1, 3),
                     beta_bounds =(1e-2, 2),
                     maxiter     = 300,       # Increased from 100
                     popsize     = 150,       # Increased from 50
                     mutation    = (0.1, 0.5), # Wider range for better exploration
                     recombination = 0.4,
                     tol         = 1e-2,      # Decreased from 5*1e-1 for higher accuracy
                     rng_seed    = None,
                     fixed_alpha = False,
                     fixed_alpha_value = 1,
                     ord         = 1/2,
                     use_lookup_tables = False):  # Added finish_method parameter

    print('fixed_alpha', fixed_alpha)
    max_workers = 5

    if fixed_alpha:
        res = differential_evolution(
                                    psi_m_wrapper_no_alpha,
                                    bounds=[beta_bounds],
                                    args=(pis, sigma, Delta, fixed_alpha_value, ord, use_lookup_tables),
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
        #return grid_search(pis, sigma, Delta)

        res = differential_evolution(
            psi_m_wrapper,
            bounds,
            args=(pis, sigma, Delta, ord, use_lookup_tables),
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
    print(f'model is done with alpha {α_hat:.4f} and beta {β_hat:.4f}, error {res.fun:.4f}, achieved {res.success}, num iterations {res.nit}')

    return np.array([α_hat, β_hat])



from MLE.score_function import psi_m


def grid_search(pis, sigma, Delta):
    lookup_data = load_lookup_tables(len(sigma), Delta=Delta)
    alpha_vals = lookup_data['alpha_vals']
    beta_vals = lookup_data['beta_vals']
    #beta_vals = np.array([b for b in beta_vals if b < 0.5 and b > 0.1])
    #alpha_vals = np.array([a for a in alpha_vals if a < 1.5 and a > 0.5])
    # drop the alpha and beta values if their index is even
    alpha_vals = np.array(alpha_vals[::32])
    beta_vals = np.array(beta_vals[::16])
    psi0_values = {}
    psi1_values = {}
    bestbeta_for_alpha = {}
    

    for alpha in alpha_vals:
        min_err = np.inf
        for i, beta in enumerate(beta_vals):
            psi_value = (psi_m(pis, sigma, alpha, beta, lookup_data))
            psi0_values[(alpha, beta)] = psi_value[0] 
            psi1_values[(alpha, beta)] = psi_value[1] 
            if np.abs(psi_value[0]) < min_err:
                min_err = np.abs(psi_value[0])
                bestbeta_for_alpha[alpha] = {'beta': beta, 'psi': psi_value}
    min_err = np.inf
    for i, alpha in enumerate(alpha_vals):
        if np.abs(bestbeta_for_alpha[alpha]["psi"][1]) < min_err:
            min_err = np.abs(bestbeta_for_alpha[alpha]["psi"][1])
            bestalpha_for_beta = alpha
        print(f'alpha {alpha:2.2f}, beta {bestbeta_for_alpha[alpha]["beta"]:2.2f}, error {bestbeta_for_alpha[alpha]["psi"][1]:2.2f}')
    print(f'best alpha {bestalpha_for_beta:2.2f}, beta {bestbeta_for_alpha[bestalpha_for_beta]["beta"]:2.2f}, error {bestbeta_for_alpha[bestalpha_for_beta]["psi"]}')

    return np.array([bestalpha_for_beta, bestbeta_for_alpha[bestalpha_for_beta]["beta"]])