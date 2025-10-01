import numpy as np
from scipy.optimize import differential_evolution

from MLE.score_function import psi_m_wrapper, psi_m_wrapper_no_alpha, psi_m, load_lookup_tables



def solve_alpha_beta(pis, sigma, Delta,
                     alpha_bounds=(1e-1, 2.5),
                     beta_bounds =(1e-3, 2),
                     *,
                     maxiter     = 300,       # Increased from 100
                     popsize     = 150,       # Increased from 50
                     mutation    = (0.1, 0.5), # Wider range for better exploration
                     recombination = 0.4,
                     tol         = 1e-5,      # Decreased from 5*1e-1 for higher accuracy
                     rng_seed    = None,
                     fixed_alpha = False,
                     fixed_alpha_value = 1,
                     ord         = 1/2):  # Added finish_method parameter
    if not fixed_alpha:
        alpha_hat, beta_hat = grid_search(pis, sigma, Delta)
        return np.array([alpha_hat, beta_hat])

    """
    Zeroth-order search for (α̂, β̂) such that Ψ_m(α̂,β̂;σ)=0.
    np.ndarray([α̂, β̂])
    """
    print('fixed_alpha', fixed_alpha)
    max_workers = 5
    tol = 1e-5
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
    print(f'model is done with alpha {α_hat:.4f} and beta {β_hat:.4f}, error {res.fun:.4f}, achieved {res.success}, num iterations {res.nit}')
    #if fixed_alpha:
   #     α_hat = 1
    # α_hat, β_hat = find_root_from_approx(pis, sigma, np.array([α_hat, β_hat]))
    return np.array([α_hat, β_hat])



from MLE.score_function import psi_m


def grid_search(pis, sigma, Delta):
    lookup_data = load_lookup_tables(len(sigma), Delta=Delta)
    alpha_vals = lookup_data['alpha_vals']
    beta_vals = lookup_data['beta_vals']
    psi0_values = {}
    psi1_values = {}
    bestbeta_for_alpha = {}

    for alpha in alpha_vals:
        bestbeta_for_alpha[alpha] = 0
        min_err = 100
        for beta in beta_vals:
            psi_value = (psi_m(pis, sigma, alpha, beta, lookup_data))
            psi0_values[(alpha, beta)] = psi_value[0] 
            psi1_values[(alpha, beta)] = psi_value[1] 
            if np.abs(psi_value[0]) < min_err:
                min_err = np.abs(psi_value[0])
                bestbeta_for_alpha[alpha] = {'beta': beta, 'psi': psi_value}
    min_err = 100
    for alpha in alpha_vals:
        if np.abs(bestbeta_for_alpha[alpha]["psi"][1]) < min_err:
            min_err = np.abs(bestbeta_for_alpha[alpha]["psi"][1])
            bestalpha_for_beta = alpha
        print(f'alpha {alpha:2.2f}, beta {bestbeta_for_alpha[alpha]["beta"]:2.2f}, error {bestbeta_for_alpha[alpha]["psi"][1]:2.2f}')
    print(f'best alpha {bestalpha_for_beta:2.2f}, beta {bestbeta_for_alpha[bestalpha_for_beta]["beta"]:2.2f}, error {bestbeta_for_alpha[bestalpha_for_beta]["psi"]}')

    return np.array([1, 1])