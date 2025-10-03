import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from MLE.score_function import create_interpolators
from MLE.score_function import psi_m_wrapper, psi_m_wrapper_no_alpha, psi_m, load_lookup_tables

def plot_3D_landscape(sigma, pis, Delta):
    lookup_data = load_lookup_tables(len(sigma), Delta)
    E_d_interp, E_dot_d_interp = create_interpolators(lookup_data)
    alpha_vals = lookup_data['alpha_vals']
    beta_vals = lookup_data['beta_vals']
    beta_vals = np.array([b for b in beta_vals if b < 0.5 and b > 0.1])
    alpha_vals = np.array([a for a in alpha_vals if a < 1.5 and a > 0.6])
    print(f'alpha_vals shape {alpha_vals.shape}')
    print(f'beta_vals shape {beta_vals.shape}')
    
    # Create meshgrid for 3D plotting
    Alpha, Beta = np.meshgrid(alpha_vals, beta_vals, indexing='ij')
    
    # Evaluate on the grid
    d_vals = np.zeros_like(Alpha)
    d_dot_vals = np.zeros_like(Alpha)
    d_norm_vals = np.zeros_like(Alpha)
    d_emp_vals = np.zeros_like(Alpha)
    d_dot_emp_vals = np.zeros_like(Alpha)
    d_mc_vals = np.zeros_like(Alpha)
    d_dot_mc_vals = np.zeros_like(Alpha)
    for i, alpha in enumerate(alpha_vals):
        for j, beta in enumerate(beta_vals):
            hat_psi_m, d_emp, d_dot_emp, d_mc, d_dot_mc = psi_m(pis, sigma, alpha, beta, lookup_data, return_all=True)
            print(f'alpha {alpha}, beta {beta}, d_emp {d_emp}, d_dot_emp {d_dot_emp}, d_mc {d_mc}, d_dot_mc {d_dot_mc}')
            d_vals[i, j] = hat_psi_m[0]
            d_dot_vals[i, j] = hat_psi_m[1]
            d_norm_vals[i, j] = np.linalg.norm(hat_psi_m, ord=2)
            d_emp_vals[i, j] = d_emp
            d_dot_emp_vals[i, j] = d_dot_emp
            d_mc_vals[i, j] = d_mc
            d_dot_mc_vals[i, j] = d_dot_mc
    # Create 3D plots for each metric
    all_infos = [ d_norm_vals, d_mc_vals, d_dot_mc_vals]
    labels = ['||ψ||', 'theoretical E[d]', 'theoretical E[d·log(d)]']
    
    # Create all figures before showing them
    for data, label in zip(all_infos, labels):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create surface plot
        surf = ax.plot_surface(Alpha, Beta, data, cmap='viridis', 
                              alpha=0.8, edgecolor='none', antialiased=True)
        
        # Customize the plot
        ax.set_xlabel('Alpha (α)', fontsize=12)
        ax.set_ylabel('Beta (β)', fontsize=12)
        ax.set_zlabel(label, fontsize=12)
        ax.set_title(f'3D Landscape: {label}', fontsize=14, fontweight='bold')
        
        # Add colorbar
        fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.1)
        
        # Rotate for better viewing angle
        ax.view_init(elev=25, azim=45)
    
    plt.show()

def solve_alpha_beta(pis, sigma, Delta,
                     alpha_bounds=(1e-1, 3),
                     beta_bounds =(1e-2, 2),
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
        print(f'pis shape {pis.shape}')
        # pis = np.array([pis[0]]) #pis[np.abs(pis - sigma).mean(axis=1) <= 10]
        # print(f'pis shape after dropping: {pis.shape}')
        #plot_3D_landscape(sigma, pis, Delta)
        # alpha_hat, beta_hat = grid_search(pis, sigma, Delta)
        # return np.array([alpha_hat, beta_hat])

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
    #beta_vals = np.array([b for b in beta_vals if b < 0.5 and b > 0.1])
    #alpha_vals = np.array([a for a in alpha_vals if a < 1.5 and a > 0.5])
    # drop the alpha and beta values if their index is even
    alpha_vals = np.array(alpha_vals[::32])
    beta_vals = np.array(beta_vals[::16])
    psi0_values = {}
    psi1_values = {}
    bestbeta_for_alpha = {}

    for alpha in alpha_vals:
        bestbeta_for_alpha[alpha] = 0
        min_err = 100
        for i, beta in enumerate(beta_vals):
            psi_value = (psi_m(pis, sigma, alpha, beta, lookup_data))
            psi0_values[(alpha, beta)] = psi_value[0] 
            psi1_values[(alpha, beta)] = psi_value[1] 
            if np.abs(psi_value[0]) <= min_err:
                min_err = np.abs(psi_value[0])
                bestbeta_for_alpha[alpha] = {'beta': beta, 'psi': psi_value}
    min_err = 100
    for i, alpha in enumerate(alpha_vals):
        if np.abs(bestbeta_for_alpha[alpha]["psi"][1]) < min_err:
            min_err = np.abs(bestbeta_for_alpha[alpha]["psi"][1])
            bestalpha_for_beta = alpha
        print(f'alpha {alpha:2.2f}, beta {bestbeta_for_alpha[alpha]["beta"]:2.2f}, error {bestbeta_for_alpha[alpha]["psi"][1]:2.2f}')
    print(f'best alpha {bestalpha_for_beta:2.2f}, beta {bestbeta_for_alpha[bestalpha_for_beta]["beta"]:2.2f}, error {bestbeta_for_alpha[bestalpha_for_beta]["psi"]}')

    return np.array([bestalpha_for_beta, bestbeta_for_alpha[bestalpha_for_beta]["beta"]])