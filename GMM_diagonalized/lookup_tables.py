import numpy as np
import pickle
import csv
from sampling import dp_wrapper
# from partition_estimation import Z_wrap
from joblib import Parallel, delayed

# Parameters
def generate_lookup_tables(n, Delta):
    print(f'Starting to generate lookup tables for n={n}, Delta={Delta}.')
    alpha_vals = np.linspace(0.1, 3, 200)
    beta_vals_unstandardized = np.linspace(0.1, 25, 100)


    # Helper functions for finite difference approximations
    def compute_entry(alpha, beta_unstandardized, n, Delta, h=1e-5):

        beta = beta_unstandardized/ (n**alpha) # BETA = beta/n^alpha (standardized beta)
        #print(f'Starting to compute entry for alpha={alpha}, beta={beta}.')
        Z = dp_wrapper(n, alpha, beta, Delta)
        perm_alpha_plus = dp_wrapper(n, alpha + h, beta, Delta)
        perm_alpha_minus = dp_wrapper(n, alpha - h, beta, Delta)
        perm_beta_plus = dp_wrapper(n, alpha, beta + h, Delta)
        perm_beta_minus = dp_wrapper(n, alpha, beta - h, Delta)

        E_d = -(np.log(perm_beta_plus) - np.log(perm_beta_minus)) / (2 * h)
        E_dot_d = -(np.log(perm_alpha_plus) - np.log(perm_alpha_minus)) / (2 * h) 
        E_dot_d = E_dot_d/ beta
        print(f'Done for alpha={alpha}, beta={beta}, beta_unstandardized={beta_unstandardized}.')
        return Z, E_d, E_dot_d

    # Parallel computation
    results = Parallel(n_jobs=20, verbose=10)(
        delayed(compute_entry)(alpha, beta, n, Delta)
        for alpha in alpha_vals
        for beta in beta_vals_unstandardized
    )

    # Reshape results to match alpha-beta grid
    Z_table = np.array([r[0] for r in results]).reshape(len(alpha_vals), len(beta_vals_unstandardized))
    E_d_table = np.array([r[1] for r in results]).reshape(len(alpha_vals), len(beta_vals_unstandardized))
    E_dot_d_table = np.array([r[2] for r in results]).reshape(len(alpha_vals), len(beta_vals_unstandardized))

    # Save lookup tables in pickle format
    lookup_data = {
        'n': n,
        'alpha_vals': alpha_vals,
        'beta_vals_unstandardized': beta_vals_unstandardized,
        'Z_table': Z_table,
        'E_d_table': E_d_table,
        'E_dot_d_table': E_dot_d_table
    }

    import os
    pickle_filename = f'GMM_diagonalized/lookup_tables/mallows_lookup_tables_n{n}_Delta{Delta}.pkl'
    os.makedirs(os.path.dirname(pickle_filename), exist_ok=True)
    with open(pickle_filename, 'wb') as f:
        pickle.dump(lookup_data, f)
    print(f'Lookup tables saved to {pickle_filename}')


# Example usage
if __name__ == "__main__":
    #generate_lookup_tables(5)
    # generate_lookup_tables(10)
    for n in [10]:
        for Delta in [7]:
            generate_lookup_tables(n=n, Delta=Delta)
