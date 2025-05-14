import numpy as np
import pickle
from sampling import dp_wrapper
# from partition_estimation import Z_wrap
from joblib import Parallel, delayed

# Parameters
def generate_lookup_tables(n):
    Delta = 8          # Truncation width for fast permanent
    alpha_vals = np.linspace(0.1, 3, 40)
    beta_vals = np.linspace(0.05, 2.0, 30)

    # Helper functions for finite difference approximations
    def compute_entry(alpha, beta, n, Delta, h=1e-5):
        Z = dp_wrapper(n, alpha, beta, Delta)
        perm_alpha_plus = dp_wrapper(n, alpha + h, beta, Delta)
        perm_alpha_minus = dp_wrapper(n, alpha - h, beta, Delta)
        perm_beta_plus = dp_wrapper(n, alpha, beta + h, Delta)
        perm_beta_minus = dp_wrapper(n, alpha, beta - h, Delta)

        E_d = -(np.log(perm_beta_plus) - np.log(perm_beta_minus)) / (2 * h)
        E_dot_d = -(np.log(perm_alpha_plus) - np.log(perm_alpha_minus)) / (2 * h) / beta

        return Z, E_d, E_dot_d

    # Parallel computation
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(compute_entry)(alpha, beta, n, Delta)
        for alpha in alpha_vals
        for beta in beta_vals
    )

    # Reshape results to match alpha-beta grid
    Z_table = np.array([r[0] for r in results]).reshape(len(alpha_vals), len(beta_vals))
    E_d_table = np.array([r[1] for r in results]).reshape(len(alpha_vals), len(beta_vals))
    E_dot_d_table = np.array([r[2] for r in results]).reshape(len(alpha_vals), len(beta_vals))

    # Save lookup tables
    lookup_data = {
        'n': n,
        'alpha_vals': alpha_vals,
        'beta_vals': beta_vals,
        'Z_table': Z_table,
        'E_d_table': E_d_table,
        'E_dot_d_table': E_dot_d_table
    }

    filename = f'mallows_lookup_tables_n{n}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(lookup_data, f)

    print(f"Lookup tables for n={n} successfully saved to {filename}.")

# Example usage
if __name__ == "__main__":
    generate_lookup_tables(5)
    # generate_lookup_tables(10)
    # generate_lookup_tables(15)
    # generate_lookup_tables(20)