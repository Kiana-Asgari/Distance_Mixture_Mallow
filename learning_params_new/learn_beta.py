from GMM_diagonalized.partial_partition_estimation import marginal_probabilities
from utils import distance_alpha_batch
from GMM_diagonalized.DP_partition_estimation import get_partition_estimate_via_dp
import numpy as np
from scipy.optimize import root
from scipy.optimize import bisect





def learn_beta(permutation_samples,
              alpha,
              sigma,
              beta_init,
              Delta,
              max_iter=5000,
              learning_rate=1e-4,
              tol=1e-8,
              gtol=1e-3,
              verbose=False,
              lambda_reg=0,
              seed=42):
    """
    Perform root finding to solve for beta where the gradient is zero.
    """
    print(f'learning beta with beta_init: {beta_init} for alpha: {alpha}:')
    np.random.seed(seed)
    m, n = len(permutation_samples), len(permutation_samples[0])

    # Define a function for root finding that returns the gradient

    root = bisect(lambda beta: nll_and_grad_beta([beta], permutation_samples,
                                    alpha=alpha, sigma=sigma, Delta=Delta, lambda_reg=lambda_reg),
                                    a=0, b=2, xtol=1e-4, maxiter=100)
    print(f'root: {root}')
    return root

    for beta_hat in np.linspace(0.4,1e-6, 30):
        nll, grad = nll_and_grad_beta([beta_hat], permutation_samples,
                                    alpha=alpha, sigma=sigma, Delta=Delta, lambda_reg=lambda_reg)
        if grad < 1e-4:
            break

    # Use scipy's root to find where the gradient is zero

    return beta_hat
    



def nll_and_grad_beta(args, permutation_samples,
                      alpha,
                      sigma,
                      Delta,
                      verbose=False,
                      lambda_reg=1,
                      eps=1e-3):
    
    n = len(permutation_samples[0])
    m = len(permutation_samples)
    beta = args[0]
    beta_2 = beta + eps

    distance_matrix = distance_alpha_batch(perms=permutation_samples, sigma=sigma, alpha=alpha)
    partition= get_partition_estimate_via_dp(n=n, beta=beta, alpha=alpha, Delta=Delta)
    partition_2 = get_partition_estimate_via_dp(n=n, beta=beta_2, alpha=alpha, Delta=Delta)

    nll = 1/m * (beta * np.sum(distance_matrix) + m * np.log(partition))

    grad_beta = 1/m * np.sum(distance_matrix)  +  (np.log(partition_2) - np.log(partition)) / eps
    print(f'   nll: {nll:3f}, grad_beta: {grad_beta:3f}, beta: {beta:3f}')

    return grad_beta














