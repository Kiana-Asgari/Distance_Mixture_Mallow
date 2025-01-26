import math
import numpy as np
from itertools import permutations
from collections import Counter
import random

from utils import distance_alpha
from estimate_paramaters.optimal_sigma import optimal_sigma
from estimate_paramaters.dispersion_gradient import gradient_dispersion_parametes
from GMM_diagonalized.partition_estimation import partition_estimation


def fit_mallow_baseline(
    permutation_samples,
    alpha_init=2.0,
    beta_init=2.0,
    sigma_init=None,
    max_iter=50000,
    learning_rate_beta=1e-2,
    learning_rate_alpha=1e-2,
    tol=1e-8,
    gtol = 1e-3,
    Delta=5,
    seed=42,
    verbose=False):
        # todo: change delta based on beta and alpha, or add proper lower bounds
        """
        Estimate the parameters beta, alpha, and sigma of the GMM using MLE with exact Z and sampler-based sums.

        Args:
            permutations: List of observed permutations (nd array).
            alpha_init: Initial guess for alpha (>=1).
            beta_init: Initial guess for beta (>0).
            sigma_init: Initial guess for sigma (permutation). If None, uses a preliminary local search.
            max_outer_iters: Maximum number of outer iterations.
            max_sigma_search_iters: Maximum iterations for local search when updating sigma.
            learning_rate_beta: Learning rate for updating beta.
            learning_rate_alpha: Learning rate for updating alpha.
            eps_beta: Convergence threshold for beta.
            eps_alpha: Convergence threshold for alpha.
            eps_sigma: Convergence threshold for sigma distance change.
            verbose: If True, prints detailed progress. 

        Returns:
            beta, alpha, sigma: Estimated parameters.
        """
        print(f'*****************fitting mallow with the implemented baseline*****************')
        m = len(permutation_samples)
        n = len(permutation_samples[0])
        np.random.seed(seed)
        # Initialize parameters
        beta = beta_init
        alpha = alpha_init
        if sigma_init is None:
            sigma, min_cost = optimal_sigma(permutation_samples, alpha)
        else:
            sigma = sigma_init

        if verbose:
            print(f"Initial parameters: beta={beta}, alpha={alpha}, sigma={sigma}")

        for outer_iter in range(max_iter):
            if verbose and outer_iter % 100 == 0:
                print(f"\nIteration {outer_iter + 1}:")
                print(f"  Current beta: {beta}")
                print(f"  Current alpha: {alpha}")
                print(f"  Current sigma: {sigma}")
            ###############################################################
            # --- Step A and B: Update beta and alpha, given sigma ---
            ###############################################################
            # Compute Z using permanent_fast
            Z = partition_estimation(beta=beta, alpha=alpha, sigma=sigma)
            if verbose and outer_iter % 100 == 0:
                print(f"  Computed Z: {Z}")

            if Z == 0:
                if verbose:
                    print("  [Warning] Partition function Z is zero. Skipping beta and alpha updates.")
                break  # Avoid division by zero

            neg_gradient_alpha, neg_gradient_beta = gradient_dispersion_parametes(permutation_samples, beta, alpha, sigma, Delta=Delta)

            # Update beta via gradient ascent
            old_beta = beta
            beta = beta - learning_rate_beta * neg_gradient_beta

            # Ensure beta stays positive
            if beta <= 0:
                beta = 1e-6  # Assign a small positive value
                if verbose and outer_iter % 100 == 0:
                    print("  [Info] Beta adjusted to a small positive value to maintain positivity.")

            if verbose and outer_iter % 100 == 0:
                print(f"  [Step B] Updated beta to {beta:.6f} (gradient={-1*neg_gradient_beta:.6f})")

            # Update alpha via gradient ascent
            old_alpha = alpha
            alpha = alpha - learning_rate_alpha * neg_gradient_alpha

            # Ensure alpha stays >=1
            if alpha < 1:
                alpha = 1.0

            if verbose and outer_iter % 100 == 0:
                print(f"  [Step B] Updated alpha to {alpha:.6f} (gradient={-1*neg_gradient_alpha:.6f})")

            ###############################################################
            # --- Step C: Update sigma, given beta and alpha ---
            ###############################################################
            # Find sigma that minimizes sum d_alpha(pi, sigma)
            new_sigma, min_cost = optimal_sigma(permutation_samples, alpha=alpha)

            # Compute distance change
            sigma_dist_change = distance_alpha(new_sigma, sigma, alpha)

            # Update sigma and cost
            sigma = new_sigma

            if verbose and outer_iter % 100 == 0:
                print(f"  [Step C] Updated sigma to {sigma} , sigma_dist_change={sigma_dist_change:.6f}")

            # --- Step D: Check for convergence ---
            delta_beta = abs(beta - old_beta)
            delta_alpha = abs(alpha - old_alpha)

            if verbose and outer_iter % 100 == 0:
                print(f"  [Convergence Check] delta_beta={delta_beta:.6f}, delta_alpha={delta_alpha:.6f}, sigma_dist_change={sigma_dist_change:.6f}")

            if delta_beta < tol and delta_alpha < tol and sigma_dist_change < tol:
                if verbose:
                    print(f"[estimate_gmm_parameters] Converged at iteration {outer_iter + 1} due to small changes in beta and alpha")
                break

            if np.abs(neg_gradient_alpha) < gtol and np.abs(neg_gradient_beta) < gtol and sigma_dist_change < tol:
                if verbose:
                    print(f"[estimate_gmm_parameters] Converged at iteration {outer_iter + 1} due to small gradient")
                break

        return alpha, beta, sigma

