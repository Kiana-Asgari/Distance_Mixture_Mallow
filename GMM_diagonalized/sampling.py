import numpy as np
import random
import math
from utils import distance_alpha



def metropolis_hastings_sampler(beta, alpha, sigma, Z, num_samples=10000, burn_in=1000, thin=10):
    """
    Metropolis-Hastings sampler for the GMM.

    Args:
        beta: Dispersion parameter.
        alpha: Distance parameter.
        sigma: Reference permutation (tuple or list).
        Z: Partition function Z_n(beta, alpha).
        num_samples: Number of samples to collect.
        burn_in: Number of initial samples to discard.
        thin: Thinning factor to reduce autocorrelation.

    Returns:
        samples: List of sampled permutations.
    """
    n = len(sigma)
    current = list(sigma)
    samples = []
    accepted = 0
    total = 0

    def proposal(current):
        """Generate a neighboring permutation by swapping two elements."""
        i, j = random.sample(range(n), 2)
        new = current.copy()
        new[i], new[j] = new[j], new[i]
        return new, i, j  # Return swapped indices for log_diff

    # Compute current distance
    current_d = distance_alpha(current, sigma, alpha)

    for _ in range(burn_in + num_samples * thin):
        new, i, j = proposal(current)
        new_d = distance_alpha(new, sigma, alpha)
        acceptance_ratio = math.exp(-beta * (new_d - current_d))
        if random.random() < min(1, acceptance_ratio):
            current = new
            current_d = new_d
            accepted += 1
        total += 1

        if _ >= burn_in and (_ - burn_in) % thin == 0:
            samples.append(tuple(current))

    acceptance_rate = accepted / total
    print(f"Metropolis-Hastings Acceptance Rate: {acceptance_rate:.2f}")
    return samples