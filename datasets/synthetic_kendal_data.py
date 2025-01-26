import numpy as np
import random




def generate_synthetic_kendal(n, beta, sigma, num_samples):
    samples = mallows_gibbs_sampling(n=n, phi=beta, reference_permutation=sigma,
                                      num_samples=num_samples)
    return np.array(samples)


def kendall_tau_distance(perm1, perm2):
    """
    Calculate the Kendall tau distance between two permutations.
    """
    n = len(perm1)
    inv_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (perm1[i] < perm1[j]) != (perm2[i] < perm2[j]):
                inv_count += 1
    return inv_count

def mallows_gibbs_sampling(n, phi, num_samples, reference_permutation=None):
    """
    Generate permutations from the Mallows model using Gibbs sampling.

    Parameters:
        n: Number of elements in the permutation.
        phi: Dispersion parameter (phi > 0, where 1 means no preference, and lower values mean tighter distribution).
        num_samples: Number of samples to generate.
        reference_permutation: The center of the distribution (defaults to the identity permutation if None).

    Returns:
        A list of sampled permutations.
    """
    if reference_permutation is None:
        reference_permutation = list(range(n))  # Default to the identity permutation
    
    def sample_position(current_permutation, i):
        """
        Sample a new position for element i using the conditional probabilities.
        """
        weights = []
        for j in range(i + 1):
            temp_perm = current_permutation[:j] + [i] + current_permutation[j:i] + current_permutation[i + 1:]
            distance = kendall_tau_distance(reference_permutation, temp_perm)
            weights.append(phi ** distance)
        weights = np.array(weights) / sum(weights)
        new_pos = np.random.choice(range(i + 1), p=weights)
        return new_pos

    # Start with a random permutation
    current_permutation = list(range(n))
    random.shuffle(current_permutation)

    samples = []

    for _ in range(num_samples):
        for i in range(n):
            # Sample a new position for each element
            current_permutation.remove(i)
            new_position = sample_position(current_permutation, i)
            current_permutation.insert(new_position, i)
        samples.append(current_permutation[:])  # Append a copy of the current permutation

    return samples