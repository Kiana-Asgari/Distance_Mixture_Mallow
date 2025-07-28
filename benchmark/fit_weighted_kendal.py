import numpy as np
from scipy.optimize import minimize
from scipy.stats import kendalltau

import math
import numpy as np
from scipy.stats import kendalltau


def compute_dcg_deltas(n):
    """
    Compute delta_i = 1/log2(i+1) - 1/log2(i+2) for i in 1..n (0-indexed).
    This is the DCG-inspired position discount difference.
    """
    deltas = []
    for i in range(n):
        d1 = 1.0 / math.log2(i + 2)       # 1 / log2(i+2)
        d2 = 1.0 / math.log2(i + 3)       # 1 / log2(i+3)
        deltas.append(d2-d1)
    return deltas


def compute_p_vector(deltas):
    """
    Compute the cumulative position potentials p_i = sum_{j=1}^i delta_j
    """
    p = [0.0] * len(deltas)
    p[0] = deltas[0]
    for i in range(1, len(deltas)):
        p[i] = p[i - 1] + deltas[i]
    return p


def compute_bar_p_vector(p, sigma):
    """
    Compute \bar{p}_i(sigma) = (p_i - p_{sigma(i)}) / (i - sigma(i))
    If i == sigma(i), then \bar{p}_i = 1
    """
    n = len(sigma)
    bar_p = [1.0] * n
    for i in range(n):
        si = sigma[i]
        if i != si:
            numerator = p[i] - p[si]
            denominator = i - si
            bar_p[i] = numerator / denominator
    return bar_p


def kendall_tau_distance(rank1, rank2):
    """
    Weighted Kendall tau using DCG-style position weights from the paper.
    """
    n = len(rank1)
    # Compute inverse permutation of rank2
    rank2_inv = [0] * n
    for i in range(n):
        rank2_inv[rank2[i]] = i

    # Compute deltas and potentials
    deltas = compute_dcg_deltas(n)
    p = compute_p_vector(deltas)
    bar_p = compute_bar_p_vector(p, rank2_inv)

    # Count weighted inversions
    distance = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if (rank2_inv[rank1[i]] > rank2_inv[rank1[j]]):
                distance += bar_p[i] * bar_p[j]

    return distance


def borda_count(rankings):
    n = len(rankings[0])
    scores = np.zeros(n)
    for rank in rankings:
        for i, item in enumerate(rank):
            scores[item] += i
    return np.argsort(scores)

def z_theta(theta, n):
    prod = 1.0
    for j in range(1, n + 1):
        prod *= (1 - np.exp(-j * theta)) / (1 - np.exp(-theta))
    return prod

def negative_log_likelihood(theta, rankings, pi_0):
    n = len(pi_0)
    total_distance = sum(kendall_tau_distance(pi, pi_0) for pi in rankings)
    return theta * total_distance + len(rankings) * np.log(z_theta(theta, n))

def learn_weighted_kendal(permutations_train, permutations_test):
    pi_0 = borda_count(permutations_train)
    result = minimize(negative_log_likelihood, x0=1.0, args=(permutations_train, pi_0), bounds=[(0.01, None)])
    theta_hat = result.x[0]
    error = 1/len(permutations_test) * negative_log_likelihood(theta_hat,permutations_test, pi_0)
    return pi_0, theta_hat, error


###########################################
#sample from the learned model
###########################################
import random
import math

def sample_weighted_kendal(theta, sigma_0, num_samples=1000):
    """
    Sample permutations from a Mallows model (Kendall's tau distance)
    with central ranking sigma_0 and concentration parameter theta.

    The insertion is reversed so that each new item in sigma_0
    is more likely to be inserted near the *end* of the partial permutation
    (rather than the front), matching the forward order given by sigma_0:

      - Start with an empty list.
      - For each item in sigma_0 (in the given order),
        you generate an insertion position in [0..len(perm)] 
        with weight propto exp(-theta*(reverse_index)).

      - The far right of the current permutation has the highest weight.
    """
    sampled_perms = []

    for _ in range(num_samples):
        perm = []
        for item in sigma_0:
            length = len(perm)
            # Compute unnormalized weights so that the *end* is favored
            # position goes from 0..length
            # reverse_index = length - position
            weights = [math.exp(-theta * (length - k)) for k in range(length + 1)]
            total = sum(weights)

            r = random.random() * total
            cumsum = 0
            for k, w in enumerate(weights):
                cumsum += w
                if r <= cumsum:
                    perm.insert(k, item)
                    break

        sampled_perms.append(perm)

    return np.array(sampled_perms)