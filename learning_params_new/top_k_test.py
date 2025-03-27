import random
import math
import itertools
import numpy as np
from collections import defaultdict
from GMM_diagonalized.sampling import sample_truncated_mallow



def top_k_accuracy(sampled_perms, true_ranking=None, top_k=5):

    # If no true_ranking is given, assume the canonical ascending ordering
    if true_ranking is None:
        n = len(sampled_perms[0])
        true_ranking = np.arange(n)

    # Identify the set of items in the top-k of the true ranking
    top_k_true = set(true_ranking[:top_k])

    total_precision = 0.0
    for i, predicted_perm in enumerate(sampled_perms):
        # Find the top-k items from the predicted permutation
        top_k_predicted = set(predicted_perm[:top_k])
        # Count how many are correct
        correct_count = len(top_k_true & top_k_predicted)
        precision_k = correct_count / top_k
        total_precision += precision_k

        # (Optional) print intermediate progress
        if i % 10 == 0:
            print(f"iter {i}: sample={predicted_perm}, top-k precision={precision_k:.4f}")

    # Average top-k precision across all sampled permutations
    mean_precision = total_precision / len(sampled_perms)
    print(f"Mean top-{top_k} precision: {mean_precision:.4f}")
    return mean_precision





def _top_k_accuracy(sigma, alpha, beta, top_k=5, Delta =10,true_ranking=None):
    print(f"top_k_accuracy: sigma={sigma}, alpha={alpha}, beta={beta}, top_k={top_k}, Delta={Delta}, true_ranking={true_ranking}")
    n = len(sigma)
    if true_ranking is None:
        true_ranking = 1 + np.arange(n)
    n_samples = 1000
    n_correct = 0
    for iter in range(n_samples):
        sampled_perm = sample_truncated_mallow(n, beta, alpha, Delta)
        composed_perm = compose_permutations(sampled_perm, sigma)

        # For demonstration, we just sum how many match in top_k positions
        # (or you could adapt to use the updated top_k_accuracy idea above)
        n_correct += sum(composed_perm[i] == true_ranking[i] for i in range(top_k))
        if iter % 100 == 0:
            match_fraction = sum(composed_perm[i] == true_ranking[i] for i in range(top_k)) / top_k
            print(f"iter {iter}: sample={composed_perm}, match fraction={match_fraction:.4f}")

    top_k_accuracy = n_correct / (top_k * n_samples)
    return top_k_accuracy

def compose_permutations(p, q):
    n = len(p)
    r = np.empty(n, dtype=int)
    for i in range(n):
        r[i] = p[q[i]]
    return r

