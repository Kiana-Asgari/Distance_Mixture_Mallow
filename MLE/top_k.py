import numpy as np
from GMM_diagonalized.sampling import sample_truncated_mallow
from collections import Counter
from typing import Callable, Sequence, Tuple, List
from benchmark.fit_placket_luce import sample_PL
from scipy.stats import kendalltau
from benchmark.fit_Mallow_kendal import sample_kendal

def soft_top_k_kendal(test_set, theta_hat, pi_0, num_mc=20_000):
    samples = sample_kendal(num_samples=num_mc, theta=theta_hat, sigma_0=pi_0)
    # Convert samples to numpy array if it's not already
    samples = 1 + np.array(samples)
    print('   kendal samples', samples[0])
    
    n = len(pi_0)
    # compute a list of length n, the probability that the i th position in the test set is in the top 1 of the samples
    top_1_marginal = [np.sum(samples[:, 0] == i)/ len(samples) for i in range(1, n+1)] 
    hit_rate_list = []
    for k in range(1, n+1):
        hit = 0
        for test_ranking in test_set:
            for position in range(k):
                hit += top_1_marginal[test_ranking[position]-1] # raked 1 in the test set
        hit_rate_list.append(hit / len(test_set))
    distances = distance_metrics(test_set, samples)
    ndcg = ndcg_at_k(test_set, samples)
    return hit_rate_list, distances, ndcg


def soft_top_k_PL(test_set, utilities, num_mc=20_000):
    samples = 1+sample_PL(utilities, 20_000)
    # Convert samples to numpy array if it's not already
    samples = np.array(samples)
    
    n = len(utilities)
    # compute a list of length n, the probability that the i th position in the test set is in the top 1 of the samples
    top_1_marginal = [np.sum(samples[:, 0] == i)/ len(samples) for i in range(1, n+1)] 
    hit_rate_list = []
    for k in range(1, n+1):
        hit = 0
        for test_ranking in test_set:
            for position in range(k):
                hit += top_1_marginal[test_ranking[position]-1] # raked 1 in the test set
        hit_rate_list.append(hit / len(test_set))
    distances = distance_metrics(test_set, samples)
    ndcg = ndcg_at_k(test_set, samples)
    return hit_rate_list, distances, ndcg


def soft_top_k(test_set, alpha_hat, beta_hat, sigma_hat, Delta=None, rng_seed=None, num_mc=20_000):
    samples = sample_truncated_mallow(num_samples=num_mc, 
                                      n=len(sigma_hat), 
                                      beta=beta_hat, 
                                      alpha=alpha_hat, 
                                      sigma=sigma_hat,
                                      Delta=Delta,
                                      rng_seed=rng_seed)
    # Convert samples to numpy array if it's not already
    samples = np.array(samples)
    
    n = len(sigma_hat)
    # compute a list of length n, the probability that the i th position in the test set is in the top 1 of the samples
    top_1_marginal = [np.sum(samples[:, 0] == i)/ len(samples) for i in range(1, n+1)] 
    hit_rate_list = []
    for k in range(1, n+1):
        hit = 0
        for test_ranking in test_set:
            for position in range(k):
                hit += top_1_marginal[test_ranking[position]-1] # raked 1 in the test set
        hit_rate_list.append(hit / len(test_set))
    distances = distance_metrics(test_set, samples)
    ndcg = ndcg_at_k(test_set, samples)
    return hit_rate_list, distances, ndcg

def distance_metrics(test_set, MC_set, *, cayley_samples: int = 50_000, rng_seed: int | None = None):
    """
    Compute similarity / distance metrics between an empirical test set of
    permutations (`test_set`) and a Monte-Carlo sample (`MC_set`).

    Returned
    --------
    kendall_tau : float
        Expected Kendall-τ correlation coefficient.
    hamming_distance : float
        Expected Hamming distance (number of positions that differ).
    spearman_rho : float
        Expected Spearman rank-correlation coefficient.
    cayley_distance : float
        Expected Cayley distance (minimum number of transpositions).
        Obtained by Monte-Carlo pairing (`cayley_samples` pairs).
    """
    # ------------------------------------------------------------------ #
    # Helper functions                                                   #
    # ------------------------------------------------------------------ #
    def item_position_probs(perms: np.ndarray) -> np.ndarray:
        """P[item, position] = probability that `item` is placed at `position`."""
        m, n = perms.shape
        items     = perms - 1                                 # 0-based item labels
        positions = np.tile(np.arange(n), (m, 1))

        counts = np.zeros((n, n), dtype=np.int64)
        np.add.at(counts, (items.ravel(), positions.ravel()), 1)
        return counts.astype(np.float64) / m

    def precedence_probs(perms: np.ndarray, batch_size: int = 2_000) -> np.ndarray:
        """P[i, j] = probability that i precedes j (i ≠ j)."""
        m, n = perms.shape
        pos = np.argsort(perms, axis=1)                       # positions[row, item]
        counts = np.zeros((n, n), dtype=np.int64)

        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            pos_batch = pos[start:end]                        # (b, n)
            precedes  = pos_batch[:, :, None] < pos_batch[:, None, :]
            counts   += precedes.sum(axis=0)

        return counts.astype(np.float64) / m

    # ------------------------------------------------------------------ #
    # Standardise inputs                                                 #
    # ------------------------------------------------------------------ #
    test_set = np.asarray(test_set, dtype=np.int64)
    MC_set   = np.asarray(MC_set,   dtype=np.int64)

    if test_set.ndim != 2 or MC_set.ndim != 2:
        raise ValueError("Inputs must be 2-D array-like collections of permutations")
    if test_set.shape[1] != MC_set.shape[1]:
        raise ValueError("Permutations in both sets must have the same length")

    n_items   = test_set.shape[1]
    n_pairs   = n_items * (n_items - 1) // 2                  # #unordered pairs
    positions = np.arange(1, n_items + 1, dtype=np.float64)   # 1-based positions
    pos_sq    = positions ** 2

    # ------------------------------------------------------------------ #
    # Expected Hamming distance                                          #
    # ------------------------------------------------------------------ #
    P_test_pos = item_position_probs(test_set)                # (n, n)
    P_MC_pos   = item_position_probs(MC_set)

    match_prob_per_pos = (P_test_pos * P_MC_pos).sum(axis=0)
    expected_hamming   = n_items - match_prob_per_pos.sum()

    # ------------------------------------------------------------------ #
    # Expected Kendall-τ                                                 #
    # ------------------------------------------------------------------ #
    P_test_prec = precedence_probs(test_set)                   # (n, n)
    P_MC_prec   = precedence_probs(MC_set)

    discord = (
        P_test_prec * (1.0 - P_MC_prec) +
        (1.0 - P_test_prec) * P_MC_prec
    )
    upper_mask        = np.triu(np.ones_like(discord, dtype=bool), k=1)
    expected_discord  = discord[upper_mask].sum()
    kendall_tau       = 1.0 - 2.0 * expected_discord / n_pairs

    # ------------------------------------------------------------------ #
    # Expected Spearman ρ                                                #
    # ------------------------------------------------------------------ #
    # Per-item mean / second-moment of position under each distribution
    mean_test     = (P_test_pos @ positions)                  # (n,)
    mean_MC       = (P_MC_pos   @ positions)
    mean_sq_test  = (P_test_pos @ pos_sq)
    mean_sq_MC    = (P_MC_pos   @ pos_sq)

    expected_D = (
        mean_sq_test + mean_sq_MC - 2.0 * mean_test * mean_MC
    ).sum()                                                   # E[ Σ d_i² ]

    denom        = n_items * (n_items ** 2 - 1)
    spearman_rho = 1.0 - 6.0 * expected_D / denom

    # ------------------------------------------------------------------ #
    # Expected Cayley distance (Monte-Carlo estimate)                    #
    # ------------------------------------------------------------------ #
    def cayley_dist_pair(perm_a: np.ndarray, perm_b: np.ndarray) -> int:
        """
        Cayley distance between two permutations = n - #cycles( a⁻¹ ∘ b ).
        Both inputs are 1-D numpy arrays of 1…n.
        """
        n = perm_a.size
        inv_a = np.empty(n, dtype=np.int64)
        inv_a[perm_a - 1] = np.arange(n)                      # a⁻¹

        gamma   = inv_a[perm_b - 1]                           # composition a⁻¹∘b
        visited = np.zeros(n, dtype=bool)
        cycles  = 0

        for start in range(n):
            if not visited[start]:
                cycles += 1
                j = start
                while not visited[j]:
                    visited[j] = True
                    j = gamma[j]
        return n - cycles

    # Use a manageable number of random pairs to avoid     #
    # quadratic blow-up.                                   #
    rng          = np.random.default_rng(rng_seed)
    total_pairs  = test_set.shape[0] * MC_set.shape[0]
    if cayley_samples >= total_pairs:                        # exact enumeration
        idx_test = np.repeat(np.arange(test_set.shape[0]), MC_set.shape[0])
        idx_MC   = np.tile  (np.arange(MC_set.shape[0]),    test_set.shape[0])
    else:                                                    # Monte-Carlo sample
        idx_test = rng.integers(0, test_set.shape[0], size=cayley_samples)
        idx_MC   = rng.integers(0, MC_set.shape[0],   size=cayley_samples)

    cayley_vals = np.fromiter(
        (cayley_dist_pair(test_set[i], MC_set[j]) for i, j in zip(idx_test, idx_MC)),
        dtype=np.float64,
        count=idx_test.size,
    )
    expected_cayley = cayley_vals.mean()

    # ------------------------------------------------------------------ #
    return kendall_tau, expected_hamming, spearman_rho, expected_cayley

def ndcg_at_k(test_perms, sampled_perms, k: int = 5) -> float:
    """
    Compute the (mean) Normalised Discounted Cumulative Gain @ k.

    Parameters
    ----------
    test_perms    : array-like, shape (m, n)
        Ground-truth permutations.
    sampled_perms : array-like, shape (N, n)
        Permutations sampled from the model.  They are aggregated to build
        a single *predicted* ranking by counting how often each item appears
        in the top-k positions.
    k             : int (default=5)
        Depth at which NDCG is evaluated.

    Returns
    -------
    float
        Mean NDCG@k over the `m` test permutations.
    """
    # ------------------------------------------------------------------ #
    # Standardise / sanity-check inputs                                  #
    # ------------------------------------------------------------------ #
    test_perms    = np.asarray(test_perms,    dtype=np.int64)
    sampled_perms = np.asarray(sampled_perms, dtype=np.int64)

    if test_perms.ndim != 2 or sampled_perms.ndim != 2:
        raise ValueError("Inputs must be 2-D collections of permutations")
    if test_perms.shape[1] != sampled_perms.shape[1]:
        raise ValueError("Permutations in both inputs must have the same length")

    n_items = test_perms.shape[1]
    k       = min(k, n_items)                    # guard against k > n_items

    # ------------------------------------------------------------------ #
    # 1) Build a single *predicted* ranking from the Monte-Carlo sample  #
    #    by counting occurrences in the top-k positions.                 #
    # ------------------------------------------------------------------ #
    topk_MC_items = sampled_perms[:, :k].ravel()         # (N · k,)
    counts        = np.bincount(topk_MC_items - 1,
                                minlength=n_items)       # 0-based indexing
    ranked_items  = np.argsort(-counts)[:k] + 1          # 1…n label space
    # ranked_items.shape == (k,)

    # ------------------------------------------------------------------ #
    # 2) Compute relevance matrix:                                       #
    #       R[i, j] = 1  iff  ranked_items[j] present in                 #
    #                          top-k of test permutation i.              #
    # ------------------------------------------------------------------ #
    test_topk  = test_perms[:, :k]                        # (m, k)
    # Broadcast comparison: (m, k, 1) == (1, 1, k)  →  (m, k, k)
    relevance  = (test_topk[:, :, None] == ranked_items[None, None, :]) \
                 .any(axis=1).astype(np.float64)          # (m, k)

    # ------------------------------------------------------------------ #
    # 3) DCG and IDCG per test permutation                               #
    # ------------------------------------------------------------------ #
    # Discount weights 1 / log2( rank + 1 )  for ranks starting at 1
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))  # (k,)

    dcg   = (relevance * discounts).sum(axis=1)           # (m,)

    rel_cnt      = relevance.sum(axis=1).astype(np.int64)  # number of relevant
    cum_disc     = np.cumsum(discounts)                   # partial sums
    idcg         = np.where(rel_cnt > 0,
                            cum_disc[rel_cnt - 1],
                            0.0)

    ndcg = np.where(idcg > 0, dcg / idcg, 0.0)

    # ------------------------------------------------------------------ #
    return ndcg.mean()
    