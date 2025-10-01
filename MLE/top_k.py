import numpy as np
from GMM_diagonalized.sampling import sample_truncated_mallow
from collections import Counter
from typing import Callable, Sequence, Tuple, List, Optional
from benchmark.fit_placket_luce import sample_PL
from scipy.stats import kendalltau
from benchmark.fit_Mallow_kendal import sample_kendal




import numpy as np


import numpy as np


def evaluate_metrics(test_set, sampled_set):
    """
    Metrics are the expectations  E [f(T, S)]  with T ⟂ S.  They are estimated
    by averaging f over *all* pairs of independent draws (t_i , s_j) – i.e. the
    Cartesian product between the two Monte-Carlo samples.  Everything is fully
    vectorised; no explicit Python loops over samples.

    Returns
    -------
    top_k_hit_rates : list[float]   –  Hit@k  (true top-1 item appears in model top-k)
    spearman_rho    : float         –  ⟨Spearman ρ⟩
    hamming_distance: float         –  ⟨normalised Hamming distance⟩
    kendall_tau     : float         –  ⟨Kendall τ⟩
    ndcg            : float         –  ⟨nDCG⟩  (rel = n−rank, log₂ discount)
    pairwise_acc    : float         –  ⟨pair-wise concordance⟩
    """
    # ---------- array prep ----------
    test_set, sampled_set = _align_0_based(np.asarray(test_set), np.asarray(sampled_set))
    n_t, n_items  = test_set.shape
    n_p, n_items_ = sampled_set.shape
    if n_items != n_items_:
        raise ValueError("Both sets must rank the same number of items.")

    r = np.arange(n_items)

    # inverse permutations:  item → rank

    pos_t = np.empty_like(test_set)
    pos_p = np.empty_like(sampled_set)
    pos_t[np.arange(n_t)[:, None],  test_set]    = r
    pos_p[np.arange(n_p)[:, None],  sampled_set] = r





    # ---------- 1) top-k hit rates ----------
    top1          = test_set[:, 0]                       # (n_t,)
    ranks_top1    = pos_p[:, top1].T                     # (n_t , n_p)
    top_k_hit_rates = [(ranks_top1 < k).mean()
                       for k in range(1, n_items + 1)]

    # ---------- 2) Spearman ρ ----------
    d2_sum        = ((pos_t[:, None, :] - pos_p[None, :, :]) ** 2).sum(-1)
    spearman_rho  = (1 - 6 * d2_sum / (n_items * (n_items**2 - 1))).mean()

    # ---------- 3) Hamming distance ----------
    hamming_distance = (test_set[:, None, :] != sampled_set[None, :, :]).mean()

    # ---------- 4) Kendall τ  &  pairwise accuracy ----------
    # Using scipy.stats.kendalltau for robust computation
    # Average Kendall tau across all test-sampled pairs
    kendall_taus = []
    for test_perm in test_set:
        for sampled_perm in sampled_set:
            tau, _ = kendalltau(test_perm, sampled_perm)
            kendall_taus.append(tau)
    kendall_tau = np.mean(kendall_taus)
    
    # Pairwise accuracy: fraction of concordant pairs
    # Can be computed from kendall_tau as: pairwise_acc = (tau + 1) / 2
    pairwise_acc = (kendall_tau + 1) / 2

    # ---------- 5) nDCG ----------
    ndcg          = 0

    # ---------- 6) Total Variation Distance for first position distribution ----------
    # Extract items ranked first in test and sampled sets
    test_first_items = test_set[:, 0]        # (n_t,)
    sampled_first_items = sampled_set[:, 0]  # (n_p,)
    
    # Compute empirical distributions for first position only
    test_dist = np.bincount(test_first_items, minlength=n_items) / n_t
    sampled_dist = np.bincount(sampled_first_items, minlength=n_items) / n_p
    
    # Total Variation Distance: TVD = 0.5 * sum(|p(x) - q(x)|)
    tvd_first_position = 0.5 * np.sum(np.abs(test_dist - sampled_dist))
    
    # TVD for combined first AND second positions
    # Concatenate first and second position items from both sets
    test_top2_items = np.concatenate([test_set[:, 0], test_set[:, 1]])      # (2*n_t,)
    sampled_top2_items = np.concatenate([sampled_set[:, 0], sampled_set[:, 1]])  # (2*n_p,)
    
    # Compute empirical distributions over combined top-2 positions
    test_dist_top2 = np.bincount(test_top2_items, minlength=n_items) / len(test_top2_items)
    sampled_dist_top2 = np.bincount(sampled_top2_items, minlength=n_items) / len(sampled_top2_items)
    
    # TVD for top-2 combined distribution
    tvd_1and2_position = 0.5 * np.sum(np.abs(test_dist_top2 - sampled_dist_top2))

    # ---------- 7) Mean Average Precision (MAP) ----------
    # For each test ranking, compute Average Precision (AP) against all sampled rankings
    # AP measures how well the top-ranked items in test appear in sampled rankings
    
    # For ranking evaluation: treat test ranking as ground truth, sampled as predictions
    # AP@k = (1/k) * sum_{i=1}^{k} Precision@i * relevance(i)
    # where relevance(i) = 1 if item at position i in sampled is in top-k of test
    
    # Average over all test-sampled pairs
    average_precisions = []
    for test_perm in test_set:
        for sampled_perm in sampled_set:
            # Compute AP for this pair
            # Consider top-k items from test as "relevant"
            # For different k values, we'll use k = min(10, n_items)
            k = min(10, n_items)
            test_top_k = set(test_perm[:k])
            
            # Compute precision at each position in sampled ranking
            precisions = []
            num_relevant_found = 0
            for i in range(1, k + 1):
                if sampled_perm[i-1] in test_top_k:
                    num_relevant_found += 1
                    precision_at_i = num_relevant_found / i
                    precisions.append(precision_at_i)
            
            # Average Precision for this pair
            if len(precisions) > 0:
                ap = np.mean(precisions)
            else:
                ap = 0.0
            average_precisions.append(ap)
    
    mean_average_precision = np.mean(average_precisions)

    metric_results = {
        '@top_1': top_k_hit_rates[0],
        '@top_5': top_k_hit_rates[4],
        '@top_10': top_k_hit_rates[9],
        'tau': float(kendall_tau),
        'map': float(mean_average_precision),
        'tvd12': float(tvd_1and2_position)
    }
    print(metric_results)

    return metric_results







def _align_0_based(test_set, sampled_set):
    test_set = np.asarray(test_set).copy()
    sampled_set = np.asarray(sampled_set).copy()
    for t in range(len(test_set)):
        test_set[t] = test_set[t] - np.min(test_set[t])
    for s in range(len(sampled_set)):
        sampled_set[s] = sampled_set[s] - np.min(sampled_set[s])

    return test_set, sampled_set




