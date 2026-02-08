import numpy as np
from GMM_diagonalized.sampling import sample_truncated_mallow
from collections import Counter
from typing import Callable, Sequence, Tuple, List
from benchmark.fit_placket_luce import sample_PL
from scipy.stats import kendalltau
from benchmark.fit_Mallow_kendal import sample_kendal

from real_world_datasets.utils import check_zero_based_index
import numpy as np
from sklearn.metrics import ndcg_score
from scipy.stats import kendalltau, spearmanr

def evaluate_metrics(test_sets, sampled_sets):
    """
    Evaluate the metrics for the sampled set.
    test_set: a 2D list of multiple ground truth permutations, each permutation is a list of integers
    sampled_set: a 2D list of multiple predicted permutations, each permutation is a list of integers
    return: a dictionary of the metrics
    """
    
    # ---------- array prep ----------
    test_set = check_zero_based_index(test_sets)
    sampled_set = check_zero_based_index(sampled_sets)
    
    n_queries = len(test_set)
    
    # Initialize metric lists
    recalls = {k: [] for k in [1, 2, 5, 10]}
    precisions = {k: [] for k in [1, 2, 5, 10]}
    mrrs = []
    ndcgs = {k: [] for k in [1, 5, 10]}
    hammings = []
    pairwise_accs = []
    kendall_taus = []
    spearman_rhos = []
    
    for true_rank, pred_rank in zip(test_set, sampled_set):
        true_rank = list(true_rank)
        pred_rank = list(pred_rank)
        
        # Recall and Precision at k
        for k in [1, 2, 5, 10]:
            k = min(k, len(true_rank))
            test_relevant = set(true_rank[:k])
            top_k_pred = pred_rank[:k]
            
            hits = sum(item in test_relevant for item in top_k_pred)
            recalls[k].append(hits / k)
            precisions[k].append(hits / k)
        
        # MRR - find rank of first relevant item (top-10 from ground truth)
        test_relevant = set(true_rank[:10])
        mrr = next((1/(i+1) for i, item in enumerate(pred_rank) 
                   if item in test_relevant), 0)
        mrrs.append(mrr)
        
        # NDCG at k - use ground truth positions as relevance scores
        for k in [1, 5, 10]:
            k = min(k, len(true_rank))
            # Relevance: higher score for items that appear earlier in ground truth
            true_pos = {item: len(true_rank) - i for i, item in enumerate(true_rank)}
            
            top_k_pred = pred_rank[:k]
            
            if k == 1:
                # For k=1, NDCG is 1 if top item is correct, 0 otherwise
                ndcg = 1.0 if pred_rank[0] == true_rank[0] else 0.0
            else:
                y_true = np.array([[true_pos.get(item, 0) for item in top_k_pred]])
                y_score = np.arange(k, 0, -1).reshape(1, -1)
                
                ndcg = ndcg_score(y_true, y_score) if y_true.sum() > 0 else 0
            
            ndcgs[k].append(ndcg)
        
        # Hamming distance - number of positions where items differ
        min_len = min(len(true_rank), len(pred_rank))
        hamming = sum(true_rank[i] != pred_rank[i] for i in range(min_len))
        hammings.append(hamming / min_len)  # Normalized
        
        # Pairwise accuracy - fraction of pairs with correct relative order
        n = len(true_rank)
        true_pos = {item: i for i, item in enumerate(true_rank)}
        pred_pos = {item: i for i, item in enumerate(pred_rank)}
        
        correct_pairs = 0
        total_pairs = 0
        
        for i in range(n):
            for j in range(i+1, n):
                item_i, item_j = true_rank[i], true_rank[j]
                if item_i in pred_pos and item_j in pred_pos:
                    if pred_pos[item_i] < pred_pos[item_j]:
                        correct_pairs += 1
                    total_pairs += 1
        
        pairwise_acc = correct_pairs / total_pairs if total_pairs > 0 else 0
        pairwise_accs.append(pairwise_acc)
        
        # Kendall's tau
        tau, _ = kendalltau(true_rank, pred_rank)
        kendall_taus.append(tau if not np.isnan(tau) else 0)
        
        # Spearman's rho
        rho, _ = spearmanr(true_rank, pred_rank)
        spearman_rhos.append(rho if not np.isnan(rho) else 0)
    
    metric_results = {
        '@recall_1': np.mean(recalls[1]),
        '@recall_2': np.mean(recalls[2]),
        '@recall_5': np.mean(recalls[5]),
        '@recall_10': np.mean(recalls[10]),
        '@precision_1': np.mean(precisions[1]),
        '@precision_2': np.mean(precisions[2]),
        '@precision_5': np.mean(precisions[5]),
        '@precision_10': np.mean(precisions[10]),
        '@mmr': np.mean(mrrs),
        '@ndcg1': np.mean(ndcgs[1]),
        '@ndcg5': np.mean(ndcgs[5]),
        '@ndcg10': np.mean(ndcgs[10]),
        'hamming': np.mean(hammings),
        'pairwise_acc': np.mean(pairwise_accs),
        'Kendall_tau': np.mean(kendall_taus),
        'Spearman_rho': np.mean(spearman_rhos),
    }
    
    return metric_results