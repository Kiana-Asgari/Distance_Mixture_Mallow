from MLE.consensus_ranking_estimation import consensus_ranking_estimation
from MLE.alpha_beta_estimation import solve_alpha_beta
import numpy as np
import json
import os
import pandas as pd
from numpy.random import default_rng
import requests
from MLE.top_k import soft_top_k, soft_top_k_PL, soft_top_k_kendal, evaluate_in_sample_metrics
from benchmark.fit_placket_luce import learn_PL
from benchmark.fit_Mallow_kendal import learn_kendal

def fit_apa(n_trials=10, train_ratio=0.8, Delta=3, seed=42):
    np.random.seed(seed)  # For reproducibility
    data = pd.read_csv('APA/APA_data.csv').values
    print(f'APA data loaded: {data.shape}')

    train_size = int(data.shape[0] * train_ratio)
    results_file = 'APA/results/APA_fit_results.json'

    # Check if results file exists and load existing results
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing trials")
    else:
        results = []

    for trial in range(n_trials):
        if trial < len(results):
            print(f"Skipping trial {trial+1} (already computed)")
            continue

        print(f"Running trial {trial+1} of {n_trials}")

        train_data, test_data, train_indices, test_indices = train_split(data, train_size, trial + seed)

        # Fit consensus ranking
        sigma_0 = consensus_ranking_estimation(train_data)
        print(f"     Consensus ranking estimated: {sigma_0}")
        alpha, beta = solve_alpha_beta(train_data, sigma_0, Delta=Delta)
        print(f"     Alpha and beta estimated: {alpha}, {beta}")

        # Evaluate ML Model
        top_hit_rates, distances, ndcg = soft_top_k(test_data, alpha, beta, sigma_0, Delta, rng_seed=42)
        print(f"     [ML] Top-k hit rates: {top_hit_rates}")

        # Evaluate PL Model
        pl_utilities, _ = learn_PL(train_data - 1, test_data - 1)
        top_hit_rates_PL, distances_PL, ndcg_PL, pairwise_acc_PL = soft_top_k_PL(test_data, pl_utilities)
        print(f"     [PL] Top-k hit rates: {top_hit_rates_PL}")

        # Evaluate Kendall Model
        pi_0, theta_hat, _ = learn_kendal(train_data - 1, test_data - 1)
        top_hit_rates_kendal, distances_kendal, ndcg_kendal, pairwise_acc_kendal = soft_top_k_kendal(test_data, theta_hat, pi_0)
        print(f"     [Kendal] Top-k hit rates: {top_hit_rates_kendal}")

        train_metrics = evaluate_in_sample_metrics(train_data, alpha, beta, sigma_0)
        print("     Train metrics calculated")

        trial_results = {
            'alpha': alpha.tolist() if isinstance(alpha, np.ndarray) else alpha,
            'beta': beta.tolist() if isinstance(beta, np.ndarray) else beta,
            'sigma_0': sigma_0.tolist() if isinstance(sigma_0, np.ndarray) else sigma_0,
            'top_hit_rates': top_hit_rates,
            'distances': distances,
            'ndcg': ndcg,
            'pairwise': train_metrics.get("acc"),
            'top_hit_rates_PL': top_hit_rates_PL,
            'distances_PL': distances_PL,
            'pairwise_PL': pairwise_acc_PL,
            'ndcg_PL': ndcg_PL,
            'top_hit_rates_kendal': top_hit_rates_kendal,
            'pairwise_kendal': pairwise_acc_kendal,
            'distances_kendal': distances_kendal,
            'ndcg_kendal': ndcg_kendal,
            'test_indices': test_indices.tolist()
        }

        results.append(trial_results)

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"     Trial {trial+1} results saved")

    print(f"Completed {len(results)} trials. Results saved to '{results_file}'")


def train_split(data, train_size, seed):
    rng = default_rng(seed)
    indices = np.arange(len(data))
    rng.shuffle(indices)
    return data[indices[:train_size]], data[indices[train_size:]], indices[:train_size], indices[train_size:]

# Example usage:
# fit_apa(n_trials=10, train_ratio=0.8, Delta=7, seed=42)
