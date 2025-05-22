from sushi_dataset.load_data import load_sushi
from MLE.consensus_ranking_estimation import consensus_ranking_estimation
from MLE.alpha_beta_estimation import solve_alpha_beta
import numpy as np
from numpy.random import default_rng
import json
import os
import sys
from MLE.top_k import evaluate_metrics
from benchmark.fit_placket_luce import sample_PL, learn_PL
from GMM_diagonalized.sampling import sample_truncated_mallow
from benchmark.fit_Mallow_kendal import learn_kendal,sample_kendal
from benchmark.fit_Mallow_spearman import learn_spearman, sample_spearman

def fit_and_save_sushi(seed=42, num_trials=25, training_ratio=0.8, Delta=7):

    sushi_data = load_sushi()    
    results_file = 'sushi_dataset/results/sushi_fit_results.json'
    #train_size = int(len(sushi_data) * training_ratio)
    train_size = 3500
    # Create directory if it doesn't exist
    os.makedirs('sushi_dataset/results', exist_ok=True)
    
    # Check if results file exists and load existing results
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing trials")
    else:
        results = []
    
    for trial in range(num_trials):
        # Skip if this trial has already been computed
        if trial < len(results):
            print(f"Skipping trial {trial+1} (already computed)")
            continue
            
        print(f"Running trial {trial+1} of {num_trials}")
        
        train_data, test_data, train_indices, test_indices = train_split(sushi_data, train_size, trial + seed)
        
         ##########################################################
        # L_alpha Mallow model
        ##########################################################
        sigma_0 = consensus_ranking_estimation(train_data)
        alpha, beta = solve_alpha_beta(train_data, sigma_0, Delta=Delta)
        sampled_set = sample_truncated_mallow(n=len(sigma_0), alpha=alpha, beta=beta, sigma=sigma_0, Delta=Delta, num_samples=20_000)
        top_k_hit_rates_ML, spearman_rho_ML, hamming_distance_ML, kendall_tau_ML, ndcg_ML, pairwise_acc_ML = evaluate_metrics(test_data, sampled_set)
        ##########################################################
        # PL model
        ##########################################################
        pl_utilities, nll = learn_PL(train_data-1, test_data-1)
        sampled_set = sample_PL(utilities=pl_utilities, n_samples=20_000)
        top_k_hit_rates_PL, spearman_rho_PL, hamming_distance_PL, kendall_tau_PL, ndcg_PL, pairwise_acc_PL = evaluate_metrics(test_data, sampled_set)
        ##########################################################
        # Kendal model
        ##########################################################
        pi_0, theta_hat, _ = learn_kendal(train_data-1, test_data-1)
        sampled_set = sample_kendal(sigma_0=sigma_0, theta=theta_hat, num_samples=20_000)
        top_k_hit_rates_kendal, spearman_rho_kendal, hamming_distance_kendal, kendall_tau_kendal, ndcg_kendal, pairwise_acc_kendal = evaluate_metrics(test_data, sampled_set)
        ##########################################################
        # Spearman model
        ##########################################################
        sigma_hat_spearman, beta_hat_spearman = learn_spearman(train_data-1)
        sampled_set = sample_spearman(beta=beta_hat_spearman, sigma=sigma_hat_spearman, num_samples=20_000)
        top_k_hit_rates_spearman, spearman_rho_spearman, hamming_distance_spearman, kendall_tau_spearman, ndcg_spearman, pairwise_acc_spearman = evaluate_metrics(test_data, sampled_set)
        print(type(sigma_hat_spearman))
        print(type(beta_hat_spearman))
        print(type(top_k_hit_rates_spearman))
        print(type(spearman_rho_spearman))
        print(type(hamming_distance_spearman))
        print(type(kendall_tau_spearman))
        print(type(ndcg_spearman))
        print(type(pairwise_acc_spearman))
        # Store results for this trial
        trial_results = {
            'alpha': alpha.tolist() if isinstance(alpha, np.ndarray) else alpha,
            'beta': beta.tolist() if isinstance(beta, np.ndarray) else beta,
            'sigma_0': sigma_0.tolist() if isinstance(sigma_0, np.ndarray) else sigma_0,
            'top_k_hit_rates_ML': top_k_hit_rates_ML,
            'spearman_rho_ML': spearman_rho_ML,
            'hamming_distance_ML': hamming_distance_ML,
            'kendall_tau_ML': kendall_tau_ML,
            'ndcg_ML': ndcg_ML,
            'pairwise_acc_ML': pairwise_acc_ML,
            'top_k_hit_rates_PL': top_k_hit_rates_PL,
            'spearman_rho_PL': spearman_rho_PL,
            'hamming_distance_PL': hamming_distance_PL,
            'kendall_tau_PL': kendall_tau_PL,
            'ndcg_PL': ndcg_PL,
            'pairwise_acc_PL': pairwise_acc_PL,
            'top_k_hit_rates_kendal': top_k_hit_rates_kendal,
            'spearman_rho_kendal': spearman_rho_kendal,
            'hamming_distance_kendal': hamming_distance_kendal,
            'kendall_tau_kendal': kendall_tau_kendal,
            'ndcg_kendal': ndcg_kendal,
            'pairwise_acc_kendal': pairwise_acc_kendal,
            'beta_hat_spearman': float(beta_hat_spearman),
            'sigma_hat_spearman': sigma_hat_spearman.tolist() if isinstance(sigma_hat_spearman, np.ndarray) else sigma_hat_spearman,
            'top_k_hit_rates_spearman': top_k_hit_rates_spearman.tolist() if isinstance(top_k_hit_rates_spearman, np.ndarray) else top_k_hit_rates_spearman,
            'spearman_rho_spearman': float(spearman_rho_spearman),
            'hamming_distance_spearman': float(hamming_distance_spearman),
            'kendall_tau_spearman': kendall_tau_spearman,
            'ndcg_spearman': ndcg_spearman,
            'pairwise_acc_spearman': pairwise_acc_spearman,
            }
        
        print(f"     Trial {trial+1} results saved with data: {trial_results}")
        results.append(trial_results)
        
        # Save results after each trial
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"     Trial {trial+1} results saved")
    
    print(f"Completed {len(results)} trials. Results saved to '{results_file}'")


def train_split(sushi_data, train_size, seed):
    # Randomly select indices for training using the RNG
    # Initialize random number generator with seed
    rng = default_rng(seed)
    all_indices = np.arange(len(sushi_data))
    rng.shuffle(all_indices)
    train_indices = all_indices[:train_size]
    test_indices = all_indices[train_size:]
    
    # Split data into training and testing sets
    train_data = sushi_data[train_indices]
    test_data = sushi_data[test_indices]
    return train_data, test_data, train_indices, test_indices

