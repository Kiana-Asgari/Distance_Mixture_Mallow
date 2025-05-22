from MLE.consensus_ranking_estimation import consensus_ranking_estimation
from MLE.alpha_beta_estimation import solve_alpha_beta
import numpy as np
import json
import os
import sys
from numpy.random import default_rng

from benchmark.fit_Mallow_spearman import learn_spearman, sample_spearman
from benchmark.fit_placket_luce import sample_PL, learn_PL
from benchmark.fit_Mallow_kendal import learn_kendal, sample_kendal
from football.load_football import load_data, get_top_teams_borda, get_full_rankings
from football.load_recent_football_data import load_football_data
from GMM_diagonalized.sampling import sample_truncated_mallow
from MLE.top_k import evaluate_metrics
"""
logging the error, best beta, best sigma for each alpha
for each n_file and desired_teams there is a different json file

"""

def fit_football(training_ratio=0.8, Delta=7, seed=42, n_trials=50, n_teams_to_keep=10):
    full_data = load_football_data(n_teams_to_keep=n_teams_to_keep)
    train_size = int(len(full_data) * training_ratio)
    results_file = f'football/results/football_2019_n_top_teams={n_teams_to_keep}(chronological).json'

    
    # Create directory if it doesn't exist
    os.makedirs('football/results', exist_ok=True)
    
    # Check if results file exists and load existing results
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing trials")
    else:
        results = []

    for trial in range(n_trials):
        train_data, test_data, _, _ = train_split(full_data, train_size, trial + seed)


       ##########################################################
        # L_alpha Mallow model
        ##########################################################
        sigma_0 = consensus_ranking_estimation(train_data)
        print(f"     Consensus ranking estimated", sigma_0)
        alpha, beta = solve_alpha_beta(train_data, sigma_0, Delta=Delta)
        print(f"     Alpha and beta estimated", alpha, beta)
        sampled_set = sample_truncated_mallow(n=n_teams_to_keep, alpha=alpha, beta=beta, sigma=sigma_0, Delta=6, num_samples=10_000)
        print(f"     L_alpha model sampled")
        top_k_hit_rates_ML, spearman_rho_ML, hamming_distance_ML, kendall_tau_ML, ndcg_ML, pairwise_acc_ML = evaluate_metrics(test_data, sampled_set)
        print(f"     L_alpha model done with top1={top_k_hit_rates_ML[0]}")
        ##########################################################
        # PL model
        ##########################################################
        pl_utilities, nll = learn_PL(train_data-1, test_data-1)
        print(f"     PL model done with utilities={pl_utilities}")
        sampled_set = sample_PL(utilities=pl_utilities, n_samples=10_000)
        print(f"     PL model done sampling")
        top_k_hit_rates_PL, spearman_rho_PL, hamming_distance_PL, kendall_tau_PL, ndcg_PL, pairwise_acc_PL = evaluate_metrics(test_data, sampled_set)
        print(f"     PL model done with top1={top_k_hit_rates_PL[0]}")
        ##########################################################
        # Kendal model
        ##########################################################
        pi_0, theta_hat, _ = learn_kendal(train_data-1, test_data-1)
        print(f"     Kendal model done with pi_0={pi_0}")
        sampled_set = sample_kendal(sigma_0=sigma_0, theta=theta_hat, num_samples=10_000)
        print(f"     Kendal model done sampling")
        top_k_hit_rates_kendal, spearman_rho_kendal, hamming_distance_kendal, kendall_tau_kendal, ndcg_kendal, pairwise_acc_kendal = evaluate_metrics(test_data, sampled_set)
        print(f"     Kendal model done with top1={top_k_hit_rates_kendal[0]}")
        # Store results for this trial
 # Store results for this trial
        trial_results = {
            'full_data_size': len(full_data),
            'train_size': len(train_data),
            'test_size': len(test_data),
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
            }
        
        print(f"     Trial {trial+1} results saved with data: {trial_results}")
        results.append(trial_results)
        
        # Save results after each trial
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"     Trial {trial+1} results saved")
    
    print(f"Completed {len(results)} trials. Results saved to '{results_file}'")

























##########################################################
#old code
#######################################################
def _fit_football(n_file, n_top_teams=11, n_bottom_teams=10, 
                                   Delta=7, seed=42):
    np.random.seed(seed)  # For reproducibility

    teams, votes_dict = load_data(limit=n_file)

    top_teams = get_top_teams_borda(teams, votes_dict)
    print(f'top_teams: {top_teams}')
    # desired_teams = 1+np.concatenate([top_teams[1:n_top_teams], top_teams[-n_bottom_teams:-5]])
    desired_teams = top_teams[::8] 
    football_data = get_full_rankings(teams, votes_dict, which_team_to_keep = desired_teams[:10])
    print(f'football_data: {football_data.shape}')
    print(f"Loaded {len(results)} existing trials")
    

    num_trials = 10
    train_size = 1000
    Delta = 7
    results_file = f'football/results/football_fit_results_{n_file}_{n_top_teams}_{n_bottom_teams}.json'
    
    # Create directory if it doesn't exist
    os.makedirs('football/results', exist_ok=True)
    
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
        
        train_data, test_data, train_indices, test_indices = train_split(football_data, train_size, trial + seed)
        
        # Fit consensus ranking
        sigma_0 = consensus_ranking_estimation(train_data)
        print(f"     Consensus ranking estimated", sigma_0)
        alpha, beta = solve_alpha_beta(train_data, sigma_0, Delta=Delta)
        print(f"     Alpha and beta estimated", alpha, beta)
        
        # Calculate Top-k hit rates
        top_hit_rates, distances, ndcg = soft_top_k(test_data,
                                                       alpha_hat=alpha, 
                                                       beta_hat=beta, 
                                                       sigma_hat=sigma_0,
                                                       Delta=Delta,
                                                       rng_seed=42)
        print(f"     [ML] Top-k hit rates calculated: {top_hit_rates}")
        pl_utilities, nll = learn_PL(train_data-1, test_data-1)
        top_hit_rates_PL, distances_PL, ndcg_PL, pairwise_acc_PL = soft_top_k_PL(test_data, pl_utilities)
        print(f"     [PL] Top-k hit rates calculated: {top_hit_rates_PL}")

        pi_0, theta_hat, _ = learn_kendal(train_data-1, test_data-1)
        print('       [kendal] pi_0', 1+pi_0)
        top_hit_rates_kendal, distances_kendal, ndcg_kendal, pairwise_acc_kendal = soft_top_k_kendal(test_data, theta_hat, pi_0)
        print(f"     [Kendal] top-k hit rates calculated: {top_hit_rates_kendal}")

        
        # Store results for this trial
        trial_results = {
            'alpha': alpha.tolist() if isinstance(alpha, np.ndarray) else alpha,
            'beta': beta.tolist() if isinstance(beta, np.ndarray) else beta,
            'sigma_0': sigma_0.tolist() if isinstance(sigma_0, np.ndarray) else sigma_0,
            'top_hit_rates': top_hit_rates,
            'distances': distances,
            'ndcg': ndcg,
            'top_hit_rates_PL': top_hit_rates_PL,
            'distances_PL': distances_PL,
            'ndcg_PL': ndcg_PL,
            'top_hit_rates_kendal': top_hit_rates_kendal,
            'distances_kendal': distances_kendal,
            'ndcg_kendal': ndcg_kendal,
            'test_indices': test_indices.tolist()
        }
        
        results.append(trial_results)
        
        # Save results after each trial
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"     Trial {trial+1} results saved")
    
    print(f"Completed {len(results)} trials. Results saved to '{results_file}'")


def train_split(sport_data, training_ratio, seed=None):
    """
    Split data chronologically, with training data from earlier time periods
    and test data randomly sampled from the later time period.
    
    Parameters:
    -----------
    sport_data : array-like
        The sports data ordered chronologically by date
    train_size : int
        Number of samples to include in the training set
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    train_data, test_data, train_indices, test_indices
    """
    # Initialize random number generator with seed
    rng = default_rng(seed)
    
    # Convert sport_data to numpy array if it's not already
    sport_data_array = np.asarray(sport_data)
    n_samples = len(sport_data_array)
    
    # Use 80% of data for potential training, 20% for potential testing
    train_test_split = int(n_samples * 0.7)
    
    # All indices before the split point are potential training data
    train_pool_indices = np.arange(train_test_split)
    
    # All indices after the split point are potential test data
    test_pool_indices = np.arange(train_test_split, n_samples)
    
    # Randomly sample train_size indices from the training pool
    train_size = 800#int(training_ratio * len(train_pool_indices))
    test_size = 250#int(training_ratio * len(test_pool_indices))
    
    train_indices = rng.choice(train_pool_indices, size=train_size, replace=False)
    test_indices = rng.choice(test_pool_indices, size=test_size, replace=False)
    
    # Split data into training and testing sets
    train_data = sport_data_array[train_indices]
    test_data = sport_data_array[test_indices]
    
    return train_data, test_data, train_indices, test_indices

    
    
   
    
    


