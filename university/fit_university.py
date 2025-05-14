from MLE.consensus_ranking_estimation import consensus_ranking_estimation
from MLE.alpha_beta_estimation import solve_alpha_beta
import numpy as np
import json
import os
import sys
from numpy.random import default_rng

from MLE.top_k import soft_top_k, soft_top_k_PL, soft_top_k_kendal, evaluate_in_sample_metrics
from benchmark.fit_placket_luce import sample_PL, learn_PL
from benchmark.fit_Mallow_kendal import learn_kendal
from university.load_university import load_data, get_top_teams_borda, get_full_rankings
"""
logging the error, best beta, best sigma for each alpha
for each n_file and desired_teams there is a different json file

"""

def fit_uni(n_file, n_top_teams=11, n_bottom_teams=10, 
                                   Delta=7, seed=42):
    np.random.seed(seed)  # For reproducibility

    teams, votes_dict = load_data(limit=n_file)

    top_teams = get_top_teams_borda(teams, votes_dict)
    desired_teams = np.concatenate([top_teams[1:n_top_teams], top_teams[-n_bottom_teams:-5]])
    data = get_full_rankings(teams, votes_dict, which_team_to_keep = desired_teams)

    print(f'university_data: {data.shape}')
    

    num_trials = 10
    train_size = int(data.shape[0]*.8)
    Delta = 7
    results_file = f'university/results/fit_results_{n_file}_{n_top_teams}_{n_bottom_teams}.json'
    
    # Create directory if it doesn't exist
    os.makedirs('university/results', exist_ok=True)
    
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
        
        train_data, test_data, train_indices, test_indices = train_split(data, train_size, trial + seed)
        
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
        top_hit_rates_kendal, distances_kendal, ndcg_kendal,pairwise_acc_kendal = soft_top_k_kendal(test_data, theta_hat, pi_0)
        print(f"     [Kendal] top-k hit rates calculated: {top_hit_rates_kendal}")

        train_metrics = evaluate_in_sample_metrics(train_data, alpha, beta, sigma_0)
        print("train metrics")
        print(train_metrics)


        
        # Store results for this trial
        trial_results = {
            'alpha': alpha.tolist() if isinstance(alpha, np.ndarray) else alpha,
            'beta': beta.tolist() if isinstance(beta, np.ndarray) else beta,
            'sigma_0': sigma_0.tolist() if isinstance(sigma_0, np.ndarray) else sigma_0,
            'top_hit_rates': top_hit_rates,
            'distances': distances,
            'ndcg': ndcg,
            'pairwise':train_metrics["acc"],
            'top_hit_rates_PL': top_hit_rates_PL,
            'distances_PL': distances_PL,            
            'pairwise_PL': pairwise_acc_PL,
            'ndcg_PL': ndcg_PL,
            'top_hit_rates_kendal': top_hit_rates_kendal,
            'pairwise_kendal': top_hit_rates_kendal,
            'distances_kendal': pairwise_acc_kendal,
            'ndcg_kendal': ndcg_kendal,
            'test_indices': test_indices.tolist()
        }
        
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

    
    
   
    
    


