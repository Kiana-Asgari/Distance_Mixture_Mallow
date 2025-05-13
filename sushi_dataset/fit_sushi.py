from sushi_dataset.load_data import load_sushi
from MLE.consensus_ranking_estimation import consensus_ranking_estimation
from MLE.alpha_beta_estimation import solve_alpha_beta
import numpy as np
from numpy.random import default_rng
import json
import os
import sys
from MLE.top_k import soft_top_k, soft_top_k_PL, soft_top_k_kendal
from benchmark.fit_placket_luce import sample_PL, learn_PL
from benchmark.fit_Mallow_kendal import learn_kendal

def fit_and_save_sushi(seed=42):

    sushi_data = load_sushi()    
    num_trials = 10
    train_size = 3500
    Delta = 7
    results_file = 'sushi_dataset/results/sushi_fit_results.json'
    
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
        top_hit_rates_PL, distances_PL, ndcg_PL = soft_top_k_PL(test_data, pl_utilities)
        print(f"     [PL] Top-k hit rates calculated: {top_hit_rates_PL}")

        pi_0, theta_hat, _ = learn_kendal(train_data-1, test_data-1)
        print('       [kendal] pi_0', 1+pi_0)
        top_hit_rates_kendal, distances_kendal, ndcg_kendal = soft_top_k_kendal(test_data, theta_hat, pi_0)
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


def plot_heat_map(alpha_hat, beta_hat, sigma_hat, test_data, Delta=7, n_samples=2000, save_path='sushi_dataset/results/position_heatmap.png'):
    """
    Creates and saves a heat map visualization comparing the position probabilities
    from the Mallows model sampling and the empirical distribution in the test data.
    
    Parameters
    ----------
    alpha_hat, beta_hat, sigma_hat : parameters of the trained model
    test_data : array-like
        Test data containing rankings
    Delta : int
        Delta parameter for Mallows sampling
    n_samples : int
        Number of samples to generate from the Mallows model
    save_path : str
        Path where to save the heat map image
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from GMM_diagonalized.sampling import sample_truncated_mallow
    
    m, n = test_data.shape
    
    # Generate samples from the Mallows model
    samples = sample_truncated_mallow(num_samples=n_samples,
                                     n=n,
                                     alpha=alpha_hat,
                                     beta=beta_hat,
                                     sigma=sigma_hat,
                                     Delta=Delta)
    samples = np.array(samples)
    
    # Calculate position probabilities from model samples
    model_position_counts = np.zeros((n, n))
    for pos in range(n):
        items_at_pos = samples[:, pos] - 1  # Convert 1-based to 0-based
        for item in range(n):
            model_position_counts[pos, item] = np.sum(items_at_pos == item)
    
    model_position_probs = model_position_counts / n_samples
    
    # Calculate empirical position probabilities from test data
    test_position_counts = np.zeros((n, n))
    for ranking in test_data:
        for pos, item in enumerate(ranking):
            test_position_counts[pos, item-1] += 1  # Convert 1-based to 0-based
    
    test_position_probs = test_position_counts / len(test_data)
    
    # Create the heat map plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot model probabilities
    sns.heatmap(model_position_probs, ax=ax1, cmap="YlGnBu", vmin=0, vmax=max(model_position_probs.max(), test_position_probs.max()))
    ax1.set_title('Model Position Probabilities P(item i at position j)')
    ax1.set_xlabel('Item (i)')
    ax1.set_ylabel('Position (j)')
    
    # Plot test data probabilities
    sns.heatmap(test_position_probs, ax=ax2, cmap="YlGnBu", vmin=0, vmax=max(model_position_probs.max(), test_position_probs.max()))
    ax2.set_title('Empirical Position Probabilities P(item i at position j)')
    ax2.set_xlabel('Item (i)')
    ax2.set_ylabel('Position (j)')
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heat map saved to {save_path}")
    
    return model_position_probs, test_position_probs