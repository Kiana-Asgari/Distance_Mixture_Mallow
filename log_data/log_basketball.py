import numpy as np
import json
import os
from datasets.learn_american_football import get_full_rankings, get_top_teams
from datasets.load_basketball import load_data
from learning_params_new.learn_alpha import learn_beta_and_sigma
from learning_params_new.likelihood_test import test_error
from learning_params_new.learn_kendal import learn_kendal
from learning_params_new.learn_PL import learn_PL
import matplotlib.pyplot as plt
import sys

"""
logging the error, best beta, best sigma for each alpha
for each n_file and desired_teams there is a different json file

"""

def top_k_accuracy_basketball(n_file, n_top_teams, n_bottom_teams, 
                                   Delta, seed=42):
    np.random.seed(seed)  # For reproducibility

    teams, votes_dict = load_data(limit=n_file)

    top_teams = get_top_teams(teams, votes_dict)
    desired_teams = 1+np.concatenate([top_teams[1:n_top_teams], top_teams[-n_bottom_teams:-5]])
    full_rankings = get_full_rankings(teams, votes_dict, which_team_to_keep = desired_teams)
    print(f'full_rankings: {full_rankings.shape}')
    
    # Split data into train and test (20% test)
    n_samples = full_rankings.shape[0]
    indices = np.random.permutation(n_samples)
    test_size = int(0.2 * n_samples)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    full_rankings_test = full_rankings[test_indices]
    full_rankings_train = full_rankings[train_indices]
    
    
    sigma_0_kendal, theta_kendal, kendal_error = learn_kendal(full_rankings_train, full_rankings_test)
    print(f"sigma_0_kendal: {sigma_0_kendal}")
    theta_PL, nll_test = learn_PL(permutations_train=full_rankings_train, permutations_test=full_rankings_test)
    print(f"theta_PL: {theta_PL}")

    """
    Reads the basketball results from the JSON file and creates a plot showing
    -error vs alpha with standard deviation bands.
    """
    # Create figures directory and basketball subdirectory if they don't exist
    figures_dir = 'log_data/figures'
    basketball_dir = os.path.join(figures_dir, 'basketball')
    os.makedirs(basketball_dir, exist_ok=True)
    
    # Read the results
    filename = f'log_data/basketball_results_n{n_file}_1:top{n_top_teams}_bottom{n_bottom_teams}:-5.json'
    with open(filename, 'r') as f:
        results = json.load(f)
    
    # Convert data to lists for plotting
    alphas = []
    errors = []    
    for alpha, data in results.items():
        alphas.append(float(alpha))
        errors.append(-1 * data['error'])  # Multiply by -1 here
    
    # Find the minimum error and corresponding values
    min_error_idx = np.argmin(errors)
    min_alpha = alphas[min_error_idx]
    
    # Get the corresponding data from the results
    min_alpha_key = f"{min_alpha:.6f}"
    min_data = results[min_alpha_key]
    alpha_mallow = min_alpha
    beta_mallow = min_data['beta']
    sigma_mallow = min_data['sigma']
    
    
    return alpha_mallow, beta_mallow, sigma_mallow, theta_kendal, sigma_0_kendal, theta_PL



###########################################################################################

def log_basketball_vs_alpha(n_file, n_top_teams, n_bottom_teams, 
                                   Delta, seed=42):
    # Create filename based on parameters
    filename = f'log_data/basketball_results_n{n_file}_1:top{n_top_teams}_bottom{n_bottom_teams}:-5.json'
    
    # Load existing data if file exists
    existing_data = {}
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            existing_data = json.load(f)
    
    np.random.seed(seed)  # For reproducibility

    teams, votes_dict = load_data(limit=n_file)

    top_teams = get_top_teams(teams, votes_dict)
    desired_teams = 1+np.concatenate([top_teams[1:n_top_teams], top_teams[-n_bottom_teams:-5]])
    full_rankings = get_full_rankings(teams, votes_dict, which_team_to_keep = desired_teams)
    print(f'full_rankings: {full_rankings.shape}')
    
    # Split data into train and test (20% test)
    n_samples = full_rankings.shape[0]
    indices = np.random.permutation(n_samples)
    test_size = int(0.2 * n_samples)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    full_rankings_test = full_rankings[test_indices]
    full_rankings_train = full_rankings[train_indices]
    
    print(f'Train shape: {full_rankings_train.shape}, Test shape: {full_rankings_test.shape}')

    alpha_list = np.linspace(2, 3, 100)

    for alpha in alpha_list:
        # Convert alpha to string for JSON key (with limited precision)
        alpha_key = f"{alpha:.6f}"
        

            
        print(f'Processing alpha={alpha_key}...')
        
        pi_0_hat, theta_hat, kendal_error = learn_kendal(full_rankings_train, full_rankings_test)
        print(f'Kendal: error: {kendal_error:3f}')
        print("Kendal: Estimated consensus ranking (pi_0):", pi_0_hat)
        print("Kendal: Estimated dispersion parameter (theta):", theta_hat)
    
        theta_PL, nll_test = learn_PL(permutations_train=full_rankings_train,
                 permutations_test=full_rankings_test)
        print(f'PL: theta: {theta_PL}')
        print(f'PL: nll_test: {nll_test}')
        # Skip if this alpha has already been processed
        if alpha_key in existing_data:
            print(f'Skipping alpha={alpha_key} (already processed)')
            continue
        beta_opt, sigma_opt = learn_beta_and_sigma(permutation_samples=full_rankings_train,
                                                 alpha=alpha,
                                                 beta_init=1,
                                                 Delta=Delta)
        
        # Test model
        error = test_error(full_rankings_test, beta_hat=beta_opt, sigma_hat=sigma_opt, alpha_hat=alpha)
        
        # Store results
        existing_data[alpha_key] = {
            'beta': float(beta_opt),
            'sigma': sigma_opt.tolist(),
            'error': float(error),
            'kendal_error': float(kendal_error),
            'PL_error': float(nll_test)
        }
        
        # Save after each iteration to prevent data loss
        with open(filename, 'w') as f:
            json.dump(existing_data, f, indent=4)
            
        print(f'*for alpha={alpha}:')
        print(f'  Error: {error:.3f}')
        print(f'  Beta: {beta_opt:.3f}')
   

def plot_basketball_results(n_file, n_top_teams, n_bottom_teams):
    """
    Reads the basketball results from the JSON file and creates a plot showing
    -error vs alpha with standard deviation bands.
    """
    # Create figures directory and basketball subdirectory if they don't exist
    figures_dir = 'log_data/figures'
    basketball_dir = os.path.join(figures_dir, 'basketball')
    os.makedirs(basketball_dir, exist_ok=True)
    
    # Read the results
    filename = f'log_data/basketball_results_n{n_file}_1:top{n_top_teams}_bottom{n_bottom_teams}:-5.json'
    with open(filename, 'r') as f:
        results = json.load(f)
    
    # Convert data to lists for plotting
    alphas = []
    errors = []
    kendal_errors = []
    
    for alpha, data in results.items():
        alphas.append(float(alpha))
        errors.append(-1 * data['error'])  # Multiply by -1 here
        kendal_errors.append(data['kendal_error'])  # Multiply by -1 here
    
    # Find the minimum error and corresponding values
    min_error_idx = np.argmin(errors)
    min_alpha = alphas[min_error_idx]
    min_error = errors[min_error_idx]
    
    # Get the corresponding data from the results
    min_alpha_key = f"{min_alpha:.6f}"
    min_data = results[min_alpha_key]
    
    # Print the statistics
    print("\nResults Summary:")
    print(f"Minimum Error: {-1 * min_error:.6f} at alpha = {min_alpha:.6f}")
    print(f"Corresponding beta: {min_data['beta']:.6f}")
    print(f"Kendall Error: {min_data['kendal_error']:.6f}")
    print(f"PL Error: {min_data['PL_error']:.6f}")
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, errors, 'b-', label='Negative Error vs Alpha')
    # Use the first Kendall error value since it should be constant
    #plt.axhline(y=kendal_errors[0], color='r', linestyle='--', label='Kendal Error')
    
    plt.xlabel('Alpha')
    plt.ylabel('Negative Error')
    plt.title(f'Basketball Results (n={n_file}, top={n_top_teams}, bottom={n_bottom_teams})')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plot_filename = os.path.join(basketball_dir, f'basketball_results_n{n_file}_top{n_top_teams}_bottom{n_bottom_teams}.png')
    plt.savefig(plot_filename)
    plt.close()


def plot_basketball_k_top(n_file, n_top_teams, n_bottom_teams, k=5, seed=42):
    # Create filename based on parameters
    filename = f'log_data/basketball_results_n{n_file}_1:top{n_top_teams}_bottom{n_bottom_teams}:-5.json'
    
    # Load existing data if file exists
    existing_data = {}
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            existing_data = json.load(f)
    
    np.random.seed(seed)  # For reproducibility

    teams, votes_dict = load_data(limit=n_file)

    top_teams = get_top_teams(teams, votes_dict)
    desired_teams = 1+np.concatenate([top_teams[1:n_top_teams], top_teams[-n_bottom_teams:-5]])
    full_rankings = get_full_rankings(teams, votes_dict, which_team_to_keep = desired_teams)
    print(f'full_rankings: {full_rankings.shape}')
    
    # Split data into train and test (20% test)
    n_samples = full_rankings.shape[0]
    indices = np.random.permutation(n_samples)
    test_size = int(0.2 * n_samples)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    full_rankings_test = full_rankings[test_indices]
    full_rankings_train = full_rankings[train_indices]
    
    # Read the results file
    filename = f'log_data/basketball_results_n{n_file}_1:top{n_top_teams}_bottom{n_bottom_teams}:-5.json'
    with open(filename, 'r') as f:
        results = json.load(f)
    
    # Convert data to lists and find minimum error
    alphas = []
    errors = []
    for alpha, data in results.items():
        alphas.append(float(alpha))
        errors.append(-1 * data['error'])  # Convert to negative error
    
    # Find the minimum error and corresponding alpha
    min_error_idx = np.argmin(errors)
    min_alpha = alphas[min_error_idx]
    min_error = errors[min_error_idx]
    
    # Get the corresponding parameters
    min_alpha_key = f"{min_alpha:.6f}"
    min_data = results[min_alpha_key]
    optimal_beta = min_data['beta']
    optimal_sigma = min_data['sigma']
    
    print("\nOptimal Parameters:")
    print(f"Alpha: {min_alpha:.6f}")
    print(f"Beta: {optimal_beta:.6f}")
    print(f"Minimum Negative Error: {min_error:.6f}")
    print(f"Sigma: {optimal_sigma}")
    alpha_mallow_top_k = []
    kendal_mallow_top_k = []