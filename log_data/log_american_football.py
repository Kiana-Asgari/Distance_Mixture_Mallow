import numpy as np
import json
import os
from datasets.learn_american_football import get_full_rankings, get_top_teams
from datasets.load_american_football import load_data
from learning_params_new.learn_alpha import learn_beta_and_sigma
from learning_params_new.likelihood_test import test_error

"""

logging the error, best beta, best sigma for each alpha
for each n_file and desired_teams there is a different json file


"""
def log_american_football_vs_alpha(n_file, n_top_teams, n_bottom_teams, 
                                   Delta, seed=42):
    # Create filename based on parameters
    filename = f'log_data/football_results_n{n_file}_1:top{n_top_teams}_bottom{n_bottom_teams}:-5.json'
    
    # Load existing data if file exists
    existing_data = {}
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            existing_data = json.load(f)
    
    np.random.seed(seed)  # For reproducibility

    teams, votes_dict = load_data(limit=n_file)

    top_teams = get_top_teams(teams, votes_dict)
    desired_teams = 1+np.concatenate([top_teams[1:n_top_teams], top_teams[-n_bottom_teams:-5]])
    full_rankings = get_full_rankings(teams, votes_dict, which_team_to_keep = desired_teams) #keep 20 first teams
    print(f'full_rankings: {full_rankings.shape}')
    
    # Randomly select 50 rankings for the test set
    test_indices = np.random.choice(full_rankings.shape[0], 50, replace=False)
    full_rankings_test = full_rankings[test_indices]
    
    # Use the remaining rankings for training
    train_indices = np.setdiff1d(np.arange(full_rankings.shape[0]), test_indices)
    full_rankings_train = full_rankings[train_indices]

    print(f'rankings train: {full_rankings_train.shape}')
    print(f'rankings test: {full_rankings_test.shape}')

    alpha_list = np.linspace(1, 2, 100)

    for alpha in alpha_list:
        # Convert alpha to string for JSON key (with limited precision)
        alpha_key = f"{alpha:.6f}"
        
        # Skip if this alpha has already been processed
        if alpha_key in existing_data:
            print(f'Skipping alpha={alpha_key} (already processed)')
            continue
            
        print(f'Processing alpha={alpha_key}...')
        
        beta_opt, sigma_opt = learn_beta_and_sigma(permutation_samples=full_rankings_train,
                                                        alpha=alpha,
                                                        beta_init=1,
                                                        Delta=Delta)
        error = test_error(full_rankings_test, beta_hat=beta_opt, sigma_hat=sigma_opt, alpha_hat=alpha)
        
        # Store results - convert sigma numpy array to list
        existing_data[alpha_key] = {
            'beta': float(beta_opt),
            'sigma': sigma_opt.tolist(),  # Convert numpy array to list for JSON serialization
            'error': float(error)
        }
        
        # Save after each iteration to prevent data loss
        with open(filename, 'w') as f:
            json.dump(existing_data, f, indent=4)
            
        print(f'*for alpha={alpha}, beta_opt: {beta_opt:3f}, error: {error:3f}, sigma_opt: {sigma_opt}')
   
