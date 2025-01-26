import numpy as np
import json
import os
from learning_params_new.learn_alpha import learn_beta_and_sigma
from learning_params_new.likelihood_test import test_error
from datasets.load_sushi_prefrence import load_sushi_data
from learning_params_new.learn_kendal import estimate_mallows_parameters, negative_log_likelihood

"""

logging the error, best beta, best sigma for each alpha
for each n_file and desired_teams there is a different json file


"""
def log_sushi_vs_alpha(Delta, seed=42):
    # Create filename based on parameters
    filename = f'log_data/sushi_results.json'
    
    # Load existing data if file exists
    existing_data = {}
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            existing_data = json.load(f)
    
    np.random.seed(seed)  # For reproducibility


    print('***********learn_sushi_preference***********')
    full_rankings = load_sushi_data()
    print(f'sushi_rankings data: {full_rankings.shape}')

    # Create 5 folds
    np.random.seed(42)  # For reproducibility
    n_samples = full_rankings.shape[0]
    indices = np.random.permutation(n_samples)
    fold_size = n_samples // 5
    
    alpha_list = np.linspace(1, 3, 200)

    for alpha in alpha_list:
        # Convert alpha to string for JSON key (with limited precision)
        alpha_key = f"{alpha:.6f}"
        
        # Skip if this alpha has already been processed
        if alpha_key in existing_data:
            print(f'Skipping alpha={alpha_key} (already processed)')
            continue
            
        print(f'Processing alpha={alpha_key}...')
        
        # Initialize arrays to store results from each fold
        errors = []
        betas = []
        sigmas = []
        
        # Perform 5-fold cross validation
        for fold in range(5):
            # Create test indices for this fold
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < 4 else n_samples
            test_indices = indices[start_idx:end_idx]
            train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
            
            # Split data into train and test
            full_rankings_test = full_rankings[test_indices]
            full_rankings_train = full_rankings[train_indices]
            
            print(f'Fold {fold+1}: train shape: {full_rankings_train.shape}, test shape: {full_rankings_test.shape}')
            
            # Train model
            pi_0_hat, theta_hat = estimate_mallows_parameters(full_rankings_train)
            kendal_error = 1/len(full_rankings_test) * negative_log_likelihood(rankings=full_rankings_test, theta=theta_hat, pi_0=pi_0_hat)



            print(f'Kendal: error: {kendal_error:3f}')
            print("Kendal: Estimated consensus ranking (pi_0):", pi_0_hat)
            print("Kendal: Estimated dispersion parameter (theta):", theta_hat)
            beta_opt, sigma_opt = learn_beta_and_sigma(permutation_samples=full_rankings_train,
                                                     alpha=alpha,
                                                     beta_init=1,
                                                     Delta=Delta)
            
            # Test model
            error = test_error(full_rankings_test, beta_hat=beta_opt, sigma_hat=sigma_opt, alpha_hat=alpha)
            
            errors.append(error)
            betas.append(beta_opt)
            sigmas.append(sigma_opt)
        
        # Store average results across folds
        existing_data[alpha_key] = {
            'beta': float(np.mean(betas)),
            'beta_std': float(np.std(betas)),
            'sigma': np.mean(sigmas, axis=0).tolist(),  # Average sigma across folds
            'error': float(np.mean(errors)),
            'error_std': float(np.std(errors)),
            # Store individual fold results
            'fold_errors': [float(e) for e in errors],
            'fold_betas': [float(b) for b in betas],
            'fold_sigmas': [s.tolist() for s in sigmas],  # Store each fold's sigma
            'fold_details': [{  # Store detailed info for each fold
                'fold': i+1,
                'error': float(errors[i]),
                'beta': float(betas[i]),
                'sigma': sigmas[i].tolist()
            } for i in range(5)]
        }
        
        # Save after each alpha to prevent data loss
        with open(filename, 'w') as f:
            json.dump(existing_data, f, indent=4)
            
        print(f'*for alpha={alpha}:')
        print(f'  Mean error: {np.mean(errors):.3f} ± {np.std(errors):.3f}')
        print(f'  Mean beta: {np.mean(betas):.3f} ± {np.std(betas):.3f}')
   
