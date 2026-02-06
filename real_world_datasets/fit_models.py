import numpy as np
import pandas as pd
import os
import json
# Data loading
from real_world_datasets.loaders.data_loader import load_data
from real_world_datasets.utils import train_split, chronologically_train_split, print_online_results

# Model fitting and sampling
from MLE.consensus_ranking_estimation import consensus_ranking_estimation
from MLE.alpha_beta_estimation import solve_alpha_beta
from GMM_diagonalized.sampling import sample_truncated_mallow
from benchmark.fit_placket_luce import learn_PL, sample_PL
from benchmark.fit_Mallow_kendal import learn_kendal, sample_kendal

# Evaluation
from MLE.test_metrics import evaluate_metrics


DELTA_MC = 4  # Monte Carlo sampling parameter



def fit_models(dataset_name: str = "basketball",
                Delta: int = 7,
                seed: int = 42,
                n_trials: int = 50,
                n_teams: int = 10,
                mc_samples: int = 10_000,
                verbose: bool = True):
    """
    Fit and compare ranking models on real-world datasets using Monte Carlo cross-validation.
    """
    say = print if verbose else lambda *_, **__: None
    data = load_data(dataset_name, n_teams)
    results = _monte_carlo_cv(data, dataset_name, n_trials, n_teams, mc_samples, Delta, seed, say)
    experiment_name = f"{dataset_name}n={n_teams}k={Delta}trial={n_trials}"

    saving_path = os.path.join(os.path.dirname(__file__), 'results_json', experiment_name)
    os.makedirs(saving_path, exist_ok=True)
    with open(os.path.join(saving_path, 'results.json'), 'w') as f:
        json.dump(results.to_dict(), f)
    print_online_results(results, dataset_name=experiment_name)






def _monte_carlo_cv(data, dataset_name, n_trials, n_teams, mc_samples, Delta, seed, say):
    """Run Monte Carlo cross-validation with multiple train/test splits."""
    rng = np.random.default_rng(seed)
    random_seeds = rng.integers(0, 1_000_000, 150)    
    # Models to benchmark
    models = ['our', 'L1', 'L2', 'tau', 'pl']#, 'BT', 'pl_reg']
    results_list = []

    for trial in range(n_trials):
        say(f"[Trial {trial + 1}/{n_trials}]")
        
        if dataset_name in ['movie_lens', 'news', 'sushi']:
            train, test, *_ = train_split(data, 0.7, random_seeds[trial])
        else:
            train, test, *_ = chronologically_train_split(data, random_seeds[trial])
        
        train, test = np.array(train), np.array(test)

        # Fit each model and evaluate
        for model_name in models:
            metrics, params = _fit_and_evaluate(model_name, train, test, n_teams, mc_samples, Delta, say)
            # Store results
            row = {'Model': model_name, **metrics}
            row['alpha'] = params.get('alpha', 0)
            row['beta'] = params.get('beta', 0)
            results_list.append(row)
    
    return pd.DataFrame(results_list)



def _fit_and_evaluate(model_name, train, test, n_teams, mc_samples, Delta, say):
    """Fit a model and evaluate it on test data."""
    # Fit model based on name
    if model_name in ['our', 'L1', 'L2']:
        # Mallows variants: 'our' learns alpha, 'L1' fixes alpha=1, 'L2' fixes alpha=2
        alpha_map = {'our': None, 'L1': 1, 'L2': 2}
        samples, params = _fit_mallows(train, n_teams, mc_samples, Delta, say, alpha_value=alpha_map[model_name])
    
    elif model_name in ['pl', 'BT']:
        # Plackett-Luce variants: 'pl' is standard, 'BT' is Bradley-Terry
        samples, params = _fit_plackett_luce(train, mc_samples, say, BL_model=(model_name == 'BT'))
    
    elif model_name == 'pl_reg':
        samples, params = _fit_plackett_luce_reg(train, n_teams, mc_samples, say)
    
    elif model_name == 'tau':
        samples, params = _fit_kendall(train, mc_samples, say)
   
    # Evaluate on test data
    metrics = evaluate_metrics(test, samples)
    return metrics, params



def _fit_mallows(train, k, mc, delta, say, alpha_value=None):
    say(f"====> Learning Mallows model ({len(train[0])} items)...")   
    # Determine if alpha is fixed
    alpha_fixed = (alpha_value is not None)
    alpha_val = alpha_value if alpha_fixed else 1
    
    # Estimate consensus ranking
    sigma_0 = consensus_ranking_estimation(train, alpha_fixed=alpha_fixed, alpha_fixed_value=alpha_val)
    
    # Estimate alpha and beta parameters
    alpha, beta = solve_alpha_beta(train, sigma_0, Delta=delta, fixed_alpha=alpha_fixed, fixed_alpha_value=alpha_val)
   
    # Generate samples
    samples = sample_truncated_mallow(n=k, alpha=alpha, beta=beta, sigma=sigma_0, Delta=DELTA_MC, num_samples=mc)
    
    return samples, {"sigma_0": sigma_0, "alpha": alpha, "beta": beta}


def _fit_plackett_luce(train, mc_samples, say, BL_model=False):
    model_name = "Bradley-Terry" if BL_model else "Plackett-Luce"
    say(f"====> Learning {model_name} model ({len(train[0])} items)...")
    
    # Learn utilities (convert to 0-indexing)
    util, _ = learn_PL(train - 1, train - 1, BL_model=BL_model)
    
    # Generate samples (returns 0-indexed rankings)
    samples = sample_PL(util, n_samples=mc_samples)
    
    # Convert back to 1-indexed to match test data format
    samples = samples + 1
    
    return samples, {"util": util}


def _fit_plackett_luce_reg(train, n_teams, mc_samples, say):
    say(f"====> Learning Regularized Plackett-Luce model ({len(train[0])} items)...")
    # Set regularization parameter based on team count
    if n_teams < 20:
        lambda_reg = 0.001
    elif n_teams > 60:
        lambda_reg = 0.1
    else:
        lambda_reg = 0.01
    
    # Learn utilities with regularization (convert to 0-indexing)
    util, _ = learn_PL(train - 1, train - 1, lambda_reg=lambda_reg, BL_model=False)
    
    # Generate samples (returns 0-indexed rankings)
    samples = sample_PL(util, n_samples=mc_samples)
    
    # Convert back to 1-indexed to match test data format
    samples = samples + 1
    
    return samples, {"util": util, "lambda_reg": lambda_reg}


def _fit_kendall(train, mc_samples, say):
    say(f"====> Learning Kendall model ({len(train[0])} items)...")
    
    # Learn consensus and dispersion parameter (convert to 0-indexed)
    sigma_0, theta, _ = learn_kendal(train - 1, train - 1)
    say(f"    theta={theta:.4f}")
    
    # Generate samples (returns 0-indexed rankings)
    samples = sample_kendal(sigma_0=sigma_0, theta=theta, num_samples=mc_samples)
    
    # Convert back to 1-indexed to match test data format
    samples = samples + 1
    
    return samples, {"sigma_0": sigma_0, "theta": theta}
