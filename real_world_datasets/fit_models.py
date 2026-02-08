from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import sys
import pandas as pd

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

# ── data & utils ────────────────────────────────────────────────────────────────
from real_world_datasets.college_sports.load_data import load_data
from real_world_datasets.sushi_dataset.load_data import load_sushi
from real_world_datasets.utils import train_split, chronologically_train_split, convert_numpy_to_native
from real_world_datasets.print_evaluations import print_online_results
    
# ── Mallows (L-α) ──────────────────────────────────────────────────────────────
from MLE.consensus_ranking_estimation import consensus_ranking_estimation
from MLE.alpha_beta_estimation import solve_alpha_beta
from GMM_diagonalized.sampling import sample_truncated_mallow

# ── Plackett–Luce & Kendall ────────────────────────────────────────────────────
from benchmark.fit_placket_luce import learn_PL, sample_PL
from benchmark.fit_Mallow_kendal import learn_kendal, sample_kendal

from real_world_datasets.movie_lens.load_MovieLens import load_and_return_ratings_movies
from real_world_datasets.news.load_news import load_news_data

# ── common evaluation ─────────────────────────────────────────────────────────
from MLE.top_k import evaluate_metrics


#####################################################
# fitting models (Mallows, Plackett-Luce, Kendall)
# on real-world datasets (college sports, sushi)
#####################################################
def _get_data(dataset_name, n_teams):
        
    if dataset_name == 'sushi':
        data = load_sushi()
    elif dataset_name == 'movie_lens':
        data = load_and_return_ratings_movies(n_movies=n_teams)
    elif dataset_name == 'news':
        data = load_news_data(n_items_to_keep=n_teams)
    else:
        data = load_data(dataset_name=dataset_name, n_teams_to_keep=n_teams)
    return data

def _read_results(dataset_name, n_teams, save):    
    res_path = Path(f"real_world_datasets/results/{dataset_name}_n_teams={n_teams}.json")
    res_path.parent.mkdir(parents=True, exist_ok=True)
    if save and res_path.exists():
        results = json.loads(res_path.read_text())
    else:
        results = {"trials": [], "metadata": {}}
    return results, res_path

def _save_results(results, res_path, save, say):
    if save:
        res_path.write_text(json.dumps(convert_numpy_to_native(results), indent=2))
        say(f"  ⭑ Results saved → {res_path}")

def _json_to_dataframe(trials):
    """Convert JSON trial results to DataFrame for printing (only scalar metrics + alpha/beta)."""
    rows = []
    for trial in trials:
        for model_name, data in trial["models"].items():
            row = {"Model": model_name}
            # Add all metrics (including lists like top_k_hit_rates)
            row.update(data["metrics"])
            # Add only alpha/beta parameters (not util, sigma_0, etc.)
            row["alpha"] = data["args"].get("alpha", 0)
            row["beta"] = data["args"].get("beta", 0)
            rows.append(row)
    return pd.DataFrame(rows)


def print_saved_results(dataset_name: str = "basketball", Delta: int = 7,
                          n_teams: int = 10, n_trials: int = 50):
    res_path = Path(f"real_world_datasets/results/{dataset_name}_n_teams={n_teams}.json")  

    
    # Read the JSON file (same as _read_results does when loading)
    results = json.loads(res_path.read_text())
      
    # Convert JSON trials to DataFrame (same as fit_models does at line 119)
    df_results = _json_to_dataframe(results["trials"])
    
    # Print the results using the same format as fit_models (line 120)
    print_online_results(df_results, dataset_name=f"{dataset_name}n={n_teams}k={Delta}trial={n_trials}")
    



def fit_models(dataset_name: str = "basketball", Delta: int = 7, seed: int = 42,
               n_trials: int = 50, n_teams: int = 10, mc_samples: int = 10_000,
               save: bool = True, verbose: bool = True, n_jobs: int = 2):
    """
    Fit and evaluate multiple ranking models.
    
    Args:
        n_jobs: Number of parallel jobs. Use -1 for all CPUs, 1 for sequential (default).
                Requires joblib to be installed for parallel execution.
    """
    say = print if verbose else (lambda *_, **__: None)
    data = _get_data(dataset_name, n_teams)
    results, res_path = _read_results(dataset_name, n_teams, save)
    
    # Store metadata
    results["metadata"] = {
        "dataset": dataset_name, "n_teams": n_teams, "Delta": Delta,
        "seed": seed, "n_trials": n_trials, "mc_samples": mc_samples
    }
    
    # Run trials (parallel or sequential)
    results = _mote_carlo_CV(data, results, n_trials, n_teams, mc_samples, Delta, seed, say, save, dataset_name, res_path, n_jobs)
    _save_results(results, res_path, save, say)
    
    # Convert to DataFrame for printing
    df_results = _json_to_dataframe(results["trials"])
    print_online_results(df_results, dataset_name=f"{dataset_name}n={n_teams}k={Delta}trial={n_trials}")






def _process_single_trial(t, data, random_seeds, dataset_name, n_teams, mc_samples, Delta):
    """Process a single trial - designed for parallel execution."""
    # Data split
    if dataset_name in ['movie_lens', 'news', 'sushi']:
        train, test, *_ = train_split(data, 0.7, random_seeds[t])
    else:
        train, test, *_ = chronologically_train_split(data, random_seeds[t])
    train, test = np.array(train), np.array(test)
    
    # Fit all models for this trial (no logging in parallel mode)
    trial_result = {"trial_id": t + 1, "models": {}}
    for model_name in ['our', 'L1', 'L2', 'tau', 'pl', 'pl_reg']:
        metrics, args = _fit_benchmark_models(model_name, train, test, n_teams, mc_samples, Delta, lambda *_, **__: None)
        trial_result["models"][model_name] = {"metrics": metrics, "args": args}
    
    return trial_result


def _mote_carlo_CV(data, results, n_trials, n_teams, mc_samples, Delta, seed, say, save, dataset_name, res_path, n_jobs=1):
    """Main loop of the MCCV random splits (supports parallel execution)."""
    rng = np.random.default_rng(seed)
    random_seeds = rng.integers(0, 1000000, 150)
    
    start_trial = len(results["trials"])
    remaining_trials = list(range(start_trial, n_trials))
    
    if not remaining_trials:
        return results
    
    # Parallel execution if joblib available and n_jobs != 1
    if HAS_JOBLIB and n_jobs != 1 and len(remaining_trials) > 1:
        say(f"Running {len(remaining_trials)} trials in parallel with {n_jobs} jobs...")
        
        trial_results = Parallel(n_jobs=n_jobs, verbose=5 if save else 0)(
            delayed(_process_single_trial)(t, data, random_seeds, dataset_name, n_teams, mc_samples, Delta)
            for t in remaining_trials
        )
        
        # Add all results
        results["trials"].extend(trial_results)
        
        # Save once after all parallel work
        if save:
            _save_results(results, res_path, save, say)
        
        say(f"Completed {len(trial_results)} trials")
    
    else:
        # Sequential execution (original behavior)
        for t in remaining_trials:
            say(f"Trial {t + 1}/{n_trials}")
            trial_result = _process_single_trial(t, data, random_seeds, dataset_name, n_teams, mc_samples, Delta)
            results["trials"].append(trial_result)
            
            # Save after each trial in sequential mode
            if save:
                _save_results(results, res_path, save, say)
    
    return results



def _fit_benchmark_models(model_name, train, test, n_teams, mc_samples, Delta, say):
    model_map = {
        'our': lambda: fit_mallows(train, test, n_teams, mc_samples, Delta, say),
        'L1': lambda: fit_mallows(train, test, n_teams, mc_samples, Delta, say, alpha_fixed=True),
        'L2': lambda: fit_mallows(train, test, n_teams, mc_samples, Delta, say, alpha_fixed=True, alpha_fixed_value=2),
        'pl': lambda: fit_pl(train, test, n_teams, mc_samples, Delta, say),
        'pl_reg': lambda: fit_pl_reg(train, test, n_teams, mc_samples, Delta, say),
        'BT': lambda: fit_pl(train, test, n_teams, mc_samples, Delta, say, BL_model=True),
        'tau': lambda: fit_kendall(train, test, n_teams, mc_samples, Delta, say)
    }

    
    samples, args = model_map[model_name]()
    metrics = evaluate_metrics(test, samples)
    return metrics, args



def fit_mallows(train, test, k, mc, delta, say, alpha_fixed=False, alpha_fixed_value=1):
    say(f"===== Mallows {len(train[0])} items ...")
    sigma_0 = consensus_ranking_estimation(train, alpha_fixed=alpha_fixed, alpha_fixed_value=alpha_fixed_value)
    alpha, beta = solve_alpha_beta(train, sigma_0, Delta=delta, fixed_alpha=alpha_fixed, fixed_alpha_value=alpha_fixed_value)
    #beta = beta / (len(train[0])**alpha)  # Standardized beta
    Delta_mc = 4 if len(train[0]) > 20 else 7
    print(f'Delta_mc={Delta_mc} for {len(train[0])} items')
    samples = sample_truncated_mallow(n=k, alpha=alpha, beta=beta, sigma=sigma_0, Delta=Delta_mc, num_samples=mc)
    say(f"  Learned sigma_0: {sigma_0}")
    return samples, {"sigma_0": sigma_0.tolist(), "alpha": float(alpha), "beta": float(beta)}


def fit_pl(train, test, n_teams, mc_samples, Delta, say, BL_model=False):
    say(f"===== Plackett-Luce {len(train[0])} items ...")
    util, _, _ = learn_PL(train - 1, test - 1, BL_model=BL_model)
    samples = sample_PL(util, n_samples=mc_samples)
    return samples, {"util": util.tolist()}

def fit_pl_reg(train, test, n_teams, mc_samples, Delta, say, BL_model=False):
    say(f"===== Plackett-Luce (Regularized) {len(train[0])} items ...")
    util, _, lambda_reg = learn_PL(train - 1, test - 1, use_cv=True, BL_model=BL_model)
    samples = sample_PL(util, n_samples=mc_samples)
    return samples, {"util": util.tolist(), "lambda_reg": float(lambda_reg) if lambda_reg else 0.0}

def fit_kendall(train, test, n_teams, mc_samples, Delta, say):
    say(f"===== Kendall {len(train[0])} items ...")
    sigma_0, theta, _ = learn_kendal(train - 1, test - 1)
    samples = sample_kendal(sigma_0=sigma_0, theta=theta, num_samples=mc_samples)
    say(f"  Learned sigma_0: {sigma_0}")
    return samples, {"sigma_0": sigma_0.tolist(), "theta": float(theta)}
