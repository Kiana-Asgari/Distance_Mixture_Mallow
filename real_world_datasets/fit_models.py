from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import sys

# ── data & utils ────────────────────────────────────────────────────────────────
from real_world_datasets.college_sports.load_data import load_data
from real_world_datasets.sushi_dataset.load_data import load_sushi
from real_world_datasets.utils import train_split, convert_numpy_to_native
from real_world_datasets.utils import chronologically_train_split, convert_numpy_to_native
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
from real_world_datasets.config import METRIC_NAMES


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
        print('first ranking:', data[0])
    return data

def _read_results(dataset_name, n_teams, save):    
    res_path = Path(f"real_world_datasets/results/{dataset_name}_n_teams={n_teams}.json")
    res_path.parent.mkdir(parents=True, exist_ok=True) # create the directory if it doesn't exist
    if save==True:
        results = json.loads(res_path.read_text()) if res_path.exists() else []
    else:
        results = []
    return results, res_path

def _save_results(results, res_path, save, say):
    if save:
        res_path.write_text(json.dumps(convert_numpy_to_native(results), indent=2))
        say("  ⭑ intermediate results saved")
        say(f"Final results → {res_path}")

def fit_models( dataset_name: str = "basketball",
                Delta: int = 7,
                seed: int = 42,
                n_trials: int = 50,
                n_teams: int = 10,
                mc_samples: int = 10_000,
                save: bool = False,
                verbose: bool = True
                ):
    mc_samples = 500
    # print if verbose else do nothing
    say = print if verbose else (lambda *_, **__: None) 
    # get the data 
    data = _get_data(dataset_name, n_teams)
    # get the results if they exist; else create an empty list
    results, res_path = _read_results(dataset_name, n_teams, save) 
    # fit the models using MCCV random splits
    results = _mote_carlo_CV(data, results, n_trials, n_teams, mc_samples, Delta, seed, say, save, dataset_name, res_path)
    # save the results
    _save_results(results, res_path, save, say)

    say(f"\nCompleted {len(results)} trials")
    # print the table of comparison
    print_online_results(results)






def _mote_carlo_CV(data, results, n_trials, n_teams, mc_samples, Delta, seed, say, save, dataset_name, res_path):
    """Main loop of the MCCV random splits"""

    for t in range(len(results), n_trials):      
        say(f"Trial {t + 1}/{n_trials}")

        # if news or movies, perform a random split; else perform a chronologically split
        if dataset_name in ['movie_lens', 'news', 'sushi']:
            train, test, *_ = train_split(data, 0.8, seed + t)
        else:
             train, test, *_ = chronologically_train_split(data, seed + t)
        train, test = np.array(train), np.array(test)


        # create a trial dictionary
        benchmark_models_names = ['our', 'L1', 'L2', 'tau', 'pl', 'pl_reg']
        trial = {model_name: [] for model_name in benchmark_models_names}
        # create a zip of benchmark models
        for model_name in benchmark_models_names:
            res_evals, args = _fit_benchmark_models(model_name, train, test, n_teams, mc_samples, Delta, say)
            trial[model_name].append({'evals': res_evals, 'args': args})
        
        results=trial.copy() # TODO: not reading from results

    return results



def _fit_benchmark_models(model_name, train, test, n_teams, mc_samples, Delta, say):

    if model_name == 'our':
        samples, args = fit_mallows(train, test, n_teams, mc_samples, Delta, say)
    elif model_name == 'L1':
        samples, args = fit_mallows(train, test, n_teams, mc_samples, Delta, say, alpha_fixed=True)
    elif model_name == 'L2':
        samples, args = fit_mallows(train, test, n_teams, mc_samples, Delta, say, alpha_fixed=True, alpha_fixed_value=2)
    elif model_name == 'pl':
        samples, args = fit_pl(train, test, n_teams, mc_samples, Delta, say)
    elif model_name == 'pl_reg':
        samples, args = fit_pl_reg(train, test, n_teams, mc_samples, Delta, say)
    elif model_name == 'tau':
        samples, args = fit_kendall(train, test, n_teams, mc_samples, Delta, say)
    else:
        raise ValueError(f"Model {model_name} not found")
    evals = evaluate_metrics(test, samples)

    return evals, args



def fit_mallows(train, test, k, mc, delta, say, alpha_fixed=False, alpha_fixed_value=1):

    say(f"  Starting to learn Mallows model for {len(train[0])} items ...")
    sigma_0 = consensus_ranking_estimation(train)
    alpha, beta = solve_alpha_beta(train, sigma_0, Delta=delta, fixed_alpha=alpha_fixed, fixed_alpha_value=alpha_fixed_value)

    say(f"        Mallows is learned with alpha: {alpha:.4f}, beta: {beta:.4f}")
    say(f"        testing the Mallows model with {mc} samples...")
    # testing the model ...
    if len(train[0]) > 20:
        Delta_mc = 5
    else:
        Delta_mc = 7
    samples = sample_truncated_mallow(n=k, alpha=alpha, beta=beta, sigma=sigma_0, Delta=Delta_mc, num_samples=mc)
    args = {
        "sigma_0": sigma_0,
        "alpha": alpha,
        "beta": beta,
    }
    return samples, args


def fit_pl(train, test, n_teams, mc_samples, Delta, say, alpha_fixed=None, alpha_fixed_value=None):
    say(f"  Starting to learn Plackett-Luce model for {len(train[0])} items ...")
    
    
    util, _ = learn_PL(train - 1, test- 1)
    say(f"        Plackett-Luce is learned.")
    say(f"        Testing the PL model with {mc_samples} samples...")
    # testing the model ...
    samples = sample_PL(util, n_samples=mc_samples)
    args={"util": util}
    return samples, args


def fit_pl_reg(train, test, n_teams, mc_samples, Delta, say, alpha_fixed=None, alpha_fixed_value=None):
    say(f"  Starting to learn Regularized Plackett-Luce model for {len(train[0])} items ...")
    lambda_reg = 0.01
    util, _ = learn_PL(train - 1, test- 1, lambda_reg=lambda_reg)
    say(f"        Regularized Plackett-Luce is learned.")
    say(f"        Testing the regularized PL model with {mc_samples} samples...")
    # testing the model ...
    args={"util": util, "lambda_reg": lambda_reg}
    samples = sample_PL(util, n_samples=mc_samples)
    return samples, args


def fit_kendall(train, test, n_teams, mc_samples, Delta, say, alpha_fixed=None, alpha_fixed_value=None):
    say(f"  Starting to learn Kendall model for {len(train[0])} items ...")
    sigma_0, theta, _ = learn_kendal(train - 1, test - 1)
    say(f"        Kendall is learned with theta: {theta:.4f}.")
    say(f"        testing the Kendall model with {mc_samples} samples...")
    # testing the model ...
    samples = sample_kendal(sigma_0=sigma_0, theta=theta, num_samples=mc_samples)
    args={"sigma_0": sigma_0, "theta": theta}
    return samples, args

def pack(*vals, suffix=""):            # names metrics[0] → f"{prefix}metric_name"
    return {f"{n}{suffix}": v for n, v in zip(METRIC_NAMES, vals)}
