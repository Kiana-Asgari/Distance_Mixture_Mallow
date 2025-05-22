from __future__ import annotations
from pathlib import Path
import json
import numpy as np

# ── data & utils ────────────────────────────────────────────────────────────────
from real_world_datasets.college_sports.load_data import load_data
from real_world_datasets.utils import chronologically_train_split
from real_world_datasets.print_evaluations import print_online_results

# ── Mallows (L-α) ──────────────────────────────────────────────────────────────
from MLE.consensus_ranking_estimation import consensus_ranking_estimation
from MLE.alpha_beta_estimation import solve_alpha_beta
from GMM_diagonalized.sampling import sample_truncated_mallow

# ── Plackett–Luce & Kendall ────────────────────────────────────────────────────
from benchmark.fit_placket_luce import learn_PL, sample_PL
from benchmark.fit_Mallow_kendal import learn_kendal, sample_kendal

# ── common evaluation ─────────────────────────────────────────────────────────
from MLE.top_k import evaluate_metrics
from real_world_datasets.config import METRIC_NAMES


#####################################################
# fitting models (Mallows, Plackett-Luce, Kendall)
# on real-world datasets (college sports, sushi)
#####################################################

def fit_models(
    dataset_name: str = "basketball",
    Delta: int = 7,
    seed: int = 42,
    n_trials: int = 50,
    n_teams: int = 10,
    mc_samples: int = 10_000,
    save: bool = False,
    verbose: bool = True,
    ):

    say = print if verbose else (lambda *_, **__: None) # print if verbose else do nothing

    data = load_data(dataset_name=dataset_name, n_teams_to_keep=n_teams)
    res_path = Path(f"real_world_datasets/results/{dataset_name}_n_teams={n_teams}.json")
    res_path.parent.mkdir(parents=True, exist_ok=True) # create the directory if it doesn't exist
    if save==True:
        results = json.loads(res_path.read_text()) if res_path.exists() else []
    else:
        results = []

    for t in range(len(results), n_trials):
        say(f"Trial {t + 1}/{n_trials}")
        train, test, *_ = chronologically_train_split(data, seed + t)
        say(f"  train and test data shape: {train.shape}, {test.shape}")
        trial = {
            "full_data_size": len(data),
            "train_size": len(train),
            "test_size": len(test),
        }

        # Mallows (L-α)
        mallows = fit_mallows(train, test, n_teams, mc_samples, Delta, say)
        trial.update(mallows)
        # Plackett–Luce
        PL = fit_pl(train, test, mc_samples, say)
        trial.update(PL)

        # Kendall
        kendall = fit_kendall(train, test, mc_samples, say)
        trial.update(kendall)

        results.append(trial)
        if save:
            res_path.write_text(json.dumps(results, indent=2))
            say("  ⭑ intermediate results saved")

    say(f"\nCompleted {len(results)} trials")
    say(f"  [1/3] L-α Mallows chose the following central ranking: {results[0]['sigma_0']}")
    say(f"  [2/3] Plackett-Luce is learned with utilities: {results[0]['utilities_PL']}")
    say(f"  [3/3] Kendall is learned with theta: {results[0]['theta_kendal']}")
    print_online_results(results)
    if save:
        say(f"Final results → {res_path}")

import sys

def fit_mallows(train, test, k, mc, delta, say):
    say(f"  [1/3] Starting to learn L-α Mallows model.")
    sigma_0 = consensus_ranking_estimation(train)
    alpha, beta = solve_alpha_beta(train, sigma_0, Delta=delta)
    say(f"        L-α Mallows is learned with alpha: {alpha:.4f}, beta: {beta:.4f}")
    say(f"        testing the Mallows model with {mc} samples...")
    # testing the model ...
    if len(train[0]) > 20:
        Delta_mc = 5
    else:
        Delta_mc = 7
    samples = sample_truncated_mallow(n=k, alpha=alpha, beta=beta, sigma=sigma_0, Delta=Delta_mc, num_samples=mc)
    evals = evaluate_metrics(test, samples)
    full_trial_results = {"sigma_0": sigma_0, "alpha": alpha, "beta": beta, **pack(*evals, suffix="_ML")}
    say(f"             top-1 hit rate: {full_trial_results['top_k_hit_rates_ML'][0]:.4f}")
    return full_trial_results

def fit_pl(train, test, mc, say):
    if len(train[0]) > 20: 
        say(f"  [2/3] Starting to learn Plackett-Luce model. This may take a while for {len(train[0])} items ...")
    else:
        say(f"  [2/3] Starting to learn Plackett-Luce model.")
    util, _ = learn_PL(train - 1, test - 1)
    say(f"        Plackett-Luce is learned.")
    say(f"        Testing the PL model with {mc} samples...")
    # testing the model ...
    samples = sample_PL(util, n_samples=mc)
    evals = evaluate_metrics(test, samples)
    full_trial_results = {"utilities": util, **pack(*evals, suffix="_PL")}
    say(f"             top-1 hit rate: {full_trial_results['top_k_hit_rates_PL'][0]:.4f}")
    return full_trial_results


def fit_kendall(train, test, mc, say): 
    if len(train[0]) > 20:
        say(f"  [3/3] Starting to learn Kendall model. This may take a while for {len(train[0])} items ...")
    else:
        say(f"  [3/3] Starting to learn Kendall model.")
    sigma_0, theta, _ = learn_kendal(train - 1, test - 1)
    say(f"        Kendall is learned with theta: {theta}.")
    say(f"        testing the Kendall model with {mc} samples...")
    # testing the model ...
    samples = sample_kendal(sigma_0=sigma_0, theta=theta, num_samples=mc)
    evals = evaluate_metrics(test, samples)
    full_trial_results = {**pack(*evals, suffix="_kendal"), "sigma_0": sigma_0, "theta": theta}
    say(f"             top-1 hit rate: {full_trial_results['top_k_hit_rates_kendal'][0]:.4f}")
    return full_trial_results

def pack(*vals, suffix=""):            # names metrics[0] → f"{prefix}metric_name"
    return {f"{n}{suffix}": v for n, v in zip(METRIC_NAMES, vals)}
