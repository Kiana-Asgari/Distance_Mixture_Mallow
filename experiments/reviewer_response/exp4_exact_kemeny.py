"""Experiment 4 -- Stronger Mallows-tau baseline using exact Kemeny center
on n=10 datasets, and a Borda + local-search heuristic for n=100.

Outputs:
  - results/exp4_mallows_tau_exact.csv     (per dataset, per centering method)
  - results/exp4_comparison.csv            (Borda vs. Kemeny side-by-side)
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from tqdm import tqdm

from benchmark.fit_Mallow_kendal import sample_kendal
from MLE.top_k import evaluate_metrics
from real_world_datasets.utils import check_zero_based_index
from experiments.reviewer_response.common.datasets import (
    DatasetUnavailable, SPORTS, all_dataset_specs, load_dataset, split,
)
from experiments.reviewer_response.common.kemeny import (
    borda_order,
    kemeny_exact_ilp,
    kemeny_local_search,
    kemeny_objective,
    precedence_matrix,
)
from experiments.reviewer_response.common.loglik import loglik_kendall, log_Z_kendall
from experiments.reviewer_response.common.results_io import append_csv_row, existing_keys

OUT_DIR = Path(__file__).parent / "results"


def fit_theta(rankings_zb: np.ndarray, sigma_zb: np.ndarray) -> float:
    """Fit theta via 1-D MLE given a fixed center sigma_zb (0-based)."""
    n = sigma_zb.shape[0]
    W = precedence_matrix(rankings_zb)
    obj_dist = kemeny_objective(sigma_zb, W) / len(rankings_zb)

    def nll(theta):
        return theta * obj_dist + log_Z_kendall(theta, n)

    res = minimize_scalar(nll, bounds=(1e-3, 5.0), method="bounded")
    return float(res.x)


def evaluate_one(
    train_zb: np.ndarray,
    test_zb: np.ndarray,
    test_one_based: np.ndarray,
    sigma_zb: np.ndarray,
    mc_samples: int,
):
    theta = fit_theta(train_zb, sigma_zb)
    samples = sample_kendal(theta=theta, sigma_0=sigma_zb, num_samples=mc_samples)
    metrics = evaluate_metrics(test_one_based, samples)
    log_lik = loglik_kendall(test_zb, sigma_zb, theta).mean()
    return theta, metrics, log_lik


def run_dataset(ds: str, n: int, n_trials: int, mc_samples: int, seed: int, time_limit_ilp: int):
    print(f"\n=== {ds} n={n} ===")
    try:
        data = load_dataset(ds, n)
    except DatasetUnavailable as exc:
        print(f"  -- skipped: {exc}")
        return []
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 1_000_000, n_trials)

    rows_all = []
    for trial in tqdm(range(n_trials), desc=f"{ds} n={n}"):
        train, test = split(ds, data, int(seeds[trial]))
        train_zb = check_zero_based_index(train)
        test_zb = check_zero_based_index(test)

        # Borda
        sigma_borda = borda_order(train_zb)
        t_b = time.time()
        theta_b, metrics_b, ll_b = evaluate_one(train_zb, test_zb, test, sigma_borda, mc_samples)
        time_b = time.time() - t_b
        obj_b = kemeny_objective(sigma_borda, precedence_matrix(train_zb))

        # Kemeny: exact for n<=10, local-search heuristic otherwise
        center_method = "kemeny_exact" if n <= 10 else "kemeny_localsearch"
        t_k = time.time()
        if n <= 10:
            sigma_k, obj_k, _ = kemeny_exact_ilp(train_zb, time_limit_s=time_limit_ilp)
        else:
            sigma_k, obj_k, _ = kemeny_local_search(train_zb)
        ilp_time = time.time() - t_k
        theta_k, metrics_k, ll_k = evaluate_one(train_zb, test_zb, test, sigma_k, mc_samples)

        for center, theta_v, metrics_v, ll_v, obj_v, run_t in [
            ("borda", theta_b, metrics_b, ll_b, obj_b, time_b),
            (center_method, theta_k, metrics_k, ll_k, obj_k, ilp_time),
        ]:
            rows_all.append({
                "dataset": ds, "n": n, "trial": trial, "center_method": center,
                "theta": float(theta_v),
                "kendall_tau": float(metrics_v["Kendall_tau"]),
                "spearman_rho": float(metrics_v["Spearman_rho"]),
                "top1_hit_rate": float(metrics_v["@precision_1"]),
                "loglik": float(ll_v),
                "center_objective_value": float(obj_v),
                "fit_seconds": float(run_t),
            })
    return rows_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--mc-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-limit-ilp", type=int, default=120)
    parser.add_argument("--datasets", type=str, default="all",
                        help="comma-separated 'name:n' pairs, or 'all', or 'small' (n=10 only)")
    parser.add_argument("--quick", action="store_true",
                        help="2 trials, 50 mc samples, n=10 only -- smoke test")
    args = parser.parse_args()

    if args.quick:
        args.n_trials = 2
        args.mc_samples = 50
        specs = [("sushi", 10), ("news", 10)]
    elif args.datasets == "all":
        specs = all_dataset_specs()
    elif args.datasets == "small":
        specs = [(d, k) for d, k in all_dataset_specs() if k == 10]
    else:
        specs = []
        for token in args.datasets.split(","):
            name, k = token.split(":")
            specs.append((name, int(k)))

    out_csv = OUT_DIR / "exp4_mallows_tau_exact.csv"
    fieldnames = [
        "dataset", "n", "trial", "center_method", "theta",
        "kendall_tau", "spearman_rho", "top1_hit_rate",
        "loglik", "center_objective_value", "fit_seconds",
    ]

    done = existing_keys(out_csv, ("dataset", "n", "trial", "center_method"))

    for ds, n in specs:
        ds_done = {(d, nn, t, c) for (d, nn, t, c) in done if d == ds and int(nn) == n}
        if len(ds_done) >= 2 * args.n_trials:
            print(f"-- skipping {ds} n={n} (already complete: {len(ds_done)} rows)")
            continue
        rows = run_dataset(ds, n, args.n_trials, args.mc_samples, args.seed, args.time_limit_ilp)
        for r in rows:
            append_csv_row(out_csv, r, fieldnames)

    # Build comparison table
    import pandas as pd
    if not out_csv.exists():
        return
    df = pd.read_csv(out_csv)
    summary = (df.groupby(["dataset", "n", "center_method"])
                 .agg(kendall_tau_mean=("kendall_tau", "mean"),
                      kendall_tau_sd=("kendall_tau", "std"),
                      top1_mean=("top1_hit_rate", "mean"),
                      top1_sd=("top1_hit_rate", "std"),
                      loglik_mean=("loglik", "mean"),
                      loglik_sd=("loglik", "std"),
                      center_obj_mean=("center_objective_value", "mean"),
                      n_trials=("trial", "count"))
                 .reset_index())
    summary.to_csv(OUT_DIR / "exp4_comparison.csv", index=False)
    print(f"Wrote {OUT_DIR / 'exp4_comparison.csv'}")


if __name__ == "__main__":
    main()
