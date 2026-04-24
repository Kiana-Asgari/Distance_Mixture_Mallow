"""Held-out approximate log-likelihood for L_alpha Mallows, L1-Mallows,
L2-Mallows, Mallows-tau, and Plackett-Luce.

For each (dataset, model, trial):
  1. Fit the model on the training split.
  2. Compute mean log P(pi) over the held-out test split.
  3. Aggregate across all trials.

Partition-function handling per model:
  Mallows-tau, Plackett-Luce -> exact closed form
  L1 / L2 / L_alpha (a>=1)   -> banded DP, truncation Delta picked so the
                                empirical relative gap to Delta+1 is <=
                                target_tv (1e-4 by default)
  L_alpha Mallows (a<1)      -> thermodynamic integration (5 chains)

Outputs:
  - results/held_out_log_likelihood.csv
  - results/held_out_log_likelihood_summary.csv
  - figures/held_out_log_likelihood_summary.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from benchmark.fit_Mallow_kendal import learn_kendal
from benchmark.fit_placket_luce import learn_PL
from MLE.alpha_beta_estimation import solve_alpha_beta
from MLE.consensus_ranking_estimation import consensus_ranking_estimation
from real_world_datasets.utils import check_zero_based_index

from experiments.common.datasets import (
    DatasetUnavailable, all_dataset_specs, load_dataset, split,
)
from experiments.common.loglik import (
    choose_truncation,
    log_Z_distance_dp,
    loglik_PL,
    loglik_distance,
    loglik_distance_mcmc,
    loglik_kendall,
)
from experiments.common.results_io import append_csv_row, existing_keys

OUT_DIR = Path(__file__).parent / "results"
FIG_DIR = Path(__file__).parent / "figures"


# ----------------------------------------------------------------------
# Per-model fit + log-likelihood
# ----------------------------------------------------------------------
def fit_and_loglik_distance(train, test, alpha_fixed: bool, alpha_value: float | None,
                            Delta_train: int, target_tv: float, mcmc_samples: int):
    """Fit L_alpha Mallows (LDER if alpha_fixed=False, L1/L2 otherwise)."""
    sigma_0 = consensus_ranking_estimation(
        train, alpha_fixed=alpha_fixed, alpha_fixed_value=alpha_value or 1,
    )
    alpha, beta = solve_alpha_beta(
        train, sigma_0, Delta=Delta_train,
        fixed_alpha=alpha_fixed,
        fixed_alpha_value=alpha_value or 1,
    )
    n = len(sigma_0)

    if alpha >= 1.0:
        D, gap = choose_truncation(n, alpha, beta, target_tv=target_tv)
        log_z = log_Z_distance_dp(n, alpha, beta, D)
        ll = loglik_distance(test, sigma_0, alpha, beta, D, log_z=log_z)
        method = f"banded_dp_D={D}"
        z_err = float(gap)
    else:
        ll, log_z, log_z_se = loglik_distance_mcmc(
            test, sigma_0, alpha, beta,
            n_samples_logZ=mcmc_samples,
            rng_seed=0,
        )
        method = "bridge_sampling"
        z_err = float(log_z_se)
    return {
        "loglik_mean": float(ll.mean()),
        "loglik_sd": float(ll.std()),
        "alpha": float(alpha), "beta": float(beta),
        "z_method": method, "z_error": z_err,
    }


def fit_and_loglik_kendall(train, test):
    sigma_0, theta, _ = learn_kendal(train - 1, test - 1)
    ll = loglik_kendall(check_zero_based_index(test), sigma_0, theta)
    return {
        "loglik_mean": float(ll.mean()),
        "loglik_sd": float(ll.std()),
        "alpha": float("nan"), "beta": float("nan"),
        "z_method": "exact_closed_form", "z_error": 0.0,
        "theta": float(theta),
    }


def fit_and_loglik_pl(train, test, ridge: bool):
    util, _, _ = learn_PL(train - 1, test - 1, use_cv=ridge)
    ll = loglik_PL(test - 1, util)
    return {
        "loglik_mean": float(ll.mean()),
        "loglik_sd": float(ll.std()),
        "alpha": float("nan"), "beta": float("nan"),
        "z_method": "exact_closed_form", "z_error": 0.0,
    }


def run(args):
    if args.quick:
        specs = [("sushi", 10), ("news", 10)]
        models = ["our", "L1", "L2", "tau", "pl"]
        n_trials = 2
    else:
        specs = args.specs
        models = ["our", "L1", "L2", "tau", "pl", "pl_reg"]
        n_trials = args.n_trials

    out_csv = OUT_DIR / "held_out_log_likelihood.csv"
    fieldnames = [
        "dataset", "n", "trial", "model",
        "loglik_mean", "loglik_sd",
        "alpha", "beta", "theta",
        "z_method", "z_error",
    ]
    done = existing_keys(out_csv, ("dataset", "n", "trial", "model"))

    rng_master = np.random.default_rng(args.seed)
    seeds = rng_master.integers(0, 1_000_000, n_trials)

    for ds, n in specs:
        try:
            data = load_dataset(ds, n)
        except DatasetUnavailable as exc:
            print(f"-- skipping {ds} n={n}: {exc}")
            continue
        for trial in tqdm(range(n_trials), desc=f"{ds} n={n}"):
            train, test = split(ds, data, int(seeds[trial]))
            for model in models:
                key = (ds, str(n), str(trial), model)
                if key in done:
                    continue
                row = {"dataset": ds, "n": n, "trial": trial, "model": model,
                       "theta": "", "alpha": "", "beta": "",
                       "z_method": "", "z_error": ""}
                try:
                    if model == "our":
                        res = fit_and_loglik_distance(
                            train, test, alpha_fixed=False, alpha_value=None,
                            Delta_train=args.delta_train, target_tv=args.target_tv,
                            mcmc_samples=args.mcmc_samples,
                        )
                    elif model == "L1":
                        res = fit_and_loglik_distance(
                            train, test, alpha_fixed=True, alpha_value=1,
                            Delta_train=args.delta_train, target_tv=args.target_tv,
                            mcmc_samples=args.mcmc_samples,
                        )
                    elif model == "L2":
                        res = fit_and_loglik_distance(
                            train, test, alpha_fixed=True, alpha_value=2,
                            Delta_train=args.delta_train, target_tv=args.target_tv,
                            mcmc_samples=args.mcmc_samples,
                        )
                    elif model == "tau":
                        res = fit_and_loglik_kendall(train, test)
                    elif model == "pl":
                        res = fit_and_loglik_pl(train, test, ridge=False)
                    elif model == "pl_reg":
                        res = fit_and_loglik_pl(train, test, ridge=True)
                    else:
                        continue
                    row.update(res)
                    append_csv_row(out_csv, row, fieldnames)
                except Exception as exc:
                    print(f"  !! {model} on {ds} n={n} trial {trial} failed: {exc}")

    # Plot summary bar chart
    if not out_csv.exists():
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(out_csv)
    summary = df.groupby(["dataset", "n", "model"]).agg(
        ll_mean=("loglik_mean", "mean"),
        ll_sd=("loglik_mean", "std"),
        n_trials=("trial", "count"),
    ).reset_index()
    summary.to_csv(OUT_DIR / "held_out_log_likelihood_summary.csv", index=False)

    datasets_plotted = summary[["dataset", "n"]].drop_duplicates().values.tolist()
    if datasets_plotted:
        ncols = min(3, len(datasets_plotted))
        nrows = (len(datasets_plotted) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        for idx, (ds, n) in enumerate(datasets_plotted):
            ax = axes[idx // ncols][idx % ncols]
            sub = summary[(summary.dataset == ds) & (summary.n == n)]
            ax.bar(sub.model, sub.ll_mean, yerr=sub.ll_sd.fillna(0),
                   capsize=4, color="steelblue", edgecolor="black")
            ax.set_title(f"{ds} (n={n})")
            ax.set_ylabel("mean held-out log-likelihood")
            ax.tick_params(axis="x", rotation=30)
        for idx in range(len(datasets_plotted), nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        out_pdf = FIG_DIR / "held_out_log_likelihood_summary.pdf"
        fig.tight_layout()
        fig.savefig(out_pdf)
        print(f"wrote {out_pdf}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--delta-train", type=int, default=7)
    p.add_argument("--target-tv", type=float, default=1e-4)
    p.add_argument("--mcmc-samples", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--datasets", type=str, default="all")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.datasets == "all":
        args.specs = all_dataset_specs()
    else:
        args.specs = []
        for tok in args.datasets.split(","):
            name, k = tok.split(":")
            args.specs.append((name, int(k)))
    run(args)


if __name__ == "__main__":
    main()
