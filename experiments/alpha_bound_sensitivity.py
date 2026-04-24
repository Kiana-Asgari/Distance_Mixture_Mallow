"""Sensitivity of the L_alpha Mallows MLE to the lower bound on alpha.

Notes on the default lower bound in the MLE implementation
----------------------------------------------------------
* ``MLE.alpha_beta_estimation.solve_alpha_beta`` uses
  ``alpha_bounds=(1e-1, 3)`` by default, enforced by both the
  least-squares path (n <= 20) and the differential-evolution path
  (n > 20).
* The lookup tables shipped under ``GMM_diagonalized/lookup_tables/``
  only have alpha values starting at 0.1, so any fit path that consults
  the lookup tables (n >= 30 in the shipped assets) cannot explore
  ``alpha_min < 0.1`` without re-computing those tables.
* This script restricts itself to n=10 real-world datasets so the DP
  path (which does not use the lookup tables) is exercised and
  ``alpha_min`` can legitimately be varied in {0.01, 0.05, 0.1, 0.2, 0.5}.

Outputs:
  - results/alpha_bound_sensitivity.csv
  - results/alpha_bound_sensitivity_summary.csv
  - figures/alpha_bound_sensitivity.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from GMM_diagonalized.sampling import sample_truncated_mallow
from MLE.alpha_beta_estimation import solve_alpha_beta
from MLE.consensus_ranking_estimation import consensus_ranking_estimation
from MLE.top_k import evaluate_metrics

from experiments.common.datasets import (
    DatasetUnavailable, load_dataset, split,
)
from experiments.common.loglik import (
    choose_truncation,
    log_Z_distance_dp,
    loglik_distance,
    loglik_distance_mcmc,
)
from experiments.common.results_io import append_csv_row, existing_keys

OUT_DIR = Path(__file__).parent / "results"
FIG_DIR = Path(__file__).parent / "figures"
DATASETS_N10 = [("football", 10), ("basketball", 10), ("baseball", 10),
                ("sushi", 10), ("news", 10), ("movie_lens", 10)]
ALPHA_MINS = [0.01, 0.05, 0.1, 0.2, 0.5]


def fit_lder_custom_floor(train, test, alpha_min: float, Delta: int = 7,
                          mcmc_samples: int = 5000, target_tv: float = 1e-4,
                          max_refit_iters: int = 3):
    """Fit L_alpha Mallows with a custom alpha_min and evaluate held-out log-lik.

    Uses the same Delta for fitting and evaluation: after an initial fit at
    ``Delta``, ``choose_truncation`` picks the bandwidth that meets
    ``target_tv``, and the fit is repeated at that bandwidth (up to
    ``max_refit_iters`` iterations) so that the score-equation approximation
    and the log-likelihood approximation agree.
    """
    sigma_0 = consensus_ranking_estimation(train, alpha_fixed=False)
    n = len(sigma_0)

    current_delta = Delta
    alpha = beta = None
    D = Delta
    for _ in range(max_refit_iters + 1):
        alpha, beta = solve_alpha_beta(
            train, sigma_0, Delta=current_delta,
            alpha_bounds=(alpha_min, 3.0), beta_bounds=(1e-3, 2.0),
            fixed_alpha=False,
        )
        if alpha < 1.0:
            break
        D, _ = choose_truncation(n, float(alpha), float(beta), target_tv=target_tv)
        if D == current_delta:
            break
        current_delta = D

    if alpha >= 1.0:
        log_z = log_Z_distance_dp(n, float(alpha), float(beta), D)
        ll = loglik_distance(test, sigma_0, float(alpha), float(beta), D, log_z=log_z)
    else:
        ll, _, _ = loglik_distance_mcmc(
            test, sigma_0, float(alpha), float(beta),
            n_samples_logZ=mcmc_samples, rng_seed=0,
        )
    samples = sample_truncated_mallow(
        n=n, alpha=float(alpha), beta=float(beta), sigma=sigma_0,
        Delta=min(Delta, n - 1), num_samples=500,
    )
    metrics = evaluate_metrics(test, samples)
    return {
        "alpha_hat": float(alpha), "beta_hat": float(beta),
        "kendall_tau": float(metrics["Kendall_tau"]),
        "top1_hit_rate": float(metrics["@precision_1"]),
        "loglik_mean": float(ll.mean()),
    }


def run(args):
    specs = [("sushi", 10), ("news", 10)] if args.quick else DATASETS_N10
    if args.datasets:
        specs = []
        for tok in args.datasets.split(","):
            name, k = tok.split(":")
            specs.append((name, int(k)))
    n_trials = 2 if args.quick else args.n_trials
    alpha_mins = args.alpha_mins

    out_csv = OUT_DIR / args.output_csv
    fieldnames = [
        "dataset", "n", "trial", "alpha_min",
        "alpha_hat", "beta_hat",
        "kendall_tau", "top1_hit_rate", "loglik",
    ]
    done = existing_keys(out_csv, ("dataset", "n", "trial", "alpha_min"))

    rng_master = np.random.default_rng(args.seed)
    seeds = rng_master.integers(0, 1_000_000, n_trials)

    for ds, n in specs:
        try:
            data = load_dataset(ds, n)
        except DatasetUnavailable as exc:
            print(f"-- skipping {ds}: {exc}")
            continue
        for trial in tqdm(range(n_trials), desc=f"{ds} n={n}"):
            train, test = split(ds, data, int(seeds[trial]))
            for alpha_min in alpha_mins:
                key = (ds, str(n), str(trial), f"{alpha_min}")
                if key in done:
                    continue
                try:
                    res = fit_lder_custom_floor(train, test, alpha_min)
                except Exception as exc:
                    print(f"  !! {ds} trial {trial} alpha_min={alpha_min}: {exc}")
                    continue
                append_csv_row(out_csv, {
                    "dataset": ds, "n": n, "trial": trial,
                    "alpha_min": alpha_min,
                    "alpha_hat": res["alpha_hat"], "beta_hat": res["beta_hat"],
                    "kendall_tau": res["kendall_tau"],
                    "top1_hit_rate": res["top1_hit_rate"],
                    "loglik": res["loglik_mean"],
                }, fieldnames)

    # Aggregated plot
    if not out_csv.exists():
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(out_csv)
    agg = df.groupby(["dataset", "n", "alpha_min"]).agg(
        alpha_hat_mean=("alpha_hat", "mean"),
        alpha_hat_sd=("alpha_hat", "std"),
        beta_hat_mean=("beta_hat", "mean"),
        beta_hat_sd=("beta_hat", "std"),
        kendall_mean=("kendall_tau", "mean"),
        top1_mean=("top1_hit_rate", "mean"),
        loglik_mean=("loglik", "mean"),
    ).reset_index()
    agg.to_csv(OUT_DIR / "alpha_bound_sensitivity_summary.csv", index=False)

    datasets_plotted = agg[["dataset", "n"]].drop_duplicates().values.tolist()
    if datasets_plotted:
        fig, axes = plt.subplots(len(datasets_plotted), 2,
                                 figsize=(10, 3 * len(datasets_plotted)),
                                 squeeze=False)
        for i, (ds, n) in enumerate(datasets_plotted):
            sub = agg[(agg.dataset == ds) & (agg.n == n)].sort_values("alpha_min")
            axes[i][0].errorbar(sub.alpha_min, sub.alpha_hat_mean,
                                yerr=sub.alpha_hat_sd.fillna(0),
                                marker="o", capsize=3)
            axes[i][0].plot(sub.alpha_min, sub.alpha_min, "k--",
                            label="alpha_min = alpha_hat")
            axes[i][0].set_xscale("log")
            axes[i][0].set_title(f"{ds} (n={n}): alpha_hat vs alpha_min")
            axes[i][0].set_xlabel("alpha_min")
            axes[i][0].set_ylabel("alpha_hat")
            axes[i][0].legend(fontsize=8)

            axL = axes[i][1]
            axL.plot(sub.alpha_min, sub.kendall_mean, marker="o", label="Kendall tau")
            axL.plot(sub.alpha_min, sub.top1_mean, marker="s", label="Top-1 hit")
            axL.set_xscale("log")
            axL.set_xlabel("alpha_min")
            axL.set_ylabel("metric")
            axL.set_title(f"{ds} (n={n}): test metrics vs alpha_min")
            axL.legend(fontsize=8)
        fig.tight_layout()
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / "alpha_bound_sensitivity.pdf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha-mins", type=float, nargs="*", default=ALPHA_MINS)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--datasets", type=str, default="",
                   help="comma-separated 'name:n' pairs; empty = default n=10 list")
    p.add_argument("--output-csv", type=str, default="alpha_bound_sensitivity.csv",
                   help="filename under results/ (default: alpha_bound_sensitivity.csv)")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
