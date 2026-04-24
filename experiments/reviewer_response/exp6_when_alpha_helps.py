"""Experiment 6 (optional) -- When does learning alpha help?

Part A: Synthetic sweep over alpha_0 in {0.2, 0.5, 1.0, 1.5, 2.0, 3.0}.
        For each, fit LDER, L1, L2 and report held-out log-likelihood gap
        between LDER and the best fixed-alpha competitor.

Part B: Real-data scatter -- gap of LDER over L1 vs |alpha_hat - 1|.
        Reads pre-computed Exp 1 log-likelihood results.
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

from experiments.reviewer_response.common.loglik import (
    choose_truncation,
    log_Z_distance_dp,
    loglik_distance,
    loglik_distance_mcmc,
)
from experiments.reviewer_response.common.results_io import append_csv_row, existing_keys

OUT_DIR = Path(__file__).parent / "results"
FIG_DIR = Path(__file__).parent / "figures"


def fit_distance(train, test, alpha_fixed: bool, alpha_value: float | None,
                 Delta_train: int, target_tv: float, mcmc_samples: int):
    sigma_0 = consensus_ranking_estimation(train, alpha_fixed=alpha_fixed,
                                           alpha_fixed_value=alpha_value or 1)
    alpha, beta = solve_alpha_beta(train, sigma_0, Delta=Delta_train,
                                   fixed_alpha=alpha_fixed,
                                   fixed_alpha_value=alpha_value or 1)
    n = len(sigma_0)
    if alpha >= 1.0:
        D, _ = choose_truncation(n, float(alpha), float(beta), target_tv=target_tv)
        log_z = log_Z_distance_dp(n, float(alpha), float(beta), D)
        ll = loglik_distance(test, sigma_0, float(alpha), float(beta), D, log_z=log_z)
    else:
        ll, _, _ = loglik_distance_mcmc(test, sigma_0, float(alpha), float(beta),
                                        n_samples_logZ=mcmc_samples, rng_seed=0)
    return float(ll.mean()), float(alpha), float(beta)


def part_a(args):
    out_csv = OUT_DIR / "exp6_synthetic_sweep.csv"
    if out_csv.exists() and not args.append:
        out_csv.unlink()
    fieldnames = [
        "alpha_0", "beta_0", "n", "m_train", "trial",
        "ll_lder", "ll_L1", "ll_L2",
        "lder_alpha_hat", "lder_beta_hat",
        "gap_lder_vs_best",
    ]

    n = args.n_items
    beta_0 = args.beta_0
    alphas_0 = args.alphas_0

    rng_master = np.random.default_rng(args.seed)

    for alpha_0 in alphas_0:
        for trial in tqdm(range(args.n_trials), desc=f"alpha_0={alpha_0}"):
            rng_seed = int(rng_master.integers(0, 2**31))
            sigma_id = np.arange(1, n + 1)
            train = np.array(sample_truncated_mallow(
                n=n, alpha=alpha_0, beta=beta_0, sigma=sigma_id,
                Delta=min(args.delta_data, n - 1),
                num_samples=args.m_train, rng_seed=rng_seed,
            ))
            test = np.array(sample_truncated_mallow(
                n=n, alpha=alpha_0, beta=beta_0, sigma=sigma_id,
                Delta=min(args.delta_data, n - 1),
                num_samples=args.m_test, rng_seed=rng_seed + 1,
            ))

            ll_lder, a_lder, b_lder = fit_distance(
                train, test, alpha_fixed=False, alpha_value=None,
                Delta_train=args.delta_train,
                target_tv=args.target_tv,
                mcmc_samples=args.mcmc_samples,
            )
            ll_L1, _, _ = fit_distance(
                train, test, alpha_fixed=True, alpha_value=1,
                Delta_train=args.delta_train,
                target_tv=args.target_tv,
                mcmc_samples=args.mcmc_samples,
            )
            ll_L2, _, _ = fit_distance(
                train, test, alpha_fixed=True, alpha_value=2,
                Delta_train=args.delta_train,
                target_tv=args.target_tv,
                mcmc_samples=args.mcmc_samples,
            )
            best_fixed = max(ll_L1, ll_L2)
            append_csv_row(out_csv, {
                "alpha_0": alpha_0, "beta_0": beta_0, "n": n,
                "m_train": args.m_train, "trial": trial,
                "ll_lder": ll_lder, "ll_L1": ll_L1, "ll_L2": ll_L2,
                "lder_alpha_hat": a_lder, "lder_beta_hat": b_lder,
                "gap_lder_vs_best": ll_lder - best_fixed,
            }, fieldnames)


def part_b(args):
    """Reads Experiment 1's log-likelihood CSV and produces a scatter."""
    exp1_csv = OUT_DIR / "exp1_loglik.csv"
    if not exp1_csv.exists():
        print("Exp 1 results not found; skip Part B.")
        return
    df = pd.read_csv(exp1_csv)
    df = df[df.n == 10]
    summary = df.groupby(["dataset", "model"]).agg(
        ll_mean=("loglik_mean", "mean"),
        alpha_hat=("alpha", "mean"),
    ).reset_index()
    pivot = summary.pivot_table(index="dataset", columns="model",
                                values=["ll_mean", "alpha_hat"])
    rows = []
    for ds, _ in pivot.iterrows():
        try:
            ll_lder = pivot.loc[ds, ("ll_mean", "our")]
            ll_l1 = pivot.loc[ds, ("ll_mean", "L1")]
            alpha_hat = pivot.loc[ds, ("alpha_hat", "our")]
            rows.append({
                "dataset": ds,
                "alpha_hat": float(alpha_hat),
                "abs_alpha_hat_minus_1": abs(float(alpha_hat) - 1.0),
                "loglik_gain": float(ll_lder - ll_l1),
            })
        except KeyError:
            continue
    out_csv = OUT_DIR / "exp6_real_scatter.csv"
    if rows:
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        for r in rows:
            ax.scatter(r["abs_alpha_hat_minus_1"], r["loglik_gain"], s=60)
            ax.annotate(r["dataset"], (r["abs_alpha_hat_minus_1"],
                                       r["loglik_gain"]),
                        textcoords="offset points", xytext=(5, 5))
        ax.set_xlabel("|alpha_hat - 1|")
        ax.set_ylabel("LDER log-likelihood gain over L1")
        ax.set_title("LDER vs L1: log-likelihood gain by distance from alpha=1")
        ax.axhline(0, color="grey", lw=0.5)
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "exp6_scatter.pdf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=10)
    p.add_argument("--m-train", type=int, default=500)
    p.add_argument("--m-test", type=int, default=500)
    p.add_argument("--n-items", type=int, default=10)
    p.add_argument("--beta-0", type=float, default=0.5)
    p.add_argument("--alphas-0", type=float, nargs="*",
                   default=[0.2, 0.5, 1.0, 1.5, 2.0, 3.0])
    p.add_argument("--delta-data", type=int, default=8)
    p.add_argument("--delta-train", type=int, default=7)
    p.add_argument("--target-tv", type=float, default=1e-4)
    p.add_argument("--mcmc-samples", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--parts", type=str, default="A,B")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--append", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.n_trials = 2
        args.m_train = 100
        args.m_test = 100
        args.alphas_0 = [0.5, 1.0, 2.0]
        args.mcmc_samples = 2000

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parts = [s.strip().upper() for s in args.parts.split(",")]
    if "A" in parts:
        part_a(args)
    if "B" in parts:
        part_b(args)


if __name__ == "__main__":
    main()
