"""Diagnostics for the L_alpha Mallows partition function and score
equations in the small-alpha regime.

Part A: brute-force enumeration vs MCMC vs the banded-DP implementation
        for E[d_alpha] and E[d_dot_alpha], on grid
        n in {6, 8, 10}, alpha in {0.1, 0.3, 0.5, 0.8},
        beta in {0.1, 0.5, 1.0, 2.0}.

Part B: MCMC mixing diagnostics (Gelman-Rubin R-hat, Geyer ESS, trace
        plots) on the same grid, five independent chains.

Part C: Partition-function approximation error for alpha < 1 via bridge
        sampling and thermodynamic integration against exact enumeration.
"""

from __future__ import annotations

import argparse
import time
from itertools import product
from pathlib import Path

import numpy as np
from tqdm import tqdm

from MLE.score_function import compute_entry  # banded-DP-based expectations
from experiments.common.distances import (
    bridge_sample_logZ,
    d_alpha,
    d_alpha_dot,
    effective_sample_size,
    exact_Z,
    exact_expectations,
    gelman_rubin,
    mh_sample,
    ti_logZ,
)
from experiments.common.results_io import append_csv_row

OUT_DIR = Path(__file__).parent / "results"
FIG_DIR = Path(__file__).parent / "figures"


# ------------------------------------------------------------------
def part_a(args):
    out_csv = OUT_DIR / "score_equation_accuracy.csv"
    if out_csv.exists() and not args.append:
        out_csv.unlink()

    fieldnames = [
        "n", "alpha", "beta", "expectation_name",
        "exact", "mcmc", "current_impl",
        "mcmc_rel_err", "current_rel_err",
        "mcmc_seconds", "current_seconds",
    ]

    grid = list(product(args.ns, args.alphas, args.betas))
    for n, alpha, beta in tqdm(grid, desc="part A"):
        Z, Ed_e, Edot_e = exact_expectations(n, alpha, beta)

        sigma_id = np.arange(1, n + 1)
        t0 = time.time()
        samples = mh_sample(
            n=n, alpha=alpha, beta=beta,
            n_samples=args.mcmc_samples,
            burn_in=args.mcmc_burn,
            thin=1, rng_seed=args.seed,
            init=sigma_id.copy(),
        )
        d_vals = np.array([d_alpha(s, sigma_id, alpha) for s in samples])
        ddot_vals = np.array([d_alpha_dot(s, sigma_id, alpha) for s in samples])
        Ed_m = float(d_vals.mean())
        Edot_m = float(ddot_vals.mean())
        mcmc_t = time.time() - t0

        # Current implementation: use compute_entry (DP).  Use Delta = n-1 to
        # match the unrestricted distribution as closely as possible.
        t0 = time.time()
        Delta_imp = min(n - 1, 8)
        try:
            _, Ed_c, Edot_c = compute_entry(alpha, beta, n, Delta_imp)
            Ed_c = float(Ed_c); Edot_c = float(Edot_c)
        except Exception as exc:
            Ed_c = float("nan"); Edot_c = float("nan")
            print(f"compute_entry failed: {exc}")
        cur_t = time.time() - t0

        for label, exact_v, mc_v, cur_v in [
            ("E_d_alpha", Ed_e, Ed_m, Ed_c),
            ("E_ddot_alpha", Edot_e, Edot_m, Edot_c),
        ]:
            denom = max(abs(exact_v), 1e-12)
            row = {
                "n": n, "alpha": alpha, "beta": beta,
                "expectation_name": label,
                "exact": exact_v, "mcmc": mc_v, "current_impl": cur_v,
                "mcmc_rel_err": abs(mc_v - exact_v) / denom,
                "current_rel_err": abs(cur_v - exact_v) / denom if not np.isnan(cur_v) else float("nan"),
                "mcmc_seconds": mcmc_t, "current_seconds": cur_t,
            }
            append_csv_row(out_csv, row, fieldnames)


# ------------------------------------------------------------------
def part_b(args):
    out_csv = OUT_DIR / "mcmc_mixing_diagnostics.csv"
    if out_csv.exists() and not args.append:
        out_csv.unlink()
    fieldnames = ["n", "alpha", "beta", "stat", "value"]

    trace_storage = {}
    grid = list(product(args.ns, args.alphas, args.betas))
    n_chains = 5
    for n, alpha, beta in tqdm(grid, desc="part B"):
        chains_d = np.empty((n_chains, args.mcmc_samples))
        chains_ddot = np.empty((n_chains, args.mcmc_samples))
        sigma_id = np.arange(1, n + 1)
        for c in range(n_chains):
            samples = mh_sample(
                n=n, alpha=alpha, beta=beta,
                n_samples=args.mcmc_samples,
                burn_in=args.mcmc_burn,
                rng_seed=args.seed + 100 * c,
            )
            chains_d[c] = np.array([d_alpha(s, sigma_id, alpha) for s in samples])
            chains_ddot[c] = np.array([d_alpha_dot(s, sigma_id, alpha) for s in samples])
        rhat_d = gelman_rubin(chains_d)
        rhat_ddot = gelman_rubin(chains_ddot)
        ess_d = float(np.mean([effective_sample_size(c) for c in chains_d]))
        ess_ddot = float(np.mean([effective_sample_size(c) for c in chains_ddot]))
        for stat, val in [
            ("rhat_d_alpha", rhat_d), ("rhat_ddot_alpha", rhat_ddot),
            ("ess_d_alpha", ess_d), ("ess_ddot_alpha", ess_ddot),
        ]:
            append_csv_row(out_csv, {
                "n": n, "alpha": alpha, "beta": beta, "stat": stat, "value": float(val),
            }, fieldnames)
        if n == 10 and alpha in (0.1, 0.5) and beta in (0.5, 2.0):
            trace_storage[(alpha, beta)] = chains_d.copy()

    # Trace plots
    if trace_storage:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        keys = sorted(trace_storage.keys())
        fig, axes = plt.subplots(len(keys), 1, figsize=(8, 3 * len(keys)), squeeze=False)
        for i, (alpha, beta) in enumerate(keys):
            ax = axes[i][0]
            for c in range(trace_storage[(alpha, beta)].shape[0]):
                ax.plot(trace_storage[(alpha, beta)][c], lw=0.6, alpha=0.7,
                        label=f"chain {c+1}")
            ax.set_title(f"n=10, alpha={alpha}, beta={beta}: trace of d_alpha")
            ax.set_xlabel("iteration after burn-in")
            ax.set_ylabel("d_alpha")
            ax.legend(fontsize=7)
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "mcmc_traceplots.pdf")


# ------------------------------------------------------------------
def part_c(args):
    out_csv = OUT_DIR / "partition_function_error_small_alpha.csv"
    if out_csv.exists() and not args.append:
        out_csv.unlink()
    fieldnames = [
        "n", "alpha", "beta",
        "log_Z_exact",
        "log_Z_bridge", "bridge_rel_err", "bridge_seconds",
        "log_Z_ti",     "ti_rel_err",     "ti_seconds",
        "exact_seconds",
    ]
    beta = 2.0
    grid = list(product(args.ns, args.alphas))
    for n, alpha in tqdm(grid, desc="part C"):
        t0 = time.time()
        Z_exact = exact_Z(n, alpha, beta)
        log_Z_exact = float(np.log(Z_exact))
        exact_t = time.time() - t0

        t0 = time.time()
        log_Z_bridge = bridge_sample_logZ(
            n=n, alpha=alpha, beta=beta,
            n_samples=args.mcmc_samples_logZ,
            rng_seed=args.seed,
        )
        bridge_t = time.time() - t0

        t0 = time.time()
        log_Z_ti = ti_logZ(
            n=n, alpha=alpha, beta=beta,
            n_steps=16,
            n_samples_per_step=max(500, args.mcmc_samples_logZ // 16),
            burn_in=max(200, args.mcmc_samples_logZ // 32),
            rng_seed=args.seed,
        )
        ti_t = time.time() - t0

        denom = max(abs(log_Z_exact), 1e-12)
        append_csv_row(out_csv, {
            "n": n, "alpha": alpha, "beta": beta,
            "log_Z_exact": log_Z_exact,
            "log_Z_bridge": log_Z_bridge,
            "bridge_rel_err": abs(log_Z_bridge - log_Z_exact) / denom,
            "bridge_seconds": bridge_t,
            "log_Z_ti": log_Z_ti,
            "ti_rel_err": abs(log_Z_ti - log_Z_exact) / denom,
            "ti_seconds": ti_t,
            "exact_seconds": exact_t,
        }, fieldnames)


# ------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ns", type=int, nargs="*", default=[6, 8, 10])
    p.add_argument("--alphas", type=float, nargs="*", default=[0.1, 0.3, 0.5, 0.8])
    p.add_argument("--betas", type=float, nargs="*", default=[0.1, 0.5, 1.0, 2.0])
    p.add_argument("--mcmc-samples", type=int, default=20_000,
                   help="samples per chain (Part B uses 5 chains)")
    p.add_argument("--mcmc-burn", type=int, default=2_000)
    p.add_argument("--mcmc-samples-logZ", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true",
                   help="use a single small (n, alpha, beta) grid for quick validation")
    p.add_argument("--parts", type=str, default="A,B,C")
    p.add_argument("--append", action="store_true",
                   help="append to existing CSVs (default: overwrite)")
    args = p.parse_args()

    if args.quick:
        args.ns = [6]
        args.alphas = [0.5]
        args.betas = [1.0]
        args.mcmc_samples = 2000
        args.mcmc_samples_logZ = 2000

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parts = [s.strip().upper() for s in args.parts.split(",")]
    if "A" in parts:
        part_a(args)
    if "B" in parts:
        part_b(args)
    if "C" in parts:
        part_c(args)


if __name__ == "__main__":
    main()
