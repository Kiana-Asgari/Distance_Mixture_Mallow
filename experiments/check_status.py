"""Print the number of rows present in each experiment output CSV vs the
expected total for a full 50-trial run. Useful for watching long runs
from another shell and for auditing partial outputs.

Usage:
    python -m experiments.check_status
    python -m experiments.check_status --n-trials 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
FIGURES = Path(__file__).parent / "figures"


def row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return max(0, sum(1 for _ in f) - 1)  # exclude header


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--datasets-available", type=int, default=11,
                   help="how many of the 11 dataset specs your environment can load")
    args = p.parse_args()

    nt = args.n_trials
    nd = args.datasets_available

    checks = [
        (RESULTS / "held_out_log_likelihood.csv",             nd * nt * 6,    "held-out log-likelihood"),
        (RESULTS / "score_equation_accuracy.csv",             3 * 4 * 4 * 2,  "score-equation accuracy"),
        (RESULTS / "mcmc_mixing_diagnostics.csv",             3 * 4 * 4 * 4,  "MCMC mixing"),
        (RESULTS / "partition_function_error_small_alpha.csv", 3 * 4,          "partition-function error"),
        (RESULTS / "alpha_bound_sensitivity.csv",             6 * nt * 5,     "alpha-bound sensitivity"),
        (RESULTS / "kemeny_center_comparison.csv",            nd * nt * 2,    "Kemeny center comparison"),
        (RESULTS / "aggregate_metrics_high_precision.csv",    nd * 6,         "aggregate-metrics audit"),
        (RESULTS / "alpha_gain_synthetic_sweep.csv",          6 * 10,         "alpha-gain synthetic sweep"),
    ]
    figures = [
        FIGURES / "held_out_log_likelihood_summary.pdf",
        FIGURES / "alpha_bound_sensitivity.pdf",
        FIGURES / "mcmc_traceplots.pdf",
        FIGURES / "alpha_gain_scatter.pdf",
    ]

    print(f"{'experiment':35s} {'rows':>10s}  {'expected':>10s}  status")
    print("-" * 74)
    all_complete = True
    for path, expected, label in checks:
        got = row_count(path)
        status = "done" if got >= expected else f"{100*got/max(expected,1):5.1f}%"
        if got < expected:
            all_complete = False
        print(f"{label:35s} {got:10d}  {expected:10d}  {status}")

    print()
    print("Figures:")
    for fig in figures:
        print(f"  {'[x]' if fig.exists() else '[ ]'} {fig.name}")

    print()
    if all_complete:
        print("==> All tables at or above expected row count; the suite looks complete.")
    else:
        print("==> One or more CSVs are below the expected row count; re-run the affected scripts to fill in.")


if __name__ == "__main__":
    main()
