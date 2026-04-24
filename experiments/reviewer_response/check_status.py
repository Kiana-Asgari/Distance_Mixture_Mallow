"""Progress checker for the reviewer-response run. Prints the number of
rows present in each expected output CSV vs the row count expected for a
full --n-trials 50 run across all available datasets.

Usage:
    python -m experiments.reviewer_response.check_status
    python -m experiments.reviewer_response.check_status --n-trials 50
"""

from __future__ import annotations

import argparse
import csv
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
        # path,                                                expected rows,                 label
        (RESULTS / "exp1_loglik.csv",                          nd * nt * 6,                    "Exp 1 log-likelihood"),
        (RESULTS / "exp2a_score_accuracy.csv",                 3 * 4 * 4 * 2,                  "Exp 2A score accuracy"),
        (RESULTS / "exp2b_mcmc_diagnostics.csv",               3 * 4 * 4 * 4,                  "Exp 2B mixing"),
        (RESULTS / "exp2c_Z_approx_alpha_lt1.csv",             3 * 4,                          "Exp 2C Z approx"),
        (RESULTS / "exp3_alpha_floor.csv",                     6 * nt * 5,                     "Exp 3 alpha floor"),
        (RESULTS / "exp4_mallows_tau_exact.csv",               nd * nt * 2,                    "Exp 4 Kemeny"),
        (RESULTS / "exp5_table2_high_precision.csv",           nd * 6,                         "Exp 5 Table 2 audit"),
        (RESULTS / "exp6_synthetic_sweep.csv",                 6 * 10,                         "Exp 6A synthetic"),
    ]
    figures = [
        FIGURES / "exp1_loglik_summary.pdf",
        FIGURES / "exp3_sensitivity.pdf",
        FIGURES / "exp2b_traceplots.pdf",
        FIGURES / "exp6_scatter.pdf",
    ]

    print(f"{'experiment':25s} {'rows':>10s}  {'expected':>10s}  status")
    print("-" * 64)
    all_complete = True
    for path, expected, label in checks:
        got = row_count(path)
        status = "done" if got >= expected else f"{100*got/max(expected,1):5.1f}%"
        if got < expected:
            all_complete = False
        print(f"{label:25s} {got:10d}  {expected:10d}  {status}")

    print()
    print("Figures:")
    for fig in figures:
        print(f"  {'[x]' if fig.exists() else '[ ]'} {fig.name}")

    print()
    if all_complete:
        print("==> All tables at or above expected row count. Full run looks complete.")
    else:
        print("==> Run is still in progress or was interrupted. Re-invoke run_all.py to resume.")


if __name__ == "__main__":
    main()
