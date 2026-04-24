"""Top-level runner that executes all reviewer-response experiments in
the order specified by the reviewer-response plan:

  Exp 5 -> Exp 4 -> Exp 1 -> Exp 3 -> Exp 2 -> Exp 6

Use `--quick` to run smoke-test versions of every experiment.
Pass `--only 1,3` to run only specific experiments.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

EXPERIMENTS = [
    ("5", "experiments.reviewer_response.exp5_table2_precision"),
    ("4", "experiments.reviewer_response.exp4_exact_kemeny"),
    ("1", "experiments.reviewer_response.exp1_loglik"),
    ("3", "experiments.reviewer_response.exp3_alpha_floor"),
    ("2", "experiments.reviewer_response.exp2_alpha_lt1_diagnostics"),
    ("6", "experiments.reviewer_response.exp6_when_alpha_helps"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="run smoke-test versions of each experiment")
    p.add_argument("--only", type=str, default="",
                   help="comma-separated experiment IDs to run (e.g. '1,3')")
    p.add_argument("--n-trials", type=int, default=50)
    args = p.parse_args()

    only = [s.strip() for s in args.only.split(",") if s.strip()]
    selected = [(eid, mod) for eid, mod in EXPERIMENTS if not only or eid in only]

    for eid, modname in selected:
        print("\n" + "=" * 60)
        print(f"Running Experiment {eid}: {modname}")
        print("=" * 60)
        mod = importlib.import_module(modname)
        argv_backup = sys.argv[:]
        sys.argv = [modname]
        if args.quick:
            sys.argv.append("--quick")
        elif eid in {"1", "3", "4"}:
            sys.argv += ["--n-trials", str(args.n_trials)]
        t0 = time.time()
        try:
            mod.main()
            print(f"\nFinished Experiment {eid} in {time.time() - t0:.1f}s")
        except Exception as exc:
            print(f"\n!! Experiment {eid} crashed: {exc}")
        sys.argv = argv_backup


if __name__ == "__main__":
    main()
