"""Top-level runner that executes every experiment script in sequence.

The default ordering is:

    aggregate_metrics_audit -> kemeny_center_comparison
      -> held_out_log_likelihood -> alpha_bound_sensitivity
      -> partition_function_diagnostics -> alpha_gain_analysis

Pass `--only name1,name2` to run a subset, and `--quick` to execute each
script with a reduced grid for quick validation.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time

EXPERIMENTS = [
    ("aggregate_metrics_audit",         "experiments.aggregate_metrics_audit"),
    ("kemeny_center_comparison",        "experiments.kemeny_center_comparison"),
    ("held_out_log_likelihood",         "experiments.held_out_log_likelihood"),
    ("alpha_bound_sensitivity",         "experiments.alpha_bound_sensitivity"),
    ("partition_function_diagnostics",  "experiments.partition_function_diagnostics"),
    ("alpha_gain_analysis",             "experiments.alpha_gain_analysis"),
]

# Scripts that accept `--n-trials`; the rest have self-contained grids.
N_TRIALS_SUPPORT = {
    "held_out_log_likelihood",
    "alpha_bound_sensitivity",
    "kemeny_center_comparison",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="run every selected script with a reduced grid for quick validation")
    p.add_argument("--only", type=str, default="",
                   help="comma-separated script names (e.g. "
                        "'held_out_log_likelihood,alpha_bound_sensitivity')")
    p.add_argument("--n-trials", type=int, default=50)
    args = p.parse_args()

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    known = {name for name, _ in EXPERIMENTS}
    unknown = only - known
    if unknown:
        print(f"Unknown script names: {sorted(unknown)}", file=sys.stderr)
        sys.exit(2)

    selected = [(name, mod) for name, mod in EXPERIMENTS if not only or name in only]

    for name, modname in selected:
        print("\n" + "=" * 60)
        print(f"Running {name}")
        print("=" * 60)
        mod = importlib.import_module(modname)
        argv_backup = sys.argv[:]
        sys.argv = [modname]
        if args.quick:
            sys.argv.append("--quick")
        elif name in N_TRIALS_SUPPORT:
            sys.argv += ["--n-trials", str(args.n_trials)]
        t0 = time.time()
        try:
            mod.main()
            print(f"\nFinished {name} in {time.time() - t0:.1f}s")
        except Exception as exc:
            print(f"\n!! {name} crashed: {exc}")
        sys.argv = argv_backup


if __name__ == "__main__":
    main()
