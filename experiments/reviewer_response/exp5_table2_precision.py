"""Experiment 5 -- High-precision regeneration of Table 2 entries that
appeared identical across LDER, L1-Mallows, and L2-Mallows on Basketball
(n=100) and Baseball (n=100).

Source of numbers: real_world_datasets/results/{basketball,baseball}_n_teams=100.json
which contains the full 50-trial output from the main pipeline.

Outputs:
  - results/exp5_table2_high_precision.csv
  - results/exp5_explanation.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from experiments.reviewer_response.common.results_io import append_csv_row

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "real_world_datasets" / "results"
OUT_DIR = Path(__file__).parent / "results"

DATASETS = [
    ("football", 10),
    ("football", 100),
    ("basketball", 10),
    ("basketball", 100),
    ("baseball", 10),
    ("baseball", 100),
    ("sushi", 10),
    ("movie_lens", 10),
    ("movie_lens", 50),
    ("movie_lens", 100),
    ("news", 10),
]
MODELS = ["our", "L1", "L2", "tau", "pl", "pl_reg"]
METRICS = ["Kendall_tau", "Spearman_rho", "@precision_1", "@precision_5", "@precision_10"]


def load_trials(dataset: str, n: int) -> List[Dict]:
    path = RESULTS_DIR / f"{dataset}_n_teams={n}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())["trials"]


def main():
    out_csv = OUT_DIR / "exp5_table2_high_precision.csv"
    if out_csv.exists():
        out_csv.unlink()

    fieldnames = [
        "dataset", "n", "model",
        "n_trials",
        "alpha_mean", "alpha_sd",
        "beta_mean", "beta_sd",
    ] + [f"{m}_mean" for m in METRICS] + [f"{m}_sd" for m in METRICS]

    notes = []
    summary_table: Dict[tuple, Dict] = {}

    for ds, n in DATASETS:
        trials = load_trials(ds, n)
        if not trials:
            notes.append(f"- {ds} n={n}: results JSON missing")
            continue
        for model in MODELS:
            try:
                metric_arrs = {
                    m: np.array([tr["models"][model]["metrics"][m] for tr in trials])
                    for m in METRICS
                }
            except KeyError:
                continue
            args0 = trials[0]["models"][model].get("args", {})
            if "alpha" in args0:
                alpha_arr = np.array(
                    [tr["models"][model]["args"].get("alpha", np.nan) for tr in trials]
                )
                beta_arr = np.array(
                    [tr["models"][model]["args"].get("beta", np.nan) for tr in trials]
                )
                alpha_mean = float(np.nanmean(alpha_arr))
                alpha_sd = float(np.nanstd(alpha_arr))
                beta_mean = float(np.nanmean(beta_arr))
                beta_sd = float(np.nanstd(beta_arr))
            else:
                alpha_mean = alpha_sd = beta_mean = beta_sd = float("nan")

            row = {
                "dataset": ds, "n": n, "model": model,
                "n_trials": len(trials),
                "alpha_mean": f"{alpha_mean:.8f}",
                "alpha_sd": f"{alpha_sd:.8f}",
                "beta_mean": f"{beta_mean:.8f}",
                "beta_sd": f"{beta_sd:.8f}",
            }
            for m in METRICS:
                row[f"{m}_mean"] = f"{metric_arrs[m].mean():.8f}"
                row[f"{m}_sd"] = f"{metric_arrs[m].std():.8f}"
            append_csv_row(out_csv, row, fieldnames)
            summary_table[(ds, n, model)] = row

    # Cross-dataset duplicate check on baseball/basketball n=100
    note_lines = ["# Experiment 5 -- Table 2 high-precision audit\n"]
    note_lines.append("Re-extracted from the cached 50-trial JSON output of the main pipeline.\n")
    note_lines.append("All numbers reported below are means over 50 trials, eight decimal places.\n")

    for ds in ("baseball", "basketball"):
        rows = [summary_table.get((ds, 100, m)) for m in ("our", "L1", "L2")]
        if all(rows):
            note_lines.append(f"\n## {ds} n=100\n")
            for r in rows:
                note_lines.append(
                    f"- {r['model']:5s}: Kendall_tau = {r['Kendall_tau_mean']}  "
                    f"Spearman_rho = {r['Spearman_rho_mean']}  "
                    f"alpha = {r['alpha_mean']}\n"
                )

    # Cross-file duplicate check
    same = []
    for m in ("our", "L1", "L2", "tau"):
        bb = summary_table.get(("baseball", 100, m))
        bk = summary_table.get(("basketball", 100, m))
        if bb and bk:
            same_metric = bb["Kendall_tau_mean"] == bk["Kendall_tau_mean"]
            same.append((m, same_metric, bb["Kendall_tau_mean"], bk["Kendall_tau_mean"]))

    note_lines.append("\n## Cross-file duplicate audit (Baseball n=100 vs Basketball n=100)\n")
    note_lines.append(
        "If `our`/`L1`/`L2`/`tau` numbers are identical across the two files, "
        "this indicates the basketball file inherited rows from the baseball "
        "run rather than a precision artefact of Table 2.\n\n"
    )
    for m, eq, v1, v2 in same:
        flag = "IDENTICAL" if eq else "different"
        note_lines.append(f"- {m:4s}: baseball={v1} basketball={v2}  --> **{flag}**\n")

    note_lines.append(
        "\n## Recommendation\n\n"
        "If the duplicate is genuine (alpha hat hits its lower bound, see Exp 3, "
        "and the consensus ranking is the same Borda center), a footnote in "
        "Table 2 should explain the collapse.  If the duplicate is a "
        "copy-paste artefact, regenerate the affected row with the n=100 "
        "Basketball pipeline and re-run.\n"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "exp5_explanation.md").write_text("".join(note_lines))
    print(f"wrote {out_csv}")
    print(f"wrote {OUT_DIR / 'exp5_explanation.md'}")


if __name__ == "__main__":
    main()
