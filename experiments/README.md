# Experiments

Extended evaluation suite for the L_alpha Mallows model, complementing
the primary benchmarks in `real_world_datasets/`. Each script is
self-contained: it re-uses the fit routines and dataset loaders from the
main codebase and writes its outputs under `results/` and `figures/`.

## Directory layout

```
experiments/
├── run_all.py                           top-level runner (executes every script in order)
├── check_status.py                      row-count progress checker
├── common/                              shared helpers
│   ├── datasets.py                      dataset loaders with graceful skip-on-unavailable
│   ├── distances.py                     d_alpha, MCMC, R-hat, ESS, bridge sampling, TI
│   ├── kemeny.py                        exact Kemeny ILP + local-search heuristic
│   ├── loglik.py                        log-likelihood for L_alpha / Mallows-tau / PL
│   └── results_io.py                    resumable CSV IO (row-append + key-dedup)
├── aggregate_metrics_audit.py           re-aggregate per-dataset JSON at 8 decimal places
├── alpha_bound_sensitivity.py           fit L_alpha with varying alpha_min
├── alpha_gain_analysis.py               L_alpha vs fixed-alpha gap across synthetic and real
├── held_out_log_likelihood.py           held-out log P(pi) for all five competing models
├── kemeny_center_comparison.py          exact Kemeny vs Borda center for Mallows-tau
├── partition_function_diagnostics.py    score-equation and Z_n accuracy, MCMC mixing
├── results/                             CSVs produced by each script
└── figures/                             PDFs produced by each script
```

## Running

Full run (50 trials across every dataset spec the environment can load):

```
python -m experiments.run_all --n-trials 50
```

Single script:

```
python -m experiments.held_out_log_likelihood --n-trials 50
python -m experiments.partition_function_diagnostics
```

Subset through the runner:

```
python -m experiments.run_all --only held_out_log_likelihood,kemeny_center_comparison
```

Reduced grid (useful for validating the pipeline end-to-end in a few minutes):

```
python -m experiments.run_all --quick
```

All scripts append results row-by-row, deduplicating on the natural key
(dataset, sample size, trial index, model, ...). A failed or
interrupted run can be resumed by re-invoking the same command.

## Scripts

### `aggregate_metrics_audit.py`

Reloads the cached 50-trial JSON outputs under
`real_world_datasets/results/*.json` and reports every metric mean / sd
at eight decimal places. Also performs a cross-dataset identity check:
two independent datasets producing bit-identical means across 50 trials
indicate a data-integrity issue rather than a rounding artefact.

Outputs: `results/aggregate_metrics_high_precision.csv`,
`results/aggregate_metrics_notes.md`.

### `kemeny_center_comparison.py`

Refits the Mallows-tau baseline with an exact Kemeny consensus center
(ILP, via PuLP + CBC) on n=10 datasets, and with a greedy adjacent-swap
local-search heuristic on n=100 datasets. Reports per-trial Kendall tau,
Spearman rho, Top-1 hit rate, held-out log-likelihood, and the Kemeny
objective value, alongside the Borda-centered baseline.

Outputs: `results/kemeny_center_comparison.csv`,
`results/kemeny_center_comparison_summary.csv`.

### `held_out_log_likelihood.py`

Fits L_alpha Mallows (learned alpha), L1-Mallows (alpha = 1), L2-Mallows
(alpha = 2), Mallows-tau, and Plackett-Luce on the training split and
evaluates mean log P(pi) on the held-out split. Partition-function
handling: exact closed form for Mallows-tau and Plackett-Luce; banded DP
with truncation chosen so the empirical relative gap to delta+1 is
<= 1e-4 for L_alpha (alpha >= 1); thermodynamic integration with five
chains for L_alpha (alpha < 1).

Outputs: `results/held_out_log_likelihood.csv`,
`results/held_out_log_likelihood_summary.csv`,
`figures/held_out_log_likelihood_summary.pdf`.

### `alpha_bound_sensitivity.py`

Refits L_alpha Mallows with `alpha_min` in {0.01, 0.05, 0.1, 0.2, 0.5}
on each n=10 real-world dataset and reports the resulting `alpha_hat`,
`beta_hat`, Kendall tau, Top-1 hit rate, and held-out log-likelihood.
The module docstring documents the default bound in
`MLE.alpha_beta_estimation.solve_alpha_beta`.

Outputs: `results/alpha_bound_sensitivity.csv`,
`results/alpha_bound_sensitivity_summary.csv`,
`figures/alpha_bound_sensitivity.pdf`.

### `partition_function_diagnostics.py`

Three parts, over grid n in {6, 8, 10}, alpha in {0.1, 0.3, 0.5, 0.8},
beta in {0.1, 0.5, 1.0, 2.0}:

* **A.** Brute-force enumeration vs MCMC vs banded-DP for E[d_alpha]
  and E[d_dot_alpha] (the score-equation expectations).
* **B.** Five-chain Metropolis-Hastings diagnostics: Gelman-Rubin R-hat,
  Geyer effective sample size, plus trace plots at selected (alpha, beta)
  cells.
* **C.** Partition-function approximation error for alpha < 1: exact Z_n
  via enumeration vs bridge sampling vs thermodynamic integration.

Outputs: `results/score_equation_accuracy.csv`,
`results/mcmc_mixing_diagnostics.csv`,
`results/partition_function_error_small_alpha.csv`,
`figures/mcmc_traceplots.pdf`.

### `alpha_gain_analysis.py`

Quantifies how much learning alpha buys over fixed-alpha baselines as a
function of how far the data-generating alpha sits from 1. Part A is a
synthetic sweep over alpha_0 in {0.2, 0.5, 1.0, 1.5, 2.0, 3.0}; Part B
reads the output of `held_out_log_likelihood.py` and plots the real-data
gap of L_alpha over L1 against `|alpha_hat - 1|`.

Outputs: `results/alpha_gain_synthetic_sweep.csv`,
`results/alpha_gain_real_scatter.csv`,
`figures/alpha_gain_scatter.pdf`.

## Datasets

`common/datasets.py` wraps the main pipeline's loaders with
`DatasetUnavailable`, so scripts skip datasets whose external sources are
unreachable (e.g. Kaggle credentials missing) rather than crashing. The
default dataset list is:

| Dataset | n |
| --- | --- |
| football | 10, 100 |
| basketball | 10, 100 |
| baseball | 10, 100 |
| sushi | 10 |
| movie_lens | 10, 50, 100 |
| news | 10 |

The sports datasets use a chronological train/test split; the rest use
a random 70/30 split, matching the conventions in `real_world_datasets`.

## Dependencies

See the top-level `requirements.txt`. Additionally, `pulp` is required
for the exact Kemeny ILP in `kemeny_center_comparison.py`. Install via:

```
pip install -r requirements.txt
pip install pulp
```
