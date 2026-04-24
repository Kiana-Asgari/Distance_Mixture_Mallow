# Reviewer Response Experiments -- Summary

This directory contains the six reviewer-response experiments described in
the response plan. All scripts share the helpers in `common/`:

* `common/distances.py` -- d_alpha, d_alpha_dot, exact enumeration over S_n,
  Metropolis-Hastings sampler with adjacent-transposition proposals,
  bridge-sampling and thermodynamic-integration estimators for log Z,
  Gelman-Rubin R-hat, and Geyer ESS.
* `common/loglik.py` -- log-likelihood routines for Mallows-tau (closed
  form), Plackett-Luce (closed form), L-alpha Mallows with alpha >= 1
  (banded DP) and L-alpha Mallows with alpha < 1 (TI / bridge sampling).
* `common/kemeny.py` -- Kemeny consensus via PuLP / CBC ILP (exact, n<=10)
  and a greedy adjacent-swap local-search heuristic (n=100).
* `common/datasets.py` -- thin wrappers around the existing dataset
  loaders that gracefully skip datasets the environment cannot load.

## How to run everything

```
# smoke test (a few minutes)
python -m experiments.reviewer_response.run_all --quick

# full reviewer-response run (50 trials, all datasets)
python -m experiments.reviewer_response.run_all --n-trials 50
```

Each script also runs standalone with its own `--quick` flag. All CSVs are
appended row-by-row so a failed run can be resumed by re-invoking the
script.

> **Environment note:** the College Sports datasets (football, basketball,
> baseball) are pulled from Kaggle and require Kaggle API credentials.
> Sushi (n=10), News (n=10), and MovieLens (n=10/50/100) work with only
> public network access. All scripts skip datasets that cannot be loaded
> rather than crashing.

> **Checked-in artefacts:** the CSVs and PDFs currently under `results/`
> and `figures/` are outputs from `run_all.py --quick` (2 trials per
> dataset, reduced grids, MCMC budget of ~2k samples). They verify that
> every script runs end-to-end and writes the expected file format; they
> are **not** the final 50-trial results. Re-run `python -m
> experiments.reviewer_response.run_all --n-trials 50` with Kaggle
> credentials to produce the full numbers for the response. Only Exp 5
> reads from the already-cached `real_world_datasets/results/*.json`, so
> its numbers are final (50-trial) in both modes.

---

## Experiment 1 -- Held-out approximate log-likelihood

* Adds a likelihood-aligned metric to complement the existing Kendall-tau
  and Top-1 hit-rate evaluations.
* For LDER with alpha >= 1 (and L1, L2), log Z is computed by the
  banded-permanent DP with the truncation Delta picked so the empirical
  relative gap to Delta+1 is <= 1e-4.
* For LDER with alpha < 1 (the regime not covered by Theorem 4.1), log Z
  is estimated by thermodynamic integration (TI), 16 grid points, with
  five chains for the standard error.

Smoke-test highlights (sushi n=10, news n=10, 2 trials each):

| dataset | model | mean log lik | log-Z method |
| --- | --- | --- | --- |
| news n=10 | our (LDER) | -9.7 | TI (alpha approx 0.13) |
| news n=10 | L1 | -11.5 | exact DP |
| sushi n=10 | our (LDER) | -15.0 | TI (alpha approx 0.25) |
| sushi n=10 | L1 | -15.0 | exact DP |

LDER beats L1 by approx 1.8 log-units on news, ties on sushi -- consistent
with the alpha_hat observations in Exp 6.

* Outputs: `results/exp1_loglik.csv`, `results/exp1_loglik_summary.csv`,
  `figures/exp1_loglik_summary.pdf`.

---

## Experiment 2 -- alpha < 1 diagnostics

Three parts, each parametrised over n in {6, 8, 10}, alpha in
{0.1, 0.3, 0.5, 0.8}, beta in {0.1, 0.5, 1.0, 2.0}:

* **Part A** -- Compares E[d_alpha] and E[d_dot_alpha] under three
  estimators: brute-force enumeration (ground truth), MH (5x10n^2
  samples), and the current implementation's banded-DP-based estimate.
  With the recommended budget the MH estimator is within a few percent of
  exact; the banded-DP estimator is essentially exact for n <= 10
  (truncation error vanishes once Delta = n - 1).
* **Part B** -- Five-chain MH diagnostics: R-hat for d_alpha and
  d_dot_alpha, ESS, plus trace plots at (n=10, alpha in {0.1, 0.5},
  beta in {0.5, 2.0}). Trace plots are saved at
  `figures/exp2b_traceplots.pdf`.
* **Part C** -- Z_n approximation error table extending Table 4 to
  alpha < 1, beta = 2. Reports both bridge-sampling and TI estimates;
  TI is consistently 1-2 orders of magnitude tighter at large beta.

Outputs: `results/exp2a_score_accuracy.csv`,
`results/exp2b_mcmc_diagnostics.csv`,
`results/exp2c_Z_approx_alpha_lt1.csv`,
`figures/exp2b_traceplots.pdf`.

---

## Experiment 3 -- alpha lower-bound sensitivity

* `MLE/alpha_beta_estimation.py:solve_alpha_beta` enforces
  `alpha_bounds=(1e-1, 3)` for both the least-squares (n <= 20) and
  differential-evolution (n > 20) paths. The shipped lookup tables also
  start at alpha=0.1, so the n=100 pipeline cannot explore lower alphas
  without recomputing the tables. This explains baseball's reported
  alpha_hat = 0.100 +/- 0.000 in the main paper -- the optimiser hits the
  bound rather than finding a true MLE there.
* For each n=10 dataset and each alpha_min in {0.01, 0.05, 0.1, 0.2, 0.5},
  the script refits LDER, then reports alpha_hat, beta_hat, Kendall tau,
  Top-1 hit rate, and held-out log-likelihood (using the same TI/DP
  estimator as Exp 1).

Smoke-test pattern (news, n=10): for alpha_min <= 0.13 the MLE is
unconstrained at alpha_hat ~ 0.13; for alpha_min in {0.2, 0.5} the MLE
hits the bound and held-out log-likelihood degrades.

Outputs: `results/exp3_alpha_floor.csv`,
`results/exp3_alpha_floor_summary.csv`,
`figures/exp3_sensitivity.pdf`.

---

## Experiment 4 -- Stronger Mallows-tau baseline (exact Kemeny)

* For n=10 datasets the consensus center is solved exactly via an ILP
  (PuLP + CBC) on the precedence variables x_ij (binary, antisymmetric,
  transitive). Verified on n=5 by brute-force enumeration of S_5
  (matching objective value).
* For n=100 datasets a greedy adjacent-swap local search starting from
  Borda is used instead, and is reported as a sensitivity check rather
  than exact.
* Refits the Mallows-tau theta with the new center and recomputes Kendall
  tau, Spearman rho, Top-1 hit rate, and held-out log-likelihood.

Outputs: `results/exp4_mallows_tau_exact.csv`,
`results/exp4_comparison.csv`.

---

## Experiment 5 -- Table 2 high-precision audit

Reads the cached 50-trial JSON outputs in `real_world_datasets/results/`
and prints all metric means / sds at eight decimal places.

**Finding (smoke run, taken straight from the cache):** for the
n=100 baseball and basketball results, the `our`, `L1`, `L2`, and `tau`
rows are *bit-identical* across the two datasets, while `pl` and `pl_reg`
differ. This means the duplication in Table 2 is not a precision-rounding
issue -- the basketball file inherited four model rows from the baseball
run and needs to be regenerated. The script writes the audit conclusion
to `results/exp5_explanation.md`.

Additionally: alpha_hat for `our` is 0.10000015 across 50 trials with
sd 1e-7 in both files, confirming Exp 3's premise that the optimiser is
hitting `alpha_bounds=(1e-1, 3)`.

Outputs: `results/exp5_table2_high_precision.csv`,
`results/exp5_explanation.md`.

---

## Experiment 6 (optional) -- When does learning alpha help?

* **Part A** synthetic sweep: alpha_0 in {0.2, 0.5, 1.0, 1.5, 2.0, 3.0},
  beta_0 = 0.5, n = 10, m = 500. For each, fit LDER, L1, L2 and report
  the held-out log-likelihood gap of LDER over the best fixed-alpha
  competitor.
* **Part B** real-data scatter: |alpha_hat - 1| (x-axis) versus
  log-likelihood gain of LDER over L1 (y-axis), for the n=10 real
  datasets. Reads Exp 1's CSV and produces `figures/exp6_scatter.pdf`.

Smoke-test pattern (synthetic, m=100): the gap is small or negative when
alpha_0 is close to 1 (matching the L1 baseline) and grows when alpha_0
is far from 1 -- precisely the message reviewer C asked us to quantify.

Outputs: `results/exp6_synthetic_sweep.csv`,
`results/exp6_real_scatter.csv`, `figures/exp6_scatter.pdf`.

---

## Mapping reviewer comments -> experiments

| Reviewer comment | Addressed by |
| --- | --- |
| Reviewer B and C: Kendall tau / Top-1 are not likelihood metrics | Exp 1 |
| Reviewer C: theory does not cover alpha < 1; how reliable are the score equations / partition function in that regime? | Exp 2 (Parts A, B, C) |
| Reviewer A: baseball alpha_hat = 0.100 +/- 0.000 looks like a hard floor; disclose | Exp 3 (and the script header documents the floor explicitly) |
| Reviewer C: Mallows-tau uses Borda, which is a weak approximate Kemeny center | Exp 4 |
| Reviewer C: Table 2 entries appear duplicated for basketball/baseball n=100 | Exp 5 |
| Reviewer C: "what is the real disadvantage of blindly picking alpha = 1?" | Exp 6 |
