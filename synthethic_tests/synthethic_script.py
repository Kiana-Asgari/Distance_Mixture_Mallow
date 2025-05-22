# -*- coding: utf-8 -*-
"""
Synthetic-data study:
    • sample from a truncated Mallow model
    • estimate consensus + (α,β)
    • log results in JSON   (one file per parameter grid)

All multiprocessing output is funnelled back to the main process so that
log lines appear in the natural sequence:
      n_samples = n₁
        trial 0 ...
        trial 1 ...
      n_samples = n₂
        …
"""

import json, os
from datetime import datetime
from functools   import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from GMM_diagonalized.sampling             import sample_truncated_mallow
from MLE.consensus_ranking_estimation      import consensus_ranking_estimation
from MLE.alpha_beta_estimation             import solve_alpha_beta


# ---------------------------------------------------------------------
# Single worker (runs in its own process – keep **quiet**!)
# ---------------------------------------------------------------------
def _single_trial(num_samples, trial,
                  *, n, Delta, sigma_0, beta_0, alpha_0):
    """
    Run exactly one experiment.  All messages are returned to the parent
    so that printing order is controlled centrally.
    """
    try:
        # ---------- sampling ----------
        train = sample_truncated_mallow(num_samples=num_samples,
                                        n=n, beta=beta_0, alpha=alpha_0,
                                        sigma=sigma_0, Delta=Delta,
                                        rng_seed=trial)

        # ---------- estimation ----------
        consensus        = consensus_ranking_estimation(train)
        alpha_hat, beta_hat = solve_alpha_beta(train, consensus, Delta=Delta)

        return dict(num_samples   = int(num_samples),
                    trial_number  = int(trial),
                    alpha         = float(alpha_hat),
                    beta          = float(beta_hat),
                    consensus_ranking = consensus.tolist())

    except Exception as err:          # propagate the error details
        return dict(num_samples  = int(num_samples),
                    trial_number = int(trial),
                    error        = str(err))


# ---------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------
def learn_synthetic_data(*,
        filename         = None,
        n                = 15,
        Delta            = 6,
        sigma_0          = None,
        beta_0           = 0.5,
        alpha_0          = 1.5,
        num_train_samples= np.arange(15, 350, 5),
        n_trials         = 50,
        max_workers      = 4,
        save             = True, 
        verbose          = True):
    if verbose:
        print(f"\n ============================ Running synthetic test with n={n}, truncation={Delta}, alpha_0={alpha_0}, beta_0={beta_0} ============================  ")

    # ------- default σ₀ -------
    if sigma_0 is None:
        sigma_0 = 1 + np.arange(n)

    # ------- output file -------
    log_dir = "synthethic_tests/log"
    os.makedirs(log_dir, exist_ok=True)

    if save and filename is None:
        filename = os.path.join(
            log_dir,
            f"estimation_n{n}_Δ{Delta}_α{alpha_0}_β{beta_0}.json"
        )

    # ------- (re-)initialise results -------
    if save and os.path.exists(filename):
        with open(filename) as f:
            results = json.load(f)
        results["parameters"]["last_updated"] = datetime.now().isoformat()
        print(f"[INFO]  Appending to existing log → {filename}")
    else:
        results = dict(parameters = dict(
                            n         = n,
                            Delta     = Delta,
                            sigma_0   = sigma_0.tolist(),
                            beta_0    = beta_0,
                            alpha_0   = alpha_0,
                            n_trials  = n_trials,
                            timestamp = datetime.now().isoformat()),
                       trials = {})
        if save:
            with open(filename, "w") as f: json.dump(results, f, indent=2)

    # -----------------------------------------------------------------
    # Multiprocessing pool (up to 4 workers)
    # -----------------------------------------------------------------
    max_workers = max(1, min(int(max_workers), 4))
    worker = partial(_single_trial,
                     n=n, Delta=Delta, sigma_0=sigma_0,
                     beta_0=beta_0, alpha_0=alpha_0)

    with ProcessPoolExecutor(max_workers=max_workers) as pool:

        # ----- loop over each training-set size in sequence -----
        for ns in num_train_samples:
            print(f"\n===== Training size: {ns} ( {n_trials} trials ) =====")

            # launch all trials in parallel
            futs = [ pool.submit(worker, ns, tr) for tr in range(n_trials) ]

            # collect as they finish
            bucket = [None] * n_trials
            for done_cnt, fut in enumerate(as_completed(futs), 1):
                res = fut.result()

                if "error" in res:
                    print(f"  !! trial {res['trial_number']} failed: {res['error']}")
                    continue

                t = res["trial_number"]
                bucket[t] = res
                print(f"  [ trial {done_cnt:2d}/{n_trials} ]  "
                      f"  α̂ = {res['alpha']:.3f}  β̂ = {res['beta']:.3f}")

            # keep only successful trials, in order
            bucket = [r for r in bucket if r is not None]
            results["trials"][str(ns)] = bucket

            if save:
                with open(filename, "w") as f: json.dump(results, f, indent=2)
                print(f"  ↳ saved {len(bucket)} trials for n_samples={ns}")
            alphas = np.array([r['alpha'] for r in bucket])
            betas = np.array([r['beta'] for r in bucket])
            print(f"        |α - α_0| = {np.mean(np.abs(alphas - alpha_0)):.3f} ± {np.std(np.abs(alphas - alpha_0)):.3f}")
            print(f"        |β - β_0| = {np.mean(np.abs(betas - beta_0)):.3f} ± {np.std(np.abs(betas - beta_0)):.3f}")
    print("\n[INFO]  All experiments finished.")
    if save: print(f"[INFO]  Full log written to  {filename}")