import json, os, sys
from datetime import datetime
from functools   import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np


from GMM_diagonalized.sampling import sample_truncated_mallow
from MLE.consensus_ranking_estimation import consensus_ranking_estimation
from MLE.alpha_beta_estimation import solve_alpha_beta


# ---------------------------------------------------------------------
# Worker – MUST be top-level picklable for Windows / ‘spawn’ start-method
# ---------------------------------------------------------------------
def _single_trial(num_samples, trial,
                  *, n, Delta, sigma_0, beta_0, alpha_0):
    """Run one (num_samples, trial) experiment and return a dict."""
    # --- sampling -----------------------------------------------------
    train_samples = sample_truncated_mallow(num_samples=num_samples,
                                            n=n, beta=beta_0, alpha=alpha_0,
                                            sigma=sigma_0, Delta=Delta,
                                            rng_seed=num_samples * 997 + trial)

    # --- estimation ---------------------------------------------------
    consensus = consensus_ranking_estimation(train_samples)
    alpha_hat, beta_hat = solve_alpha_beta(train_samples, consensus)

    return {
        "num_samples"       : num_samples,
        "trial_number"      : trial,
        "alpha"             : float(alpha_hat),
        "beta"              : float(beta_hat),
        "consensus_ranking" : consensus.tolist()
    }


# ---------------------------------------------------------------------
# Main driver ---------------------------------------------------------
# ---------------------------------------------------------------------
def save_synthetic_data(filename=None,
                        *,
                        n          = 10,
                        Delta      = 5,
                        sigma_0    = None,
                        beta_0     = 0.5,
                        alpha_0    = 1.5,
                        num_train_samples = np.arange(20, 500, 5),
                        n_trials   = 2,
                        max_workers= None):          # None ⇒ os.cpu_count()
    """Runs all experiments in parallel and logs results to a JSON file."""
    print('****************  synthetic script running  ****************')

    # --- default sigma_0 ---------------------------------------------
    if sigma_0 is None:
        sigma_0 = 1 + np.arange(n)

    # --- create / choose results file --------------------------------
    log_dir = "synthethic_tests/log"
    os.makedirs(log_dir, exist_ok=True)

    if filename is None:
        base = f"estimation_{alpha_0}_{beta_0}_{n}"
        filename = os.path.join(log_dir, f"{base}.json")
        k = 1
        while os.path.exists(filename):
            filename = os.path.join(log_dir, f"{base}_{k}.json"); k += 1

    # --- initialise JSON scaffold ------------------------------------
    results = {
        "parameters": dict(
            n=n, Delta=Delta, sigma_0=sigma_0.tolist(),
            beta_0=beta_0, alpha_0=alpha_0, n_trials=n_trials,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ),
        "trials": {}            # will become { "20": [ … ], "25": [ … ] … }
    }
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Initial header written to {filename}")

    # -----------------------------------------------------------------
    # Submit every (num_samples, trial) to the process pool
    # -----------------------------------------------------------------
    worker = partial(_single_trial,
                     n=n, Delta=Delta, sigma_0=sigma_0,
                     beta_0=beta_0, alpha_0=alpha_0)

    tasks = [(ns, tr) for ns in num_train_samples for tr in range(n_trials)]
    n_tasks = len(tasks)

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(worker, ns, tr): (ns, tr) for ns, tr in tasks}

        for idx, fut in enumerate(as_completed(futures), 1):
            trial_data = fut.result()          # raises here if worker failed
            ns = str(trial_data.pop("num_samples"))

            # append & write under the main process (file is safe)
            results["trials"].setdefault(ns, []).append(trial_data)
            with open(filename, "w") as f:
                json.dump(results, f, indent=2)

            done = futures[fut]
            print(f"[{idx}/{n_tasks}]  finished  n={done[0]}  trial={done[1]}")

    print(f"\nAll results saved to {filename}\nDone.")
