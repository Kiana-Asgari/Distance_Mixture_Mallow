import json, os, sys
from datetime import datetime
from functools   import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np


from GMM_diagonalized.sampling import sample_truncated_mallow
from MLE.consensus_ranking_estimation import consensus_ranking_estimation
from MLE.alpha_beta_estimation import solve_alpha_beta


# ---------------------------------------------------------------------
# Worker function (MUST be top-level for multiprocessing)
# ---------------------------------------------------------------------
def _single_trial(num_samples, trial, *, n, Delta, sigma_0, beta_0, alpha_0):
    """Run one (num_samples, trial) experiment and return a dict."""
    print(f'running trial {trial} of {num_samples} samples...')
    try:
        # Sampling
        train_samples = sample_truncated_mallow(num_samples=num_samples,
                                                n=n, beta=beta_0, alpha=alpha_0,
                                                sigma=sigma_0, Delta=Delta,
                                                rng_seed=num_samples * 997 + trial)

        # Estimation
        consensus = consensus_ranking_estimation(train_samples)
        alpha_hat, beta_hat = solve_alpha_beta(train_samples, consensus)

        return {
            "num_samples": num_samples,
            "trial_number": trial,
            "alpha": float(alpha_hat),
            "beta": float(beta_hat),
            "consensus_ranking": consensus.tolist()
        }
    except Exception as e:
        print(f"Error in trial {trial} with {num_samples} samples: {e}")
        return None

# ---------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------
def save_synthetic_data(filename=None,
                        *,
                        n=15,
                        Delta=6,
                        sigma_0=None,
                        beta_0=0.5,
                        alpha_0=1.5,
                        num_train_samples=np.arange(20, 480, 5),
                        n_trials=10,
                        max_workers=8):
    """Runs all experiments in parallel and logs results to a JSON file."""
    print('****************  synthetic script running  ****************')

    # --- Set default sigma_0 if not provided ---------------------------
    if sigma_0 is None:
        sigma_0 = 1 + np.arange(n)

    # --- Create or find a suitable file name ---------------------------
    log_dir = "synthethic_tests/log"
    os.makedirs(log_dir, exist_ok=True)

    if filename is None:
        base = f"estimation_{alpha_0}_{beta_0}_{n}"
        filename = os.path.join(log_dir, f"{base}.json")
        k = 1
        while os.path.exists(filename):
            filename = os.path.join(log_dir, f"{base}_{k}.json")
            k += 1

    # --- Initialise JSON structure -------------------------------------
    results = {
        "parameters": {
            "n": n,
            "Delta": Delta,
            "sigma_0": sigma_0.tolist(),
            "beta_0": beta_0,
            "alpha_0": alpha_0,
            "n_trials": n_trials,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "trials": {}
    }
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Initial results structure saved to {filename}")

    # -----------------------------------------------------------------
    # Submit every (num_samples, trial) to the process pool
    # -----------------------------------------------------------------
    worker = partial(_single_trial,
                     n=n, Delta=Delta, sigma_0=sigma_0,
                     beta_0=beta_0, alpha_0=alpha_0)

    tasks = [(ns, tr) for ns in num_train_samples for tr in range(n_trials)]
    n_tasks = len(tasks)

    # --- Ensure max_workers never exceeds 4 ---------------------------
    print(f"Using {max_workers} workers to avoid overload.")
    if max_workers > 8:
        sys.exit('max_workers must be less than 4')

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(worker, ns, tr): (ns, tr) for ns, tr in tasks}

            for idx, fut in enumerate(as_completed(futures), 1):
                try:
                    trial_data = fut.result()
                    if trial_data is None:
                        print(f"[{idx}/{n_tasks}]  Error encountered, skipping.")
                        continue
                    
                    ns = str(trial_data.pop("num_samples"))

                    # Append & write results safely under the main process
                    results["trials"].setdefault(ns, []).append(trial_data)
                    with open(filename, "w") as f:
                        json.dump(results, f, indent=2)

                    done = futures[fut]
                    print(f"[{idx}/{n_tasks}]  finished  n={done[0]}  trial={done[1]}")
                except Exception as e:
                    print(f"Error in future: {e}")

    except KeyboardInterrupt:
        print("Process interrupted! Attempting to clean up...")
        pool.shutdown(wait=False, cancel_futures=True)
        sys.exit()

    print(f"\nAll results saved to {filename}\nDone.")


def read_synthetic_data(filename):
    """
    Reads a synthetic data file and extracts n_samples, alpha values, and beta values.
    
    Args:
        filename (str): Path to the JSON file containing synthetic data
        
    Returns:
        tuple: (n_samples_list, alpha_values, beta_values) where:
            - n_samples_list is a list of sample sizes
            - alpha_values is a dict mapping n_samples to list of alpha values
            - beta_values is a dict mapping n_samples to list of beta values
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Data file not found: {filename}")
    
    with open(filename, "r") as f:
        results = json.load(f)
    
    trials = results.get("trials", {})
    
    # Extract n_samples (as integers)
    n_samples_list = sorted([int(ns) for ns in trials.keys()])
    
    # Extract alpha and beta values for each n_samples
    alpha_values = {}
    beta_values = {}
    
    for ns in trials:
        alpha_values[int(ns)] = [trial["alpha"] for trial in trials[ns]]
        beta_values[int(ns)] = [trial["beta"] for trial in trials[ns]]
    
    return n_samples_list, alpha_values, beta_values
