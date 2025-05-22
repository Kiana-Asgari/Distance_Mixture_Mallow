import json, os
from datetime                               import datetime
from functools                              import partial
from concurrent.futures                     import ProcessPoolExecutor, as_completed

import numpy as np

from GMM_diagonalized.sampling             import sample_truncated_mallow
from MLE.consensus_ranking_estimation      import consensus_ranking_estimation
from MLE.alpha_beta_estimation             import solve_alpha_beta

from synthethic_tests.plot                import plot_boxplot, plot_vs_n





# Single worker (runs in its own process – keep **quiet**!)
# ---------------------------------------------------------------------
def _single_trial(num_samples, trial,
                  *, n, Delta, Delta_data, sigma_0, beta_0, alpha_0):
    """
    Run exactly one experiment.  All messages are returned to the parent
    so that printing order is controlled centrally.
    """
    try:
        # ---------- sampling ----------
        train = sample_truncated_mallow(num_samples=num_samples,
                                        n=n, beta=beta_0, alpha=alpha_0,
                                        sigma=sigma_0, Delta=Delta_data,
                                        rng_seed=trial+42)

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
        Delta_data       = 7,
        sigma_0          = None,
        beta_0           = 0.5,
        alpha_0          = 1.5,
        num_train_samples= np.arange(15, 350, 5),
        n_trials         = 50,
        max_workers      = 16,
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
                     Delta_data=Delta_data,
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
    return results


def test_effect_of_n(n_training = 50, alpha_0=1, beta_0=1, n_trials=4, Delta=4,
                              save=False, verbose=True):
    n_items_list = [10,20,30,40,50]
    alpha_error_list = []
    beta_error_list = []
    print('starting to test the effect of n')
    for n_items in n_items_list:
        print('testing n = ', n_items)
        results = learn_synthetic_data(n=n_items, 
                                    Delta=Delta, 
                                    alpha_0=alpha_0, 
                                    beta_0=beta_0, 
                                    n_trials=n_trials,
                                    num_train_samples=[n_training],
                                    save=save,
                                    verbose=verbose)
        trial_alpha_abs_error = []
        trial_beta_abs_error = []
        for i in range(n_trials):
            trial_alpha_abs_error.append(np.abs(np.array(results["trials"][str(n_training)][i]["alpha"]) - alpha_0))
            trial_beta_abs_error.append(np.abs(np.array(results["trials"][str(n_training)][i]["beta"]) - beta_0))
            
        alpha_error_list.append(trial_alpha_abs_error)
        beta_error_list.append(trial_beta_abs_error)
    plot_vs_n(alpha_error_list, beta_error_list, n_items_list)
    return alpha_error_list, beta_error_list


def test_effect_of_truncation(n_training = 50, 
                              n_items=15, alpha_0=1, beta_0=1, n_trials=25,
                              save=False, verbose=True):
    Delta_list = [1,2,3,4,5,6]
    alpha_error_list = []
    beta_error_list = []
    print('starting to test the effect of truncation with n_items = ', n_items, 'and alpha_0 = ', alpha_0, 'and beta_0 = ', beta_0)
    for Delta in Delta_list:
        print('testing truncation with Delta = ', Delta)
        results = learn_synthetic_data(n=n_items, 
                                    Delta=Delta, 
                                    alpha_0=alpha_0, 
                                    beta_0=beta_0, 
                                    n_trials=n_trials,
                                    num_train_samples=[n_training],
                                    save=save,
                                    verbose=verbose)
        trial_alpha_abs_error = []
        trial_beta_abs_error = []
        for i in range(n_trials):
            trial_alpha_abs_error.append(np.abs(np.array(results["trials"][str(n_training)][i]["alpha"]) - alpha_0))
            trial_beta_abs_error.append(np.abs(np.array(results["trials"][str(n_training)][i]["beta"]) - beta_0))
            
        alpha_error_list.append(trial_alpha_abs_error)
        beta_error_list.append(trial_beta_abs_error)
    plot_boxplot(alpha_error_list, beta_error_list, Delta_list)
    return alpha_error_list, beta_error_list
