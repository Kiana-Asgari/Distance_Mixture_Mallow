import concurrent.futures
import csv
from GMM_diagonalized.partial_partition_estimation import marginal_probabilities
from utils import distance_alpha_batch
from GMM_diagonalized.DP_partition_estimation import *
import numpy as np
from scipy.optimize import root
from scipy.optimize import bisect
import numpy as np
from itertools import combinations
import random
import time
from utils import get_support_and_tk
def get_partition_estimate_via_dp(beta, alpha, n, Delta):
    
    T, t_vals = get_support_and_tk(beta, alpha, Delta)

    # 1) Construct states
    states, p1, p_ell, length = construct_states(T)
    M = len(states)

    # 2) Build adjacency
    adj_forward, adj_reverse = build_adjacency_lists(T, t_vals, states)

    # 3) Let's pick the start_idx = 0 as 'initial state' for DP
    start_idx = 0

    # 4) Run forward DP for 'n' steps
    now= time.time()
    DP = run_dp_forward(adj_forward, n, len(states)-1, M)
    #samples = sample_path_from_dp(DP, adj_reverse, n)
    #print('DP:', DP[n,-1])
    return DP[n,-1]

def costly_function(alpha, beta):
    """
    Replace this with the actual heavy-lifting function you want to run 
    for each (alpha, beta). This is just a placeholder example.
    """
    # Some "fake" CPU-bound work:
    Delta= 10
    n = 111
    T, t_vals = get_support_and_tk(beta, alpha, Delta)

    # 1) Construct states
    states, p1, p_ell, length = construct_states(T)
    M = len(states)

    # 2) Build adjacency
    adj_forward, adj_reverse = build_adjacency_lists(T, t_vals, states)

    # 3) Let's pick the start_idx = 0 as 'initial state' for DP
    start_idx = 0

    # 4) Run forward DP for 'n' steps
    DP = run_dp_forward(adj_forward, n, len(states)-1, M)
    DP[:,-1]
    return (alpha, beta, DP[:,-1])

def main():
    print('running tests')
    # 1. Define the range of alpha and beta values we want to compute.
    alpha_values = np.linspace(1,4, 31)
    
    arr1 = np.linspace(2, 1e-2, int((1 - 1e-2) * 50))
    arr2 = np.linspace(2, 1e-2, int((1 - 1e-2) * 50)) + 1e-3
    # Merge the arrays
    merged_arr = np.concatenate((arr1, arr2))
    beta_values = merged_arr
    
    # 2. Prepare a list of (alpha, beta) tasks.
    tasks = []
    for alpha in alpha_values:
        for beta in beta_values:
            tasks.append((alpha, beta))
    
    # 3. Use a ThreadPoolExecutor with up to 8 workers (for 8 cores).
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # Submit each task to the thread pool.
        future_to_task = {
            executor.submit(costly_function, alpha, beta): (alpha, beta)
            for (alpha, beta) in tasks
        }
        
        # As each future finishes, store its result.
        for future in concurrent.futures.as_completed(future_to_task):
            alpha, beta = future_to_task[future]
            try:
                alpha_out, beta_out, value = future.result()
                results.append((alpha_out, beta_out, value))
            except Exception as exc:
                print(f"Task (alpha={alpha}, beta={beta}) generated an exception: {exc}")
    
    # 4. Write all results to a CSV file (e.g. 'results.csv').
    with open("results.csv", mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["alpha", "beta", "computed_value"])  # header
        writer.writerows(results)
    # computed value is the list of permanents, i-th value is permanent when  the number of teams is i
    print("Done! Results saved to results.csv")



   