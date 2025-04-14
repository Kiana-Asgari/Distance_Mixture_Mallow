import random
import math
import numpy as np
from collections import defaultdict



def sample_mallow(sigma, alpha, beta, n_samples=1000, Delta=10):
    n = len(sigma)
    sampled_perms = []
    for iter in range(n_samples):
        np.random.seed(iter)
        random.seed(iter)
        sampled_perm = sample_truncated_mallow(n, beta, alpha, Delta)
        sampled_perm = compose_permutations(sampled_perm, sigma)
        sampled_perms.append(sampled_perm)
    return sampled_perms




def compose_permutations(p, q):
    n = len(p)
    r = np.empty(n, dtype=int)
    for i in range(n):
        r[i] = p[q[i]]
    return r

# sample truncated mallow
#########################################################################
def sample_truncated_mallow(n, beta, alpha, Delta):
    def truncated_A(i, j):
        if j<1 or j>n:
            return 0.0
        dist = abs(i-j)
        if dist>Delta:
            return 0.0
        return math.exp(-beta*(dist**alpha))
    
    graph, DP, states_by_layer = build_truncated_assignment_graph(n, Delta+1, truncated_A)
    sampled_perm = sample_permutation_from_dp(n, Delta+1, graph, DP, states_by_layer)
    return sampled_perm



#########################################################################
def build_truncated_assignment_graph(n, k, A):
    """
    Builds the DP graph for computing the permanent of a banded matrix.

    Returns:
    --------
    graph : list of dicts
        graph[row][state_new] = list of (state_old, col_assigned, edge_weight)

    DP : dict
        DP[(row, state)] = cumulative weight for reaching state at row.

    states_by_layer : list of lists
        List of reachable states at each row.
    """
    def shift_and_append_zero(state):
        return state[1:] + ('0',)

    def flip_bit(state, index):
        state_list = list(state)
        if state_list[index] != '0':
            raise ValueError("Attempting to flip a bit that's already set.")
        state_list[index] = '1'
        return tuple(state_list)

    init_state = ('1',) * k + ('0',) * k
    graph = [defaultdict(list) for _ in range(n + 1)]
    DP = {(0, init_state): 1.0}
    states_by_layer = [[] for _ in range(n + 1)]
    states_by_layer[0].append(init_state)

    for row in range(1, n + 1):
        for prev_state in states_by_layer[row - 1]:
            prev_val = DP.get((row - 1, prev_state), 0.0)
            if prev_val == 0.0:
                continue

            shifted_state = shift_and_append_zero(prev_state)

            # Case 1: leftmost bit is '0', must assign to col = row - k
            if prev_state[0] == '0':
                col = row - k
                if 1 <= col <= n:
                    weight = A(row, col)
                    if weight != 0.0:
                        DP[(row, shifted_state)] = DP.get((row, shifted_state), 0.0) + prev_val * weight
                        graph[row][shifted_state].append((prev_state, col, weight))
                        if shifted_state not in states_by_layer[row]:
                            states_by_layer[row].append(shifted_state)

            # Case 2: flip a '0' bit in the shifted window to assign row
            for i in range(2 * k):
                col = (row - k) + 1 + i
                if not (1 <= col <= n):
                    continue
                if shifted_state[i] == '0':
                    weight = A(row, col)
                    if weight != 0.0:
                        new_state = flip_bit(shifted_state, i)
                        DP[(row, new_state)] = DP.get((row, new_state), 0.0) + prev_val * weight
                        graph[row][new_state].append((prev_state, col, weight))
                        if new_state not in states_by_layer[row]:
                            states_by_layer[row].append(new_state)

    return graph, DP, states_by_layer

def dp_permanent(n, k, A):
    """
    Use the DP approach (banded Toeplitz approach from the paper) to compute
    the permanent of the truncated matrix A (with bandwidth k).
    We sum DP[(n, state)] over all states in layer n.
    """
    graph, DP, states_by_layer = build_truncated_assignment_graph(n, k, A)
    total = 0.0
    for s in states_by_layer[n]:
        total += DP.get((n, s), 0.0)
    return total

import bisect
import random

def sample_permutation_from_dp(n, k, graph, DP, states_by_layer):
   """
    Sample a permutation from the distribution induced by the DP table.
    This version uses bisect for binary search.
    """
    final_state = ('1',) * k + ('0',) * k
    permutation = [None] * n
    current_state = final_state

    for row in range(n, 0, -1):
        edges = graph[row][current_state]  # list of (s_old, col, w_edge)
        
        scores = []
        states = []
        cols = []
        score_sum = 0.0

        for s_old, col, w in edges:
            dp_val = DP[row - 1, s_old]
            sc = dp_val * w
            if sc > 0.0:
                score_sum += sc
                scores.append(score_sum)
                states.append(s_old)
                cols.append(col)

        if not scores:
            raise ValueError(f"No valid predecessor for state={current_state} at row={row}.")

        r = random.random() * score_sum
        idx = bisect.bisect_left(scores, r)
        permutation[row - 1] = cols[idx]
        current_state = states[idx]

    return permutation


import concurrent.futures
import time

# Version 1: Generate multiple samples sequentially (no parallelism)
def sample_many_sequential(n, k, graph, DP, num_samples=100):
    return [sample_permutation_from_dp_bisect(n, k, graph, DP,states_by_layer) for _ in range(num_samples)]

# Version 2: Generate multiple samples in parallel using concurrent.futures
def sample_many_parallel(n, k, graph, DP, num_samples=100, max_workers=4):
    def worker(_):
        return sample_permutation_from_dp_bisect(n, k, graph, DP,states_by_layer)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(worker, range(num_samples)))
    return results


## Example usage (on my laptop for n=20) sequential is better but for n=100 parallel is better.
# # Benchmark both methods
# num_samples = 100
# # Sequential
# start_seq = time.time()
# seq_results = sample_many_sequential(n, k, graph, DP, num_samples)
# end_seq = time.time()

# # Parallel
# start_par = time.time()
# par_results = sample_many_parallel(n, k, graph, DP, num_samples, max_workers=4)
# end_par = time.time()

# seq_time = end_seq - start_seq
# par_time = end_par - start_par

# seq_time, par_time, seq_time / par_time
    
