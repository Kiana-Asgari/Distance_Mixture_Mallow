import numpy as np
from itertools import combinations
import random
import time
from utils import get_support_and_tk

def construct_states(T):
    p1, p_ell = T[0], T[-1]
    length = p_ell - p1
    num_ones = -p1
    all_bits = []
    for combo in combinations(range(length), num_ones):
        bits = ['0'] * length
        for idx in combo:
            bits[idx] = '1'
        all_bits.append(''.join(bits))
    states = sorted(all_bits)
    return states, p1, p_ell, length

def build_adjacency_lists(T, t_vals, states):
    p1, p_ell = T[0], T[-1]
    length = p_ell - p1
    M = len(states)
    state_to_idx = {b: i for i, b in enumerate(states)}

    def append_zero(b): return b + '0' # 1 means it has been occupied, so append with zero to show it's available
    def flip_bit(b_with_0, k):
        offset = k - p1
        if b_with_0[offset] == '0':
            return b_with_0[:offset] + '1' + b_with_0[offset+1:]
        return None
    def left_shift(b_str): return b_str[1:]

    adj_forward = [[] for _ in range(M)]
    adj_reverse = [[] for _ in range(M)]

    for b_idx, b in enumerate(states):
        b0 = append_zero(b)
        for k in T:
            flipped = flip_bit(b0, k)
            if flipped is None:
                continue
            shifted = left_shift(flipped)
            if shifted in state_to_idx:
                s_new_idx = state_to_idx[shifted]
                w = t_vals[k]
                adj_forward[b_idx].append((s_new_idx, w))
                adj_reverse[s_new_idx].append((b_idx, w))
                

    return adj_forward, adj_reverse

def run_dp_forward(adj_forward, n, start_idx, M):
    DP = np.zeros((n+1, M), dtype=float)
    DP[0, start_idx] = 1.0
    for i in range(1, n+1):
        for s_old in range(M):
            val_old = DP[i-1, s_old]
            if val_old == 0:
                continue
            for (s_new, w_edge) in adj_forward[s_old]:
                DP[i, s_new] += val_old * w_edge
    return DP

def sample_path_from_dp(DP, adj_reverse, n):
    M = DP.shape[1]
    path = [None] * (n+1)

    # Choose final state s_n
    dp_n = DP[n]
    total_n = np.sum(dp_n)
    if total_n == 0:
        raise ValueError("No valid final states.")
    probs_n = dp_n / total_n
    s_n = np.random.choice(M, p=probs_n)
    path[n] = s_n

    # Backtrack from n down to 1
    for i in range(n, 0, -1):
        s_current = path[i]
        candidates = adj_reverse[s_current]
        scores = []
        for (s_old, w_edge) in candidates:
            scores.append(DP[i-1, s_old] * w_edge)
        sum_scores = np.sum(scores)
        if sum_scores == 0:
            raise ValueError(f"No predecessor found at step {i}.")
        probs = [s / sum_scores for s in scores]
        idx = np.random.choice(len(candidates), p=probs)
        s_prev = candidates[idx][0]
        path[i-1] = s_prev

    return path

def get_partition_estimate_via_dp(beta, alpha, n, Delta):
    # Example usage
    np.random.seed(0)
    
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