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
    Same idea as Algorithm 1 in your paper.
    Returns:
      graph[ell][state_new] = list of (state_old, col_assigned, edge_weight)
      DP dict: DP[(ell, state)] = total cumulative weight
      states_by_layer[ell] = list of all states at layer ell
    Each 'state' is a tuple of 2k bits ('0' or '1'), representing which columns in [ell-k, ell+k] are used.
    
    For a matrix that is *already truncated* (A(i,j)=0 if |i-j|>k), this DP ends up computing per(A).
    """
    init_state = ('1',)*k + ('0',)*k
    
    graph = [defaultdict(list) for _ in range(n+1)]
    DP = {}
    states_by_layer = [[] for _ in range(n+1)]
    
    # DP at layer 0 = 1 for the init_state
    DP[(0, init_state)] = 1.0
    states_by_layer[0].append(init_state)

    def shift_and_append_zero(state_tuple):
        # drop the leftmost bit, append '0' on right
        return state_tuple[1:] + ('0',)
    
    def flip_bit(state_tuple, i):
        new_state = list(state_tuple)
        new_state[i] = '1' if state_tuple[i] == '0' else '0'
        return tuple(new_state)
    
    for row in range(1, n+1):
        for s_old in states_by_layer[row-1]:
            old_val = DP.get((row-1, s_old), 0.0)
            if old_val == 0:
                continue
            s_shift = shift_and_append_zero(s_old)
            
            # We can flip exactly one '0' -> '1' in s_shift to assign row->that col
            for i_bit in range(2*k):
                # ------------------ FIX: check the column range first ------------------
                col_assigned = (row - k) + i_bit
                if col_assigned < 1 or col_assigned > n:
                    # This bit would correspond to an out-of-range column
                    continue
                # ----------------------------------------------------------------------
                
                if s_shift[i_bit] == '0':
                    s_new = flip_bit(s_shift, i_bit)
                    
                    w_edge = A(row, col_assigned)
                    if w_edge == 0.0:
                        continue
                    
                    new_val = DP.get((row, s_new), 0.0) + old_val * w_edge
                    DP[(row, s_new)] = new_val
                    graph[row][s_new].append((s_old, col_assigned, w_edge))
                    
                    if s_new not in states_by_layer[row]:
                        states_by_layer[row].append(s_new)
                        
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


def sample_permutation_from_dp(n, k, graph, DP, states_by_layer):
    """
    Given:
      n, k : problem size and band width
      graph : the layered adjacency graph (graph[row][s_new] = list of (s_old, col, w_edge))
      DP : DP[(row, state)] table built by build_truncated_assignment_graph
      states_by_layer : states_by_layer[row] holds all states at layer 'row'

    Returns:
      A list of length n, where result[row-1] = the column assigned to row.
      This is a random sample from the distribution implied by the DP.
    """
    # 1) Select a final state s at layer n with probability proportional to DP[(n, s)].
    final_states = states_by_layer[n]
    total_final = sum(DP.get((n, s), 0.0) for s in final_states)
    if total_final <= 1e-14:
        raise ValueError("No valid permutations at layer n (DP sum is 0).")

    r = random.random() * total_final
    cum = 0.0
    chosen_final = final_states[0]
    for s in final_states:
        val = DP.get((n, s), 0.0)
        cum += val
        if cum >= r:
            chosen_final = s
            break

    # We'll store the chosen column for each row in 'permutation'
    permutation = [None] * n
    current_state = chosen_final

    # 2) Backtrack from row n down to row 1
    for row in range(n, 0, -1):
        # We look at all possible predecessors (s_old, col_assigned, w_edge) in graph[row][current_state].
        # We define a score = DP[(row-1, s_old)] * w_edge. We pick s_old with probability = score / sum_of_scores.
        edges = graph[row][current_state]  # list of (s_old, col, w_edge)
        score_sum = 0.0
        for (s_old, col_assigned, w_edge) in edges:
            score_sum += DP.get((row-1, s_old), 0.0) * w_edge

        if score_sum <= 1e-14:
            raise ValueError(f"No valid predecessor for state={current_state} at row={row}.")

        # Sample among these edges based on the 'score'
        r2 = random.random() * score_sum
        c2 = 0.0
        chosen_s_old = None
        chosen_col = None
        for (s_old, col_assigned, w_edge) in edges:
            sc = DP.get((row-1, s_old), 0.0) * w_edge
            c2 += sc
            if c2 >= r2:
                chosen_s_old = s_old
                chosen_col = col_assigned
                break

        # Record the chosen column for this row
        permutation[row - 1] = chosen_col
        current_state = chosen_s_old  # step backward one layer

    return permutation