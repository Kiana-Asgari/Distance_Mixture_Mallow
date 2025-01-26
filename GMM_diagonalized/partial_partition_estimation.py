import numpy as np
from itertools import combinations
from utils import get_support_and_tk
from GMM_diagonalized.partition_estimation import permanent_via_adjacency_power


def marginal_probabilities(n, beta, alpha, Delta=6):
    T_example, t_vals_example = get_support_and_tk(beta, alpha, Delta)
    Z = permanent_via_adjacency_power(T_example, t_vals_example, n)

    A_offset = partial_permanent_sigma_i_equals_j(
        T_example, t_vals_example, n
    )

    probabilities = np.zeros((n, n))

    for offset in range(max(T_example)):
        for i in range(n):
              if i+offset>=0 and i+offset<n:
                probabilities[i, i+offset] = A_offset[offset,i]/ Z

              if i-offset>=0 and i-offset<n:
                probabilities[i, i-offset] = A_offset[offset,i]/ Z
    return probabilities






def partial_permanent_sigma_i_equals_j(T, t_vals, n):
    """
    Compute the sum of weights of all permutations of type T
    such that sigma(i) = j.

    per_{(sigma(i)=j)}(A) = sum over all permutations with sigma(i)=j.

    Using the adjacency matrix approach:
      sum_{k : k=(j-i) in T} [ (A^(i-1) * A_k * A^(n-i))[0,0] ].

    Parameters
    ----------
    T : list
        Offsets [p1, ..., p_ell].
    t_vals : dict
        p_k -> t_{p_k}.
    n : int
        Matrix dimension (so we do A^n).
    i : int
        The row index we're constraining in the permutation (1-based or 0-based).
    j : int
        The column index we want sigma(i) = j.

    Returns
    -------
    float
        The partial permanent contributed by permutations with sigma(i)=j.
    """
    # 1) Build the full adjacency matrix A
    from numpy.linalg import matrix_power

    # A_full, states = construct_adjacency_matrix(T, t_vals,n)

    # Decide if i, j are 1-based or 0-based. Let's assume the original
    # question uses 1-based indexing for sigma. If so, the offset is (j - i).

    # 2) Build "A_k" for offset_needed

    A_full, A_k_dict, states = construct_all_Ak(T, t_vals)
    A_offset_i = np.zeros(((max(T)-min(T)),n))

    for offset_needed in T:
      A_k=A_k_dict[offset_needed]
      for i in range(n):
        if i-1 > 0:
            A_left = matrix_power(A_full, i-1)
        else:
            A_left = np.eye(A_full.shape[0], dtype=float)

        if (n - i) > 0:
            A_right = matrix_power(A_full, n - i)
        else:
            A_right = np.eye(A_full.shape[0], dtype=float)

        # 4) Multiply A_left * A_k * A_right, then read [0,0]
        product_mid = A_left @ A_k @ A_right
        partial_sum = product_mid[-1, -1]
        A_offset_i[offset_needed,i]=partial_sum
    return A_offset_i



def construct_all_Ak(T, t_vals):
    """
    Build the adjacency matrices {A_k | k in T} and their sum A
    for the directed graph G_T (Construction 1).

    Returns
    -------
    A : np.ndarray
        The full adjacency matrix.
    A_k_dict : dict
        Dictionary offset -> adjacency matrix A_k
    states : list of str
        Vertex set in stable order.
    """
    p1, p_ell = T[0], T[-1]
    length = p_ell - p1
    num_ones = -p1

    # Generate all states
    all_bits = []
    for combo in combinations(range(length), num_ones):
        bits = ['0'] * length
        for idx in combo:
            bits[idx] = '1'
        all_bits.append(''.join(bits))

    states = sorted(all_bits)
    state_to_idx = {b:i for i,b in enumerate(states)}
    size = len(states)

    # Initialize A and A_k
    A = np.zeros((size, size), dtype=float)
    A_k_dict = {}
    for k in T:
        A_k_dict[k] = np.zeros((size, size), dtype=float)

    def append_zero(b):
        return b + '0'

    def flip_bit(b_with_0, k):
        offset = k - p1
        if b_with_0[offset] == '0':
            return b_with_0[:offset] + '1' + b_with_0[offset+1:]
        return None

    def left_shift(b_str):
        return b_str[1:]

    # Fill A and each A_k
    for b in states:
        i = state_to_idx[b]
        b0 = append_zero(b)
        for k in T:
            flipped = flip_bit(b0, k)
            if flipped is None:
                continue
            shifted = left_shift(flipped)
            if shifted in state_to_idx:
                j = state_to_idx[shifted]
                w = t_vals[k]
                A[i, j] += w
                A_k_dict[k][i, j] = w  # exactly w (not +=, because flipping offset k is unique for that edge)

    return A, A_k_dict, states

