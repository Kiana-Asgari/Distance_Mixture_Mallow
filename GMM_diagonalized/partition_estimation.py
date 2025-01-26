import numpy as np
from itertools import combinations
from utils import get_support_and_tk
from scipy.sparse import csr_matrix, identity




def partition_estimation(beta, alpha, sigma, Delta=4):
    """
    Args:
        beta: Dispersion parameter.
        alpha: Distance parameter.
        sigma: Reference permutation (tuple or list).
        Delta: Maximum distance for the support of the Toeplitz matrix.
    Returns:
        Z: Partition function Z_n(beta, alpha).
    """

    n = len(sigma)
    support, t_k = get_support_and_tk(beta, alpha, Delta)
    approx_perm=permanent_via_adjacency_power(support, t_k, n)

    Z = approx_perm
    return Z


def construct_adjacency_matrix(T, t_vals, n):
    """
    Construct the adjacency matrix for G_T as per Construction 1.

    Parameters
    ----------
    T : list
        The sorted list [p1, p2, ..., p_ell], with p1 < 0 < p_ell.
    t_vals : dict
        A dictionary mapping k -> t_k (over some ring, here treated as floats).
        That is, t_vals[k] = t_k for k in T.
    n : int
        The dimension relevant to computing per(A). 
        (Often ties back to the n x n Toeplitz matrix in the paper.)

    Returns
    -------
    A : np.ndarray
        The adjacency matrix for G_T of size |V| x |V|.
        Entries A[i, j] = sum of t_k for valid edges b_i -> b_j, but typically
        there is at most one valid k for each edge, so it's either t_k or 0.
    states : list of str
        An ordered list of all binary strings (vertices) in V.
        We use these to keep track of the 0-based indices.
    """
 
    # 1) Basic parameters
    p1, p_ell = T[0], T[-1]
    length = p_ell - p1  # length of each state-string
    # Each b in V has exactly -p1 ones (since p1 < 0, -p1 > 0).
    num_ones = -p1

    # 2) Generate all valid binary strings b of length = (p_ell - p1)
    #    with exactly (-p1) ones.  Index the bits by p1..(p_ell-1).
    #    We'll store them in "states" in lex order for convenience.
    all_bits = []
    for combo in combinations(range(length), num_ones):
        # Create a length-'length' list of 0's
        bits = ['0'] * length
        for idx in combo:
            bits[idx] = '1'
        all_bits.append(''.join(bits))

    # Sort them lexicographically (or any consistent order).
    # (This is just to have a reproducible indexing.)
    states = sorted(all_bits)

    # Map state-string -> index
    state_to_idx = {b: i for i, b in enumerate(states)}

    # 3) Prepare adjacency matrix
    size = len(states)
    A = np.zeros((size, size), dtype=float)  # or complex, if needed

    # Helper to "append 0" to a binary string
    def append_zero(b):
        return b + '0'  # b is like '0101', so b^0 is '01010'

    # Helper to flip bit k (where k is an integer in [p1..p_ell]) in b^0
    # Note: we need to translate from "graph index k" to "string index".
    # b^0 has length = length+1, indexed as 0..length
    # but the bit "k" in Construction 1 actually means offset (k - p1).
    def flip_bit(b_with_0, k):
        offset = k - p1  # map k to [0..(length)]
        # Flip only if it's '0'
        if b_with_0[offset] == '0':
            return (b_with_0[:offset] + '1' + b_with_0[offset+1:])
        # If it's already '1', that edge is invalid by (b^0)_k = 0 constraint
        return None

    # Helper to left-shift a length-(length+1) string, dropping the leftmost bit
    def L_shift(b_plus_e_k):
        return b_plus_e_k[1:]  # simply drop index 0

    # 4) Build edges: for each b in V, do the b -> L( b^0 + e_k ) if valid
    for b in states:
        i = state_to_idx[b]  # row index
        # b^0
        b0 = append_zero(b)
        # For each k in T, check if (b^0)_k == 0
        for k in T:
            flipped = flip_bit(b0, k)
            if flipped is None:
                continue  # means it was invalid (already '1')
            # Now left-shift
            shifted = L_shift(flipped)
            # Check if shifted is in V
            if shifted in state_to_idx:
                j = state_to_idx[shifted]
                A[i, j] += t_vals[k]  # Add the weight t_k

    return A, states


def permanent_via_adjacency_power(T, t_vals, n):
    """
    Compute per(A) using the fact that per(A) = (A(G_T)^n)_{1,1},
    i.e. the (0,0) entry in 0-based Python indexing.

    Parameters
    ----------
    T : list
        Sorted list [p1, p2, ..., p_ell].
    t_vals : dict
        Dictionary k -> t_k (over ring R, used here as floats for simplicity).
    n : int
        Matrix dimension from the original problem.

    Returns
    -------
    float
        The value of per(A).
    """
    # 1) Build adjacency matrix for G_T
    A, states = construct_adjacency_matrix(T, t_vals, n)
    #for j in range(len(A)):
    #    print(f'non zero cols of col {bin(j)[2:]:0>{len(states[0])}}: {[bin(i)[2:].zfill(len(states[0])) for i in np.nonzero(A[:,j])[0]]}')

    # Extract the (-1, -1) element of the result
    return last_element_of_A_power_n(A, n)
    #A_n = np.linalg.matrix_power(A, n)

    # print(A_n)
    # 3) The "1,1" entry in math is the (0,0) entry in Python's 0-based indexing
    #return A_n[-1, -1]



def last_element_of_A_power_n(A, n):
    """
    Returns the (-1, -1) element of A^n, where A is a sparse square matrix.
    
    A: a scipy.sparse matrix (CSR or similar) of shape (2^k, 2^k).
    n: positive integer (possibly ~2^k).
    """
    # Sanity check: dimension must match
    dim = A.shape[0]
    assert A.shape[0] == A.shape[1], "A must be square."
    assert n >= 1, "n must be >= 1."
    
    # We will compute row_vec * A^n, where row_vec is the last row basis e_{-1}^T:
    #    e_{-1}^T = [0, 0, ..., 0, 1]
    # but we don't need to store the entire vector of size dim in memory
    # if all we want is the final row's content.  However, in Python,
    # we typically DO keep it as a sparse or dense vector.  For large dim,
    # you may want a sparse representation if it's beneficial.

    # row_vec[i] = 1 if i == dim-1 else 0
    row_vec = np.zeros(dim, dtype=A.dtype)
    row_vec[dim - 1] = 1
    
    # We will do exponentiation by squaring. Let M = A initially.
    # Then repeatedly square M and multiply row_vec by M when needed.
    
    M = A.copy()  # the current power of A we are "looking at": A^(2^bit)
    power = n
    while power > 0:
        if power % 2 == 1:
            # row_vec = row_vec * M
            # We do a sparse-dense multiply => result is a 1D dense vector
            row_vec = row_vec @ M
        # Now square M: M = M^2
        M = M @ M
        power //= 2

    # After the loop, row_vec = e_{-1}^T * A^n, i.e. it is exactly the last row of A^n.
    # We just want the last element, row_vec[dim - 1].
    return row_vec[dim - 1]



