import numpy as np
from scipy.optimize import linear_sum_assignment

def learn_sigma(permutations, alpha):
    """
    Find the minimum-weight perfect matching in a bipartite graph
    given by its n x n adjacency/weight matrix A.

    Args:
        A (numpy.ndarray): An (n x n) array of non-negative weights.
                          A[i, j] is the cost/weight of matching row i to column j.

    Returns:
        row_ind (numpy.ndarray): Indices of rows in the matching.
        col_ind (numpy.ndarray): Corresponding indices of columns matched to each row.
        min_cost (float): The sum of the weights for the minimum-cost matching.
    """
    # row_ind, col_ind gives the matching: 
    # row i is matched to col j for each pair (i, j).
    n = len(permutations[0])
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            A_ij_batch = (np.abs(permutations[:,i] - j))**alpha
            A[i,j] = np.sum(A_ij_batch)

    row_ind, col_ind = linear_sum_assignment(A)
    assignment = np.zeros(A.shape[0], dtype=int)
    assignment[row_ind] = col_ind
    # Calculate the cost of this matching
    min_cost = A[row_ind, col_ind].sum()
    return assignment, min_cost