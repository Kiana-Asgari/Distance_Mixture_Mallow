import numpy as np
from itertools import permutations

def get_support_and_tk(beta, alpha, delta,): # note that this is strictly less than delta
    support = list(range(-delta + 1, delta))
    t_k = {offset: np.exp(-beta * (abs(offset)) ** alpha) for offset in support}
 
    return support, t_k

def distance_alpha(pi, sigma, alpha):
    #Compute the alpha-power distance between two permutations.
    distance = (np.abs(pi - sigma))**alpha
    return sum(distance)

def distance_alpha_batch(perms, sigma, alpha):
    #Compute the alpha-power distance between a batch of permutations and a reference permutation.
    #vectorized for the faster computation
    distance_matrix = (np.abs(perms - sigma))**alpha
    return np.sum(distance_matrix, axis=1)

# Brute Force Permanent Calculation
from itertools import permutations
def brute_force_permanent(matrix):
    n = matrix.shape[0]
    perms = permutations(range(n))
    perm_sum = 0
    for perm in perms:
        prod = 1
        for i in range(n):
            prod *= matrix[i, perm[i]]
        perm_sum += prod
    return perm_sum




