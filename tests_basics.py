import numpy as np


from utils import distance_alpha_batch, distance_alpha, brute_force_permanent
from GMM_diagonalized.partition_estimation import partition_estimation
from GMM_diagonalized.partial_partition_estimation import marginal_probabilities
"""
This File contains small tests for seprate parts of the code.
1) check if the permanent is computed correctly
2) check if the partial permanent is computed correctly
3) check if the distance between two permutations is computed correctly
"""





#step 1: test the permanent 
def test_permanent(n=10, Delta=4, alpha=1, beta=1, sigma=None):
    print('*****************testing permanent*****************')
    if sigma is None:
        sigma = np.arange(n)
    perm = partition_estimation(beta=beta, alpha=alpha, sigma=sigma, Delta=Delta)  # Your function
    print(f'Estimated permanent with diagonalization for n = {n}, Delta = {Delta}, alpha = {alpha}, beta = {beta} is: {perm}')
    
    # Generate the matrix
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    exp_matrix = np.exp(-beta * (np.abs(i - sigma[j]))**alpha)
    
    # Compute the true permanent
    #true_permanent = brute_force_permanent(exp_matrix)
    #print(f'True permanent is: {true_permanent}')
    
    # Calculate and print the relative error
    #relative_error = abs(perm - true_permanent) / true_permanent
    #print(f'Relative error of our estimate: {relative_error}')


# step 2: partial permanent
# sum_{pi[i]=j} exp(-beta * d_alpha(pi, sigma)) = offset_marginal[|i-j|,i]
def test_partial_permanent(n=10, Delta=4, alpha=1, beta=1, sigma=None): 
    print('*****************testing partial permanent*****************')
    if sigma is None:
        sigma = np.arange(n)
            # Generate the matrix
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    exp_matrix = np.exp(-beta * (np.abs(i - sigma[j]))**alpha)

    marginal_dist = marginal_probabilities(beta=beta, alpha=alpha, n=n, Delta=Delta)
    #true_partition_function = brute_force_permanent(exp_matrix)

    
    for i in range(n):
        for j in range(n):
            marginal_matrix = exp_matrix.copy()
            marginal_matrix[i, :] = 0
            marginal_matrix[i, j] = 1
            print(f'  estimated p[{i}->{j}] = {float(marginal_dist[i,j]):3f}') #,true= {float(brute_force_permanent(marginal_matrix)/true_partition_function):3f}')


#step 3: distance between two permutations:
# d_alpha(pi, sigma) = sum_{i=1}^{n} |pi_i - sigma_i|^alpha
def test_alpha_distance(n=10, alpha=1, sigma=None):
    print('*****************testing alpha distance*****************')
    print(f'        for alpha = {alpha}, sigma = {sigma}:')
    if sigma is None:
        sigma = np.arange(n)
    num_perms = 4  # number of permutations

    permutation_batch = np.array([np.random.permutation(n) for _ in range(num_perms)])
    distance_batch = distance_alpha_batch(perms=permutation_batch, sigma=sigma, alpha=alpha)
    
    for i,perm in enumerate(permutation_batch):
        d = distance_alpha(perm, sigma, alpha)
        print(f'   d({perm},{sigma}) = {d}')
        print(f'   ...computed by the batch distance function : {distance_batch[i]}')   


