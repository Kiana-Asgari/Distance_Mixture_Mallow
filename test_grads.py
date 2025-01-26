import numpy as np
from estimate_parameters_fixed.dispersion_gradient_fixed import compute_displacement_empirical
from estimate_parameters_fixed.dispersion_gradient_fixed import compute_displacement_expectation
from utils import distance_alpha_batch, brute_force_permanent, distance_alpha
from itertools import permutations
from estimate_parameters_fixed.optimal_sigma_fixed import optimal_sigma
"""
This file contains the tests for the MLE estimation of the parameters.
1) check if the gradient of beta is computed correctly

       1.1) checking if the empirical expectaion of 'displacement' is computed correctly:
            emp_displacement = sum_{pi in batch} d_alpha(pi, sigma)

       1.2) checking if the true expectation of 'displacement' is computed correctly
            displacement = E[ d_alpha(pi,id) ]

2) check if the gradient of alpha is computed correctly
       2.1) checking if the empirical expectaion of 'log displacement' is computed correctly:
            emp_log_displacement = sum_{pi in batch, i in [n]} log|pi(i)-sigma(i)| * |pi(i)-sigma(i)|^alpha

       2.2) checking if the true expectation of 'log displacement' is computed correctly
            true_log_displacement = E[ sum_{i=1}^n log|pi(i)-sigma(i)| * |pi(i)-sigma(i)|^alpha ]
3) check if the best sigma is computed correctly
"""

#step 1.1, 2.1: check if the empirical expectaion of '(log)displacement' is computed correctly
def displacement_empirical_test(n=10, alpha=1, Delta=4, sigma=None):
    print('*****************testing displacement empirical*****************')
    if sigma is None:
        sigma = np.arange(n)
    print(f'for alpha = {alpha}, Delta = {Delta}, sigma = {sigma},')

    num_perms = 4  # number of permutations
    permutation_batch = np.array([np.random.permutation(n) for _ in range(num_perms)])
    emp_displacement, emp_log_displacement = compute_displacement_empirical(permutation_samples=permutation_batch, 
                                                            sigma_0=sigma,
                                                            alpha_0=alpha,
                                                            verbose=False)
    true_displacement = np.sum(distance_alpha_batch(perms=permutation_batch, sigma=sigma, alpha=alpha))

    true_log_displacement = 0
    for perm in permutation_batch:
        for i in range(n):
            true_log_displacement += np.log(max(1, abs(perm[i]-sigma[i]))) * abs(perm[i]-sigma[i])**alpha
    print(f'set of permutations\n {permutation_batch} \n and sigma={sigma}:')
    print(f'        1)emp_displacement={emp_displacement}')
    print(f'        *)true_emp_displacement={true_displacement}')
    print(f'        2)emp_log_displacement={emp_log_displacement}')
    print(f'        *)true_emp_log_displacement={true_log_displacement}')

    

#step 1.2, 2.2: check if the true expectaion of '(log)displacement' is computed correctly
def displacement_expectation_test(n=6, alpha=1, beta=1, Delta=2, sigma=None):
    print('*****************testing displacement expectation*****************')
    if sigma is None:
        sigma = np.arange(n)
    print(f'        for alpha = {alpha}, beta = {beta}, sigma = {sigma}, Delta={Delta}:')
    displacement_exp, log_displacement_exp, partition_function=\
                        compute_displacement_expectation(beta, alpha, n, Delta, verbose=False)
    all_perms = list(permutations(range(n)))

    i, j = np.meshgrid(np.arange(n), np.arange(n))
    exp_matrix = np.exp(-beta * np.abs(i - sigma[j]))
    
    # Compute the true permanent
    true_partition_function = brute_force_permanent(exp_matrix)
    # compute the true displacement expectation
    true_displacement_exp = 0
    temp_displacement = np.zeros((n,n))
    temp_probs = np.zeros((n,n))
    for perm in all_perms:
        d = distance_alpha(perm, sigma, alpha)
        p = np.exp(-beta * d)/true_partition_function
        true_displacement_exp += d * p


    # compute the true log displacement expectation
    true_log_displacement_exp = 0
    for perm in all_perms:
        d = distance_alpha(perm, sigma, alpha)
        p = np.exp(-beta * d)/true_partition_function
        log_d = 0
        for i in range(n):
            log_d += np.log(max(1, abs(perm[i]-sigma[i]))) * abs(perm[i]-sigma[i])**alpha

        true_log_displacement_exp += log_d * p

    print(f'        1)partition_function={partition_function}')
    print(f'        *)true_partition_function={true_partition_function}')
    print(f'        2)displacement_exp={displacement_exp}')
    print(f'        *)true_displacement_exp={true_displacement_exp}')
    print(f'        3)log_displacement_exp={log_displacement_exp}')
    print(f'        *)true_log_displacement_exp={true_log_displacement_exp}')


def optimal_sigma_test(n=10, alpha=1):
    print('*****************testing optimal sigma*****************')
    print(f'          for alpha = {alpha}, n = {n}:')

    n_samples = 3
    permutation_batch = np.array([np.random.permutation(n) for _ in range(n_samples)])

    opt_sigma, min_cost = optimal_sigma(permutation_batch, alpha)
    distance_optimal_sigma = np.sum(distance_alpha_batch(perms=permutation_batch, sigma=opt_sigma, alpha=alpha))
    print(f'        for permutation batch=\n{permutation_batch}:')
    print(f'        1)opt_sigma={opt_sigma}')
    print(f'        2)min_cost={min_cost}')
    print(f'        3)distance_optimal_sigma={distance_optimal_sigma}')
    print(f'          for permutations inside the batch:')
    for perm in permutations(range(n)):
        d = np.sum(distance_alpha_batch(perms=permutation_batch, sigma=perm, alpha=alpha))
        print(f'       d({perm}) = {d}')







