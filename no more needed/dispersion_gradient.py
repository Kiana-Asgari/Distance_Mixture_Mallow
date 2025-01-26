from GMM_diagonalized.partial_partition_estimation import marginal_probabilities
from GMM_diagonalized.partition_estimation import partition_estimation
import numpy as np


def gradient_dispersion_parametes(permutation_samples,
                                  beta,
                                  alpha,
                                  sigma,
                                  Delta,
                                  verbose=False):

    n = len(permutation_samples[0])
    m = len(permutation_samples)

    exp_displacement, log_displacement, partition_func= compute_displacement_expectation(beta_0=beta, 
                                                                                        alpha_0=alpha, 
                                                                                        n=n,
                                                                                        Delta=Delta,
                                                                                        verbose=verbose)
    emp_displacement, emp_log_displacement = compute_displacement_empirical(permutation_samples,
                                                                            sigma, 
                                                                            alpha,
                                                                            verbose=verbose)
    grad_beta = -1*emp_displacement + m*exp_displacement
    grad_alpha = -beta*emp_log_displacement + m*beta*log_displacement
    return -1/m * grad_alpha, -1/m * grad_beta 




def compute_displacement_empirical(permutation_samples, sigma_0, alpha_0, verbose=False):
    if verbose:
        print('     **computing the displacement empirical for alpha={alpha_0}, n={n}, m={m}, sigma={sigma_0}')
    #permutaion_samples is a numpy 2D array of size m x n
    n = len(permutation_samples[0])
    m = len(permutation_samples)

    sigma_batch = np.tile(sigma_0, (m, 1))
    abs_matrix = np.abs(permutation_samples - sigma_batch)
    alpha_distance_matrix = abs_matrix**alpha_0

    
    log_alpha_distance_matrix = np.log(np.where(abs_matrix == 0, 1, abs_matrix))


    alpha_distance = np.sum(alpha_distance_matrix, axis=1)
    log_alpha_distance = np.sum(log_alpha_distance_matrix * alpha_distance_matrix, axis=1)


    emp_displacement_expectation = np.sum(alpha_distance)
    emp_log_displacement_expectation = np.sum(log_alpha_distance)


    return emp_displacement_expectation, emp_log_displacement_expectation








def compute_displacement_expectation(beta_0, alpha_0, n, Delta=4, verbose=False):
    sigma_0 = np.arange(n)
    partition_function = partition_estimation(beta_0, alpha_0,
                                              sigma_0, Delta=Delta)

    marginal_matrix =marginal_probabilities(n=n, beta=beta_0, alpha=alpha_0, Delta=Delta)
    # compute the (log) displacement expectation
    # displacement_expectation = E[ d_alpha(pi, id) ]
    # log_displacement_expectation = E[ sum_{i=1}^n log|pi(i)-sigma(i)| * |pi(i)-sigma(i)|^alpha ]
    displacement_expectation = 0
    log_displacement_expectation = 0

    for i in range(n): # i = row
        for j in range(n): # j = column
            prob = marginal_matrix[i,j]
            displacement_expectation += prob * (np.abs(i-j)**alpha_0)
            log_displacement_expectation += prob * (np.abs(i-j)**alpha_0) \
                                                 * np.log(max(1, np.abs(i-j)))
            #print(f'for i={i}, j={j}, prob={prob}, displacement={prob * (np.abs(i-j)**alpha_0)}')
    


    return displacement_expectation, log_displacement_expectation, partition_function