import numpy as np
from utils import distance_alpha_batch
from GMM_diagonalized.partition_estimation import partition_estimation


def log_likelihood(beta, alpha, sigma, permutations):
    n = len(sigma)
    distance_matrix = distance_alpha_batch(perms=permutations, sigma=sigma, alpha=alpha)
    partition= partition_estimation(beta, alpha, sigma, n)
    lieklihood_matrix = np.exp(-beta * distance_matrix)/partition
    return np.sum(np.log(lieklihood_matrix))