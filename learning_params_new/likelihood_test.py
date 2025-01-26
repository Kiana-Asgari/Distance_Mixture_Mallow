from utils import distance_alpha_batch
from GMM_diagonalized.DP_partition_estimation import get_partition_estimate_via_dp
import numpy as np

def test_error(full_rankings_test, alpha_hat, beta_hat, sigma_hat, Delta=10):
    n = len(sigma_hat)
    m = len(full_rankings_test)
    partition_estimate = get_partition_estimate_via_dp(beta_hat, alpha_hat, n, Delta=Delta)
    distance = np.sum(distance_alpha_batch(full_rankings_test, sigma_hat, alpha=alpha_hat))

    ll = 1/m* (-beta_hat * distance - m * np.log(partition_estimate) )
    return ll