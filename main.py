import numpy as np
import sys

from tests_basics import test_permanent, test_partial_permanent, test_alpha_distance
from test_grads import displacement_empirical_test, displacement_expectation_test, optimal_sigma_test
from test_mle_toy import test_mle_toy_1
from datasets.learn_american_football import learn_american_football
from datasets.learn_sushi_prefrence import learn_sushi_preference
from datasets.load_sushi_prefrence import load_sushi_data
from datasets.learn_synthetic_data import learn_synthetic_mallow, learn_synthetic_kendal
from GMM_diagonalized.DP_partition_estimation import get_partition_estimate_via_dp
if __name__ == "__main__":
    print('running main')
    n=100
    Delta=10
    alpha=1
    beta=1
    #step 1: test the permanent
    #test_permanent(n=n, Delta=Delta, alpha=alpha, beta=beta)
    #get_partition_estimate_via_dp(n=n, beta=beta, alpha=alpha, Delta=Delta)

    #step 2: test the marginal probabilities
    #test_partial_permanent(n=n, Delta=Delta, alpha=alpha, beta=beta)

    #step 3: test the distance between two permutations
    #test_alpha_distance(n=n, alpha=alpha)

    #step 4: test the empirical expectaion of 'displacement'
    #displacement_empirical_test(n=n, alpha=alpha)

    #step 5: test the expectation of 'displacement'
    #displacement_expectation_test(n=n, alpha=alpha, beta=beta, Delta=Delta)

    #step 6: test the optimal sigma
    #optimal_sigma_test(n=4, alpha=1)

    #step 7: test the MLE
    #test_mle_toy_1()

    #step 8: test the american football dataset
    learn_american_football()

    #step 9: test the sushi preference dataset
    #learn_sushi_preference()
    #step 10: test the synthetic data
    #learn_synthetic_mallow(n=10, alpha=1, beta=1, sigma=np.arange(10))
    #learn_synthetic_kendal(n=10, alpha=1, beta=1, sigma=np.arange(10))
    
