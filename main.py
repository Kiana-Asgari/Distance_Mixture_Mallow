import numpy as np
import sys

from log_data.log_american_football import log_american_football_vs_alpha, plot_football_results, plot_football_first_fold
from log_data.log_basketball import log_basketball_vs_alpha



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
    #log_basketball_vs_alpha(Delta=11,n_file=100,n_top_teams=21,n_bottom_teams=15) # 4:30
    #log_basketball_vs_alpha(Delta=11,n_file=100,n_top_teams=21,n_bottom_teams=10) # 3:25
    #log_basketball_vs_alpha(Delta=11,n_file=100,n_top_teams=11,n_bottom_teams=15) # 2:20
    log_basketball_vs_alpha(Delta=11,n_file=100,n_top_teams=11,n_bottom_teams=10) # 1:15
    
    #plot_football_first_fold(n_file=100, n_top_teams=11, n_bottom_teams=10)
    #step 9: test the sushi preference dataset
    #log_sushi_vs_alpha(Delta=10)
    #step 10: test the synthetic data
    #learn_synthetic_mallow(n=10, alpha=1, beta=1, sigma=np.arange(10))
    #learn_synthetic_kendal(n=10, alpha=1, beta=1, sigma=np.arange(10))

    #step 11: test the kendal dataset
    # Example usage:`


    
