import numpy as np
import sys
import time

from GMM_diagonalized.sampling import sample_truncated_mallow
from MLE.consensus_ranking_estimation import consensus_ranking_estimation
from MLE.alpha_beta_estimation import solve_alpha_beta
from synthethic_tests.synthethic_script import save_synthetic_data
from synthethic_tests.plot import plot_alpha_vs_n_samples, plot_beta_vs_n_samples

if __name__ == "__main__":
    print('****************************Running main.py****************************')

    save_synthetic_data(n=15, alpha_0=1.5, beta_0=0.5)
    #plot_alpha_vs_n_samples(alpha_0=1.5, beta_0=0.5, n=10)
    #plot_beta_vs_n_samples(alpha_0=1.5, beta_0=0.5, n=10)
    sys.exit()
    n = 15
    Delta = 6
    sigma_0 = 1+np.arange(n)
    beta_0 = 0.3
    alpha_0 = 1.5
    num_train_samples = 300



    train_samples = sample_truncated_mallow(num_samples=num_train_samples,
                                             n=n, beta=beta_0, alpha=alpha_0,
                                            sigma=sigma_0, Delta=Delta)
    print(f'done sampling {num_train_samples} samples')
    consensus_ranking = consensus_ranking_estimation(train_samples)

    print(f'done estimating consensus ranking: {consensus_ranking}')
    params = solve_alpha_beta(train_samples, consensus_ranking)
    print('alpha beta estimation finished', params)
