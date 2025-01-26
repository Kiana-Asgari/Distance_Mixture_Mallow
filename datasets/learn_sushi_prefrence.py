import numpy as np
import sys

from datasets.load_sushi_prefrence import load_sushi_data
from learning_params_new.learn_alpha import learn_beta_and_sigma
from learning_params_new.likelihood_test import test_error
from learning_params_new.learn_kendal import estimate_mallows_parameters, negative_log_likelihood

def learn_sushi_preference():
    print('***********learn_sushi_preference***********')
    full_rankings = load_sushi_data()
    print(f'sushi_rankings data: {full_rankings.shape}')
    n = full_rankings.shape[1]


 # Randomly select 50 rankings for the test set
    np.random.seed(42)  # For reproducibility
    test_indices = np.random.choice(full_rankings.shape[0], 200, replace=False)
    full_rankings_test = full_rankings[test_indices]

    # Use the remaining rankings for training
    train_indices = np.setdiff1d(np.arange(full_rankings.shape[0]), test_indices)
    full_rankings_train = full_rankings[train_indices]

    print(f'rankings train: {full_rankings_train.shape}')
    print(f'rankings test: {full_rankings_test.shape}')

    pi_0_hat, theta_hat = estimate_mallows_parameters(full_rankings_train)
    kendal_error = 1/len(full_rankings_test) * negative_log_likelihood(rankings=full_rankings_test, theta=theta_hat, pi_0=pi_0_hat)



    print(f'Kendal: error: {kendal_error:3f}')
    print("Kendal: Estimated consensus ranking (pi_0):", pi_0_hat)
    print("Kendal: Estimated dispersion parameter (theta):", theta_hat)

    for alpha in np.linspace(1, 4, 20):
        beta_opt, sigma_opt = learn_beta_and_sigma(permutation_samples=full_rankings_train,
                                                    alpha=alpha,
                                                    beta_init=1,
                                                    Delta=10)
        error = test_error(full_rankings_test, beta_hat=beta_opt, sigma_hat=sigma_opt, alpha_hat=alpha)
        print(f'*for alpha={alpha}, beta_opt: {beta_opt:3f}, error: {error:3f}, sigma_opt: {sigma_opt}')
    sys.exit()
    return sushi_rankings
