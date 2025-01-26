import numpy as np
from utils import distance_alpha_batch
from itertools import permutations
from learning_params_new.learn_alpha import learn_beta_and_sigma


def test_mle_toy_1():
    print('*****************testing mle toy 1*******************')
    n = 15
    observed_permutations = np.array([
        np.arange(n),
        np.arange(n),
        np.arange(n),
        np.arange(n)[::-1]
    ]) 

    #observed_permutations = np.array([np.random.permutation(n) for _ in range(10)])

    # Initial parameter guesses

    beta_hat, sigma_hat = learn_beta_and_sigma(observed_permutations,
                                                alpha=1, beta_init=1, Delta=9)
    print(f"beta_hat: {beta_hat}")
    print(f"sigma_hat: {sigma_hat}")
    

