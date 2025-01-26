import numpy as np
from scipy.optimize import minimize
from learning_params_new.learn_beta import learn_beta
from learning_params_new.learn_sigma import learn_sigma

def learn_beta_and_sigma(permutation_samples,
                         alpha,
                         beta_init,
                         Delta,
                         max_iter=5000,
                         learning_rate=1e-4,
                         tol=1e-8,
                         gtol=1e-3,
                         seed=42):
    sigma_opt, _ = learn_sigma(permutation_samples, alpha)
    beta_opt = learn_beta(permutation_samples,
                           alpha=alpha,
                           sigma=sigma_opt,
                           beta_init=beta_init,
                           Delta=Delta)
    return beta_opt, sigma_opt
