import numpy as np
import sys

from datasets.load_sushi_prefrence import load_sushi_data
from learning_params_new.learn_alpha import learn_beta_and_sigma

def learn_sushi_preference():
    print('***********learn_sushi_preference***********')
    sushi_rankings = load_sushi_data()
    print(f'sushi_rankings data: {sushi_rankings.shape}')
    n = sushi_rankings.shape[1]
    """
    sigma_0 = np.arange(n)
    p_hat, theta_hat, sigma_hat, result = fit_mallow(permutation_samples=sushi_rankings, 
                                                        p_init=np.exp(3),
                                                        theta_init=np.exp(-1),
                                                        sigma_init=sigma_0, 
                                                        Delta=5,
                                                        max_iter=5000, 
                                                        tol=1e-6,
                                                        verbose=True,
                                                        seed=42,
                                                        lambda_reg=1)
    beta_hat = -1*np.log(theta_hat)
    alpha_hat = np.log(p_hat)
    print(f'alpha_hat: {alpha_hat}')
    print(f'beta_hat: {beta_hat}')
    print(f'sigma_hat: {sigma_hat}')  
    """

    beta_hat, sigma_hat = learn_beta_and_sigma(sushi_rankings, alpha=2, beta_init=2)
    print(f'beta_hat: {beta_hat}')
    print(f'sigma_hat: {sigma_hat}')  
    return sushi_rankings
