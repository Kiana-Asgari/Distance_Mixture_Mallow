from datasets.synthetic_mallow_data import generate_synthetic_mallow
from datasets.synthetic_kendal_data import generate_synthetic_kendal
from estimate_parameters_fixed.MLE_BFGS_fixed import fit_mallow, test_error





def learn_synthetic_kendal(n, alpha, beta,sigma, num_samples=50):
    permutation_samples = generate_synthetic_kendal(n, beta, sigma, num_samples)
    print(f'kendal permutation_samples: {permutation_samples.shape}')
    print(f'kendal permutation_samples: {permutation_samples}')
    permutation_samples_train = permutation_samples[:-10]
    permutation_samples_test = permutation_samples[-10:]
    alpha_hat, beta_hat, sigma_hat, result = fit_mallow(permutation_samples=permutation_samples_train, 
                                                        alpha_init=alpha,
                                                        beta_init=beta,
                                                        sigma_init=sigma, 
                                                        Delta=5,
                                                        max_iter=5000, 
                                                        tol=1e-6,
                                                        verbose=True,
                                                        seed=42)
    error = test_error(permutations_test=permutation_samples_test, 
                            alpha=alpha_hat, 
                            beta=beta_hat, 
                            sigma=sigma_hat, 
                            Delta=5)
    print(f'error: {error}')
    print(f'alpha_hat: {alpha_hat}')
    print(f'beta_hat: {beta_hat}')
    print(f'sigma_hat: {sigma_hat}')
    return alpha_hat, beta_hat, sigma_hat, result, error




def learn_synthetic_mallow(n, alpha, beta, sigma, num_samples=50, burn_in=10, thin=1):
    permutation_samples = generate_synthetic_mallow(n, alpha, beta, sigma, num_samples, burn_in, thin)
    print(f'permutation_samples: {permutation_samples.shape}')
    print(f'permutation_samples: {permutation_samples}')
    permutation_samples_train = permutation_samples[:-10]
    permutation_samples_test = permutation_samples[-10:]
    alpha_hat, beta_hat, sigma_hat, result = fit_mallow(permutation_samples=permutation_samples_train, 
                                                        alpha_init=alpha,
                                                        beta_init=beta,
                                                        sigma_init=sigma, 
                                                        Delta=5,
                                                        max_iter=5000, 
                                                        tol=1e-6,
                                                        verbose=True,
                                                        seed=42)
    error = test_error(permutations_test=permutation_samples_test, 
                            alpha=alpha_hat, 
                            beta=beta_hat, 
                            sigma=sigma_hat, 
                            Delta=5)
    print(f'error: {error}')
    print(f'alpha_hat: {alpha_hat}')
    print(f'beta_hat: {beta_hat}')
    print(f'sigma_hat: {sigma_hat}')
    return alpha_hat, beta_hat, sigma_hat, result, error