import numpy as np
from scipy.optimize import minimize
from utils import distance_alpha_batch
from GMM_diagonalized.partition_estimation import partition_estimation
from estimate_paramaters.optimal_sigma import optimal_sigma
from estimate_paramaters.dispersion_gradient import gradient_dispersion_parametes



def test_error(permutations_test, alpha, beta, sigma, Delta):
    nll,_ = neg_log_likelihood_and_grads(args=[alpha, beta], 
                                          permutation_samples=permutations_test, 
                                          sigma=sigma,
                                          Delta=Delta)
    return nll


def fit_mallow(permutation_samples, 
                 alpha_init=2.0,
                 beta_init=2.0,
                 sigma_init=None, 
                 Delta=5,
                 max_iter=5000, 
                 tol=1e-6,
                 verbose=True,
                 seed=42):
    """
    Perform L-BFGS optimization to minimize the negative log-likelihood 
    optimize alpha>=1,  and beta>0
    """
    print(f'fitting mallow with scipy.optimize.minimize with alpha_init: {alpha_init}, beta_init: {beta_init}, sigma_init: {sigma_init}')
    np.random.seed(seed)
    m, n = len(permutation_samples), len(permutation_samples[0])

    # If no initial guess is provided, start from random point with norm 1
    if sigma_init is None:
        sigma_init = np.random.randint(0, n, size=n)
    if alpha_init is None:
        alpha_init = np.random.uniform(1,2) 
    if beta_init is None:
        beta_init = np.random.uniform(0,2) 
    args_init = np.array([alpha_init, beta_init])



    # Define a function for L-BFGS that returns (loss, grad)
    def objective_and_grad(args):
        nll, grad = neg_log_likelihood_and_grads(args, permutation_samples, Delta)
        grad_flat = grad.ravel()
        return nll, grad_flat

    # Use scipy's L-BFGS-B
    result = minimize(
        fun=objective_and_grad,
        x0=args_init,
        method='L-BFGS-B',
        jac=True,
        bounds=[(1, None), (1e-4, None)],
        options={'maxiter': max_iter,
                'disp': True,
                'gtol': tol}
    )
    print(f'result: {result}')

    # Reshape the final solution to (k, d)
    alpha_hat, beta_hat = result.x
    sigma_hat,_ = optimal_sigma(permutation_samples, alpha_hat)

    return alpha_hat, beta_hat, sigma_hat, result




def neg_log_likelihood_and_grads(args, permutation_samples, Delta, sigma=None):
    # returns -1/m*log_likelihood = 1/m * (beta*sum(d(pi)) + m*log(Z))
    # 
    alpha, beta = args[0], args[1]
    n = len(permutation_samples[0])
    m = len(permutation_samples)
    if sigma is None:
        sigma,_ = optimal_sigma(permutation_samples, alpha)
    else:
        sigma = sigma

    distance_matrix = distance_alpha_batch(perms=permutation_samples, sigma=sigma, alpha=alpha)
    partition= partition_estimation(beta=beta, alpha=alpha, sigma=sigma, Delta=Delta)
    nll = 1/m * (beta * np.sum(distance_matrix) + m * np.log(partition))
    
    grad_alpha, grad_beta = gradient_dispersion_parametes(permutation_samples,
                                                          alpha= alpha, 
                                                          beta= beta, 
                                                          sigma= sigma,
                                                          Delta=Delta)
    grads = np.array([grad_alpha, grad_beta])
    print(f'neg_log_likelihood_and_grads: nll: {nll}, grads: {grads}')

    return nll, grads
 