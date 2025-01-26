import numpy as np
from scipy.optimize import minimize
from utils import distance_alpha_batch
from GMM_diagonalized.partition_estimation import partition_estimation
from estimate_parameters_fixed.optimal_sigma_fixed import optimal_sigma
from estimate_parameters_fixed.dispersion_gradient_fixed import gradient_dispersion_parametes



def test_error(permutations_test, alpha, theta, sigma, Delta):
    nll,_ = neg_log_likelihood_and_grads(args=[alpha, theta], 
                                          permutation_samples=permutations_test, 
                                          sigma=sigma,
                                          Delta=Delta)
    return nll


def fit_mallow(permutation_samples, 
                 p_init=2.0,
                 theta_init=0.1,
                 sigma_init=None, 
                 Delta=5,
                 max_iter=5000, 
                 tol=1e-6,
                 verbose=True,
                 seed=42,
                 lambda_reg=1):
    """
    Perform L-BFGS optimization to minimize the negative log-likelihood 
    optimize alpha>=1,  and beta>0
    """
    print(f'fitting (fixed)mallow with scipy.optimize.minimize with p_init: {p_init}, theta_init: {theta_init}, sigma_init: {sigma_init}')
    np.random.seed(seed)
    m, n = len(permutation_samples), len(permutation_samples[0])

    # If no initial guess is provided, start from random point with norm 1
    if sigma_init is None:
        sigma_init = np.random.randint(0, n, size=n)
    if p_init is None:
        p_init = np.random.uniform(np.exp(1), np.exp(2)) 
    if theta_init is None:
        theta_init = np.random.uniform(0,1) 
    args_init = np.array([p_init, theta_init])



    # Define a function for L-BFGS that returns (loss, grad)
    def objective_and_grad(args):
        nll, grad = neg_log_likelihood_and_grads(args, permutation_samples, Delta, lambda_reg=lambda_reg)
        grad_flat = grad.ravel()
        return nll, grad_flat

    # Use scipy's L-BFGS-B
    result = minimize(
        fun=objective_and_grad,
        x0=args_init,
        method='L-BFGS-B',
        jac=True,
        bounds=[(np.exp(1), None), (0, 1)],
        options={'maxiter': max_iter,
                'disp': True,
                'gtol': tol}
    )
    print(f'result: {result}')

    # Reshape the final solution to (k, d)
    p_hat, theta_hat = result.x
    sigma_hat,_ = optimal_sigma(permutation_samples, alpha = np.log(p_hat))

    return p_hat, theta_hat, sigma_hat, result




def neg_log_likelihood_and_grads(args, permutation_samples, Delta, sigma=None, lambda_reg=1):
    # returns -1/m*log_likelihood = 1/m * (beta*sum(d(pi)) + m*log(Z))
    # 
    p, theta = args[0], args[1]
    n = len(permutation_samples[0])
    m = len(permutation_samples)
    if sigma is None:
        sigma,_ = optimal_sigma(permutation_samples, alpha=np.log(p))
    else:
        sigma = sigma

    distance_matrix = distance_alpha_batch(perms=permutation_samples, sigma=sigma, alpha=np.log(p))
    partition= partition_estimation(beta=-np.log(theta), alpha=np.log(p), sigma=sigma, Delta=Delta)

    nll = 1/m * (-np.log(theta) * np.sum(distance_matrix) + m * np.log(partition))
    
    grad_p  , grad_theta = gradient_dispersion_parametes(permutation_samples,
                                                          p= p, 
                                                          theta= theta, 
                                                          sigma=sigma,
                                                          Delta=Delta,
                                                          lambda_reg=lambda_reg)
    grads = np.array([grad_p, grad_theta])
    print(f'for p: {p:.2f}, theta: {theta:.2f}, nll: {nll:.2f}, grads: {[f"{g:.2f}" for g in grads]}, sigma: {sigma}')

    return nll, grads
 