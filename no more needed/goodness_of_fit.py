def goodness_of_fit(permutations, beta, alpha, sigma, Z, sampler_func, num_samples=10000, burn_in=1000, thin=10):
    """
    Evaluate the goodness of fit for the GMM.

    Args:
        permutations: List of observed permutations (tuples or lists).
        beta: Estimated dispersion parameter.
        alpha: Estimated distance parameter.
        sigma: Estimated reference permutation (tuple or list).
        Z: Partition function Z_n(beta, alpha).
        sampler_func: Function to sample from GMM (Metropolis-Hastings).
        num_samples: Number of samples to collect.
        burn_in: Number of initial samples to discard.
        thin: Thinning factor to reduce autocorrelation.

    Returns:
        A dictionary containing log-likelihood, AIC, BIC, average observed distance,
        and expected distance under the model.
    """
    m = len(permutations)
    n = len(sigma)
    k = 2  # beta and alpha are the continuous parameters

    # Compute log-likelihood
    sum_d_alpha = sum(distance_alpha(p, sigma, alpha) for p in permutations)
    log_Z = math.log(Z)
    log_likelihood = -beta * sum_d_alpha - m * log_Z

    # Compute AIC and BIC
    AIC = 2 * k - 2 * log_likelihood
    BIC = k * math.log(m) - 2 * log_likelihood

    # Compute average observed distance
    avg_observed_distance = sum_d_alpha / m

    # Sample from GMM
    samples = sampler_func(beta, alpha, sigma, Z, num_samples, burn_in, thin)

    # Estimate expectation E[d_alpha(pi, sigma)]
    sum_d_exp = sum(distance_alpha(pi, sigma, alpha) for pi in samples) / len(samples)
    expected_distance = sum_d_exp

    # Compute log_d_alpha * log_diff expectation
    sum_d_log_diff = 0.0
    valid_samples = 0
    for pi in samples:
        d_alpha = distance_alpha(pi, sigma, alpha)
        log_diff = 0.0
        skip_pi = False
        for i in range(n):
            diff = abs(pi[i] - sigma[i])
            if diff == 0:
                skip_pi = True
                break  # Skip to avoid log(0)
            log_diff += math.log(diff)
        if not skip_pi:
            sum_d_log_diff += d_alpha * log_diff
            valid_samples += 1
    if valid_samples > 0:
        sum_d_log_diff /= valid_samples
    else:
        sum_d_log_diff = 0.0
    # Estimate sum_d_log_diff_exp
    sum_d_log_diff_exp = Z * sum_d_log_diff

    # Compile results
    results = {
        'log_likelihood': log_likelihood,
        'AIC': AIC,
        'BIC': BIC,
        'average_observed_distance': avg_observed_distance,
        'expected_distance': expected_distance,
        'sum_d_log_diff_exp': sum_d_log_diff_exp
    }

    return results

import random