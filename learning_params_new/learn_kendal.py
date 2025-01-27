import numpy as np
from scipy.optimize import minimize
from scipy.stats import kendalltau

def kendall_tau_distance(rank1, rank2):
    n = len(rank1)
    correlation, _ = kendalltau(rank1, rank2)
    n_pairs = (n * (n-1)) // 2
    distance = int(n_pairs * (1 - correlation) / 2)
    return distance

def borda_count(rankings):
    n = len(rankings[0])
    scores = np.zeros(n)
    for rank in rankings:
        for i, item in enumerate(rank):
            scores[item] += i
    return np.argsort(scores)

def z_theta(theta, n):
    prod = 1.0
    for j in range(1, n + 1):
        prod *= (1 - np.exp(-j * theta)) / (1 - np.exp(-theta))
    return prod

def negative_log_likelihood(theta, rankings, pi_0):
    n = len(pi_0)
    total_distance = sum(kendall_tau_distance(pi, pi_0) for pi in rankings)
    return theta * total_distance + len(rankings) * np.log(z_theta(theta, n))

def learn_kendal(permutations_train, permutations_test):
    pi_0 = borda_count(permutations_train)
    result = minimize(negative_log_likelihood, x0=1.0, args=(permutations_train, pi_0), bounds=[(0.01, None)])
    theta_hat = result.x[0]
    error = 1/len(permutations_test) * negative_log_likelihood(theta_hat,permutations_test, pi_0)
    return pi_0, theta_hat, error


