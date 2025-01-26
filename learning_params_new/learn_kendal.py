import numpy as np
from scipy.optimize import minimize

def kendall_tau_distance(rank1, rank2):
    n = len(rank1)
    inv_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (rank1[i] - rank1[j]) * (rank2[i] - rank2[j]) < 0:
                inv_count += 1
    return inv_count

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

def estimate_mallows_parameters(rankings):
    pi_0 = borda_count(rankings)
    result = minimize(negative_log_likelihood, x0=1.0, args=(rankings, pi_0), bounds=[(0.01, None)])
    theta_hat = result.x[0]
    return pi_0, theta_hat

