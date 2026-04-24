"""Distance functions and exact / MCMC partition utilities for d_alpha."""

from __future__ import annotations

import math
from itertools import permutations
from typing import Iterable, Optional

import numpy as np


def d_alpha(pi: np.ndarray, sigma: np.ndarray, alpha: float) -> float:
    """d_alpha(pi, sigma) = sum_i |pi(i) - sigma(i)|^alpha.

    Both inputs use the same indexing convention (1-based or 0-based --
    only the elementwise differences matter).
    """
    pi = np.asarray(pi, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    return float(np.sum(np.abs(pi - sigma) ** alpha))


def d_alpha_dot(pi: np.ndarray, sigma: np.ndarray, alpha: float) -> float:
    """d/d_alpha of d_alpha(pi, sigma) = sum_i |pi(i)-sigma(i)|^alpha * log|pi(i)-sigma(i)|."""
    pi = np.asarray(pi, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    diff = np.abs(pi - sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        contrib = np.where(diff > 0, (diff ** alpha) * np.log(diff), 0.0)
    return float(np.sum(contrib))


def d_alpha_batch(pis: np.ndarray, sigma: np.ndarray, alpha: float) -> np.ndarray:
    pis = np.asarray(pis, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    return np.sum(np.abs(pis - sigma) ** alpha, axis=-1)


def enumerate_permutations(n: int) -> Iterable[np.ndarray]:
    base = np.arange(1, n + 1)
    for p in permutations(base):
        yield np.asarray(p, dtype=np.int64)


def exact_Z(n: int, alpha: float, beta: float, sigma: Optional[np.ndarray] = None) -> float:
    """Exact partition function Z_{alpha,beta} via S_n enumeration. Only n <= 10 is tractable."""
    if sigma is None:
        sigma = np.arange(1, n + 1)
    sigma = np.asarray(sigma, dtype=float)
    Z = 0.0
    for pi in enumerate_permutations(n):
        Z += math.exp(-beta * d_alpha(pi, sigma, alpha))
    return Z


def exact_expectations(n: int, alpha: float, beta: float, sigma: Optional[np.ndarray] = None):
    """Return (Z, E[d_alpha], E[d_dot_alpha]) under P_{alpha,beta} centered at sigma."""
    if sigma is None:
        sigma = np.arange(1, n + 1)
    sigma_f = np.asarray(sigma, dtype=float)
    Z = 0.0
    Ed = 0.0
    Edot = 0.0
    for pi in enumerate_permutations(n):
        d = d_alpha(pi, sigma_f, alpha)
        ddot = d_alpha_dot(pi, sigma_f, alpha)
        w = math.exp(-beta * d)
        Z += w
        Ed += w * d
        Edot += w * ddot
    return Z, Ed / Z, Edot / Z


# ----------------------------------------------------------------------
# Metropolis-Hastings sampler for P_{alpha,beta} via adjacent transpositions
# ----------------------------------------------------------------------
def mh_sample(
    n: int,
    alpha: float,
    beta: float,
    sigma: Optional[np.ndarray] = None,
    n_samples: int = 1000,
    burn_in: int = 1000,
    thin: int = 1,
    rng_seed: Optional[int] = None,
    init: Optional[np.ndarray] = None,
):
    """Metropolis-Hastings sampler with random adjacent-transposition proposals.

    Returns array of shape (n_samples, n) with permutations (1-based)
    drawn from P_{alpha, beta}(pi) propto exp(-beta d_alpha(pi, sigma)).
    """
    rng = np.random.default_rng(rng_seed)
    if sigma is None:
        sigma = np.arange(1, n + 1)
    sigma = np.asarray(sigma, dtype=np.int64)

    if init is None:
        pi = rng.permutation(np.arange(1, n + 1))
    else:
        pi = np.asarray(init, dtype=np.int64).copy()

    sigma_f = sigma.astype(float)

    def per_pos(pi_arr):
        return np.abs(pi_arr.astype(float) - sigma_f) ** alpha

    pi_pos = per_pos(pi)
    total = pi_pos.sum()

    samples = np.empty((n_samples, n), dtype=np.int64)
    out_idx = 0
    total_iter = burn_in + thin * n_samples
    for it in range(total_iter):
        i = rng.integers(0, n - 1)
        # propose swap of positions i and i+1
        a, b = pi[i], pi[i + 1]
        diff_a = abs(a - sigma[i + 1]) ** alpha + abs(b - sigma[i]) ** alpha
        diff_b = pi_pos[i] + pi_pos[i + 1]
        log_ratio = -beta * (diff_a - diff_b)
        if log_ratio >= 0 or rng.random() < math.exp(log_ratio):
            pi[i], pi[i + 1] = b, a
            new_a = abs(b - sigma[i]) ** alpha
            new_b = abs(a - sigma[i + 1]) ** alpha
            total += (new_a + new_b) - (pi_pos[i] + pi_pos[i + 1])
            pi_pos[i] = new_a
            pi_pos[i + 1] = new_b

        if it >= burn_in and (it - burn_in) % thin == 0:
            samples[out_idx] = pi
            out_idx += 1
            if out_idx == n_samples:
                break
    return samples


def mh_estimate_expectations(
    n: int,
    alpha: float,
    beta: float,
    sigma: Optional[np.ndarray] = None,
    n_samples: int = 5000,
    burn_in: int = 2000,
    thin: int = 1,
    rng_seed: Optional[int] = None,
):
    samples = mh_sample(
        n=n, alpha=alpha, beta=beta, sigma=sigma,
        n_samples=n_samples, burn_in=burn_in, thin=thin, rng_seed=rng_seed,
    )
    sigma_eff = np.arange(1, n + 1) if sigma is None else np.asarray(sigma)
    d_vals = np.array([d_alpha(s, sigma_eff, alpha) for s in samples])
    ddot_vals = np.array([d_alpha_dot(s, sigma_eff, alpha) for s in samples])
    return float(d_vals.mean()), float(ddot_vals.mean()), d_vals, ddot_vals


# ----------------------------------------------------------------------
# Bridge sampling for log Z when alpha < 1 (compare to a reference law)
# ----------------------------------------------------------------------
def bridge_sample_logZ(
    n: int,
    alpha: float,
    beta: float,
    sigma: Optional[np.ndarray] = None,
    n_samples: int = 5000,
    burn_in: int = 2000,
    rng_seed: Optional[int] = None,
):
    """Estimate log Z(alpha, beta) using bridge sampling against the uniform
    distribution on S_n.

    We use the identity:
        E_uniform[ exp(-beta d_alpha) ] = Z(alpha, beta) / n!
    so log Z = log(n!) + log(mean(exp(-beta d))).

    For numerical stability we use the log-sum-exp trick.
    """
    rng = np.random.default_rng(rng_seed)
    if sigma is None:
        sigma = np.arange(1, n + 1)
    sigma_f = np.asarray(sigma, dtype=float)

    log_terms = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        pi = rng.permutation(np.arange(1, n + 1))
        d = float(np.sum(np.abs(pi.astype(float) - sigma_f) ** alpha))
        log_terms[i] = -beta * d

    m = log_terms.max()
    log_mean = m + math.log(np.exp(log_terms - m).mean())
    log_factorial = math.lgamma(n + 1)
    return log_factorial + log_mean


def ti_logZ(
    n: int,
    alpha: float,
    beta: float,
    sigma: Optional[np.ndarray] = None,
    n_steps: int = 16,
    n_samples_per_step: int = 2000,
    burn_in: int = 1000,
    rng_seed: Optional[int] = None,
):
    """Thermodynamic integration estimator for log Z(beta).

    log Z(beta) = log(n!) - integral_0^beta E_t[d_alpha] dt
    where E_t is the expectation under P with temperature parameter t.

    Estimates the integral by Simpson's rule on n_steps + 1 grid points.
    """
    rng = np.random.default_rng(rng_seed)
    grid = np.linspace(0.0, beta, n_steps + 1)
    means = np.empty_like(grid)
    if sigma is None:
        sigma_arr = np.arange(1, n + 1)
    else:
        sigma_arr = np.asarray(sigma)
    for i, t in enumerate(grid):
        if t == 0:
            # E under uniform on S_n is computable in closed form for the
            # identity center: sum_i E[|X-i|^alpha] over uniform permutations.
            samples = mh_sample(
                n=n, alpha=alpha, beta=0.0, sigma=sigma_arr,
                n_samples=n_samples_per_step, burn_in=burn_in,
                rng_seed=int(rng.integers(0, 2**31)),
            )
        else:
            samples = mh_sample(
                n=n, alpha=alpha, beta=float(t), sigma=sigma_arr,
                n_samples=n_samples_per_step, burn_in=burn_in,
                init=np.arange(1, n + 1),
                rng_seed=int(rng.integers(0, 2**31)),
            )
        d_vals = np.array([float(np.sum(np.abs(s.astype(float) - sigma_arr.astype(float)) ** alpha))
                           for s in samples])
        means[i] = d_vals.mean()
    # Simpson's rule
    if n_steps % 2 != 0:
        # fall back to trapezoid if odd intervals
        integral = float(np.trapz(means, grid))
    else:
        h = (beta - 0.0) / n_steps
        integral = (h / 3.0) * (
            means[0] + means[-1]
            + 4.0 * means[1:-1:2].sum()
            + 2.0 * means[2:-1:2].sum()
        )
    return math.lgamma(n + 1) - integral


def gelman_rubin(chains: np.ndarray) -> float:
    """Potential scale reduction factor R-hat for K chains x N samples."""
    K, N = chains.shape
    if K < 2:
        return float("nan")
    chain_means = chains.mean(axis=1)
    chain_vars = chains.var(axis=1, ddof=1)
    W = chain_vars.mean()
    B = N * chain_means.var(ddof=1)
    var_hat = (1 - 1.0 / N) * W + B / N
    if W <= 0:
        return float("nan")
    return float(np.sqrt(var_hat / W))


def effective_sample_size(x: np.ndarray, max_lag: Optional[int] = None) -> float:
    """ESS via the initial monotone sequence estimator (Geyer)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if max_lag is None:
        max_lag = min(n - 1, 1000)
    x_centered = x - x.mean()
    var0 = (x_centered ** 2).mean()
    if var0 == 0:
        return float(n)
    rho = np.empty(max_lag + 1)
    rho[0] = 1.0
    for k in range(1, max_lag + 1):
        rho[k] = float(np.dot(x_centered[:-k], x_centered[k:]) / ((n - k) * var0))
    s = 1.0
    k = 1
    while k + 1 <= max_lag:
        pair = rho[k] + rho[k + 1]
        if pair < 0:
            break
        s += 2 * pair
        k += 2
    return float(n / max(s, 1e-12))
