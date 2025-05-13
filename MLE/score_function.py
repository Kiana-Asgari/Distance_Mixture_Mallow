import numpy as np

from GMM_diagonalized.sampling import sample_truncated_mallow

def psi_m_wrapper(x, pis, sigma, num_mc, rng_seed):
    """Wrapper for the objective function to avoid pickling issues."""
    ψ = psi_m(pis, sigma, x[0], x[1], num_mc=num_mc, rng_seed=rng_seed)
    return np.dot(ψ, ψ)  # scalar

def psi_m(pis: np.ndarray,
          sigma: np.ndarray,
          alpha: float,
          beta:  float,
          num_mc: int = 1000,
          rng_seed: int = 42):
    
    """
    Compute Ψₘ(α,β;σ)  for m empirical permutations `pis`
    and central permutation `sigma`.

    Parameters
    ----------
    pis      : (m, n) int array – sampled permutations π^{(1)},…,π^{(m)}
    sigma    : (n,)   int array – the central permutation σ̂
    alpha    : distance exponent α  (>0)
    beta     : Mallow scale β       (passed to the sampler)
    num_mc   : Monte-Carlo sample size for the expectations

    Returns
    -------
    np.ndarray shape (2,) with the two components of Ψₘ.
    """
    pis   = np.asarray(pis,   dtype=np.int16)
    sigma = np.asarray(sigma, dtype=np.int16)

    # ---------- helper: d_α and ẟd_α in one pass ----------
    def _d_and_ddiff(diff):
        """Return (d_α, ẟd_α) for |π−σ| array `diff` (any shape[..., n])."""
        diff_f  = diff.astype(float)
        diff_a  = diff_f ** alpha
        with np.errstate(divide='ignore'):          # 0·log0 → 0
            d_dot = diff_a * np.where(diff == 0, 0, np.log(diff_f))
        return diff_a.sum(-1), d_dot.sum(-1)

    # ---------- empirical part ----------
    diff_emp = np.abs(pis - sigma)                  # (m, n)
    d_emp, d_dot_emp = _d_and_ddiff(diff_emp)
    d_emp      = d_emp.mean()                       # 1/m Σ d_α
    d_dot_emp  = d_dot_emp.mean()                   # 1/m Σ ẟd_α

    # ---------- expectation (Monte-Carlo) ----------
    mc = sample_truncated_mallow(num_samples=num_mc, 
                                  n=len(sigma), 
                                  beta=beta, 
                                  alpha=alpha, 
                                  sigma=sigma,
                                  rng_seed=rng_seed)  # (num_mc, n)
    diff_mc = np.abs(mc - sigma)
    d_mc, d_dot_mc = _d_and_ddiff(diff_mc)
    d_mc     = d_mc.mean()
    d_dot_mc = d_dot_mc.mean()

    # ---------- assemble Ψₘ ----------
    hat_psi_m = np.array([-d_emp + d_mc, -d_dot_emp + d_dot_mc])
    if np.abs(hat_psi_m[0]) < 0.1 or np.abs(hat_psi_m[1]) < 0.1:
        print(f'        for alpha: {alpha}, beta: {beta}, hat_psi_m: {hat_psi_m}')
    return hat_psi_m
