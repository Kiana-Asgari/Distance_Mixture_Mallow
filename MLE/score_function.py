import numpy as np
from scipy.optimize import root




from GMM_diagonalized.sampling import sample_many_parallel


# ----------------------------------------------------------------------
#  Distance statistics -------------------------------------------------
# ----------------------------------------------------------------------
def _d_alpha(batch: np.ndarray, sigma: np.ndarray, alpha: float) -> np.ndarray:
    diff = np.abs(batch - sigma)            
    return np.sum(diff ** alpha, axis=-1)

def _dot_d_alpha(batch: np.ndarray, sigma: np.ndarray, alpha: float) -> np.ndarray:
    diff = np.abs(batch - sigma)
    pow_ = diff ** alpha
    return np.sum(np.log(np.maximum(diff, 1)) * pow_, axis=-1)

# ----------------------------------------------------------------------
#  Moment-equation (score) ---------------------------------------------
# ----------------------------------------------------------------------
def _psi(u: np.ndarray,
         training_data: np.ndarray,
         sigma: np.ndarray,
         m_exp: int = 512,
         rng: np.random.Generator = np.random.default_rng()) -> np.ndarray:

    alpha = 1.0 + np.exp(u[0])
    beta  = 1e-5 + np.exp(u[1])

    # sample averages ---------------------------------------------------
    d_emp   = _d_alpha(training_data, sigma, alpha).mean()
    dd_emp  = _d_alpha(training_data, sigma, alpha).mean()

    # Monte-Carlo expectation under P_{α,β}^{Δ} -------------------------
    population_data   = sample_many_parallel(m_exp, alpha, beta, rng)        # shape (m_exp,d)
    d_mod   = _d_alpha(population_data,  sigma, alpha).mean()
    dd_mod  = _d_alpha(population_data, sigma, alpha).mean()

    return np.asarray([d_emp - d_mod, dd_emp - dd_mod])

# ----------------------------------------------------------------------
#  Public wrapper ------------------------------------------------------
# ----------------------------------------------------------------------
def fit_mallows(training_data: np.ndarray,
                sigma: np.ndarray,
                x0=(0.0, 0.0),
                m_exp: int = 512,
                tol: float = 1e-6) -> tuple[float, float]:

    sol = root(_psi, x0,
               args=(training_data, sigma, m_exp),
               method='hybr', tol=tol)

    if not sol.success:
        raise RuntimeError(sol.message)

    alpha_hat = 1.0 + np.exp(sol.x[0])
    beta_hat  = 1e-5 + np.exp(sol.x[1])
    return alpha_hat, beta_hat

# ----------------------------------------------------------------------
#  Example usage -------------------------------------------------------
if __name__ == "__main__":
    n, d = 2000, 50
    rng  = np.random.default_rng(123)
    sigma = np.arange(d)                        # identity permutation
    data  = rng.permutation(np.tile(sigma, (n,1)))   # fake data

    # Fit ---------------------------------------------------------------
    alpha_hat, beta_hat = fit_mallows(data, sigma)
    print(f"α̂ = {alpha_hat:.4f},  β̂ = {beta_hat:.4f}")
