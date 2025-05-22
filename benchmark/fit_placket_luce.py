import numpy as np
from scipy.optimize import minimize


# ----------------------------------------------------------------------
#  High-level helper
# ----------------------------------------------------------------------
def learn_PL(permutations_train, permutations_test):
    """
    Fit a Plackett–Luce model and return (utilities , normalised-NLL on test set).
    """
    model = PlackettLuceModel(permutations_train)
    est_utils = model.fit()
    nll = model.compute_normalized_nll(permutations_test, est_utils)
    return est_utils, nll


# ----------------------------------------------------------------------
#  Plackett–Luce model class
# ----------------------------------------------------------------------
class PlackettLuceModel:
    """
    Maximum-likelihood learning for the Plackett–Luce model, written entirely
    with vectorised NumPy (no Python loops in the likelihood any more).
    """

    def __init__(self, rankings: np.ndarray):
        """
        Parameters
        ----------
        rankings : (n_rankings , n_items) int ndarray
            Each row is a full ranking from *best* (col 0) to *worst* (last col).
            Item values are indices 0 … n_items-1.
        """
        self.rankings = rankings
        self.n_items  = rankings.shape[1]

    # -----------------------------
    #  Vectorised negative log-likelihood
    # -----------------------------
    def negative_log_likelihood(self, params: np.ndarray) -> float:
        """
        Vectorised implementation of

            L(θ) = − Σ_r Σ_{m=0}^{n−2} log  exp(θ_{π_r[m]}) /
                                                Σ_{j≥m} exp(θ_{π_r[j]})

        where π_r is ranking r.
        """
        exp_theta   = np.exp(params)                   # (n_items,)
        exp_utils   = exp_theta[self.rankings]         # (R , n)

        # Denominator: reverse cumulative sums so that
        # denom[r, m] = Σ_{j=m}^{n−1} exp(θ_{π_r[j]})
        denom = np.flip(np.cumsum(np.flip(exp_utils, axis=1), axis=1), axis=1)

        # Exclude the last deterministic choice (prob = 1)
        numer       = exp_utils[:, :-1]
        denom       = denom[:, :-1]

        nll_matrix  = np.log(denom) - np.log(numer)    # (R , n−1)
        return nll_matrix.sum()                       # scalar

    # -----------------------------
    #  MLE fit
    # -----------------------------
    def fit(self, initial_guess: np.ndarray | None = None) -> np.ndarray:
        if initial_guess is None:
            initial_guess = np.zeros(self.n_items)

        # Simple progress print-out
        step = {'i': 0}
        def cb(xk):
            step['i'] += 1
            print(f"Iter {step['i']:3d}  NLL = {self.negative_log_likelihood(xk):.4f}")

        res = minimize(
            self.negative_log_likelihood,
            initial_guess,
            method='L-BFGS-B',
            callback=cb,
            options={'maxiter': 1000, 'disp': False}
        )
        return res.x

    # -----------------------------
    #  Vectorised test NLL
    # -----------------------------
    @staticmethod
    def _matrix_nll(perms: np.ndarray, utilities: np.ndarray) -> float:
        """Shared helper used by both train- and test-NLL routines."""
        exp_theta = np.exp(utilities)
        exp_utils = exp_theta[perms]
        denom = np.flip(np.cumsum(np.flip(exp_utils, axis=1), axis=1), axis=1)
        nll = np.log(denom[:, :-1]) - np.log(exp_utils[:, :-1])
        return nll.sum()

    def compute_normalized_nll(
        self, permutations_test: np.ndarray, estimated_utilities: np.ndarray
    ) -> float:
        total_nll = self._matrix_nll(permutations_test, estimated_utilities)
        return total_nll / permutations_test.shape[0]


# ----------------------------------------------------------------------
#  Fully-vectorised Plackett–Luce sampler (Gumbel–Max)
# ----------------------------------------------------------------------
def sample_PL(utilities: np.ndarray, n_samples: int = 1000, rng=None) -> np.ndarray:
    """
    Draw `n_samples` permutations *simultaneously* using the Gumbel–Max trick.

    Each row is a ranking from best (col 0) to worst (last col).
    """
    rng = rng or np.random.default_rng()
    gumbels   = rng.gumbel(size=(n_samples, len(utilities)))
    scores    = utilities + gumbels                      # broadcast add
    return np.argsort(-scores, axis=1).astype(np.int64)  # descending ⇒ best first
