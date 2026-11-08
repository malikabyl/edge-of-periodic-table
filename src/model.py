"""Physics baseline (liquid drop + optional shell term) + Gaussian-process correction.

Target: total nuclear binding energy B(Z, N) in MeV, measured AME2020 values only.
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

MAGIC_N = [0, 2, 8, 20, 28, 50, 82, 126, 184, 258]
MAGIC_Z = {
    "114": [0, 2, 8, 20, 28, 50, 82, 114, 164],   # proton shell closure at Z=114
    "126": [0, 2, 8, 20, 28, 50, 82, 126, 184],   # proton shell closure at Z=126
}


def F_shell(n, magic):
    """Myers–Swiatecki shell function: zero at magic numbers, negative mid-shell."""
    n = np.asarray(n, float)
    out = np.zeros_like(n)
    for lo, hi in zip(magic[:-1], magic[1:]):
        m = (n > lo) & (n < hi)
        q = 0.6 * (hi ** (5 / 3) - lo ** (5 / 3)) / (hi - lo)
        out[m] = q * (n[m] - lo) - (n[m] ** (5 / 3) - lo ** (5 / 3))
    return out


def design(Z, N, shell=None):
    Z = np.asarray(Z, float); N = np.asarray(N, float); A = Z + N
    pair = np.where((Z % 2 == 0) & (N % 2 == 0), 1, np.where((Z % 2 == 1) & (N % 2 == 1), -1, 0))
    cols = [A, -A ** (2 / 3), -Z * (Z - 1) / A ** (1 / 3), -(N - Z) ** 2 / A,
            pair / np.sqrt(A), -np.abs(N - Z) / A]
    if shell is not None:
        S = (F_shell(N, MAGIC_N) + F_shell(Z, MAGIC_Z[shell])) / (A / 2) ** (2 / 3)
        cols += [S, A ** (1 / 3)]
    return np.column_stack(cols)


class MassModel:
    def __init__(self, shell=None, use_gp=True):
        self.shell, self.use_gp = shell, use_gp

    def fit(self, Z, N, B):
        X = design(Z, N, self.shell)
        self.coef, *_ = np.linalg.lstsq(X, B, rcond=None)
        r = B - X @ self.coef
        if self.use_gp:
            k = C(1.0, (1e-2, 1e2)) * RBF([3.0, 3.0], (0.5, 50)) + WhiteKernel(0.1, (1e-4, 10))
            self.gp = GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=2,
                                               random_state=0)
            self.gp.fit(np.column_stack([Z, N]), r)
            # predictive std of the latent function, excluding the white-noise floor
            self.noise = self.gp.kernel_.k2.noise_level
        return self

    def baseline(self, Z, N):
        return design(Z, N, self.shell) @ self.coef

    def predict(self, Z, N):
        b = self.baseline(Z, N)
        if not self.use_gp:
            return b, np.zeros_like(b)
        m, s = self.gp.predict(np.column_stack([Z, N]), return_std=True)
        return b + m, np.sqrt(np.maximum(s ** 2 - self.noise, 0) + self.noise)

    def sample(self, Z, N, n=300, seed=0):
        b = self.baseline(Z, N)
        if not self.use_gp:
            return np.tile(b, (n, 1))
        y = self.gp.sample_y(np.column_stack([Z, N]), n, random_state=seed).T
        return b + y


B_HE4 = 28.29566  # MeV


def q_alpha(Bparent, Bdaughter):
    return Bdaughter + B_HE4 - Bparent
