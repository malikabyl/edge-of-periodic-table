"""Step 2: teach the mass model measured alpha-decay energies.

A measured Q-alpha is a constraint on a DIFFERENCE of masses:
    B(Z,N) - B(Z-2,N-2) = B(4He) - Q_alpha
Superheavy chains give many such differences even where no absolute mass is known.
A Gaussian process handles this exactly: observations can be any linear combination of the latent function.
Kernel hyperparameters are the ones fitted earlier on masses only (no re-tuning on the test data).
"""
import json
import numpy as np
import pandas as pd
from load import load_ame, load_nubase
from model import design, B_HE4

a = load_ame("../data/mass_1.mas20.txt")
nb = load_nubase("../data/nubase_4.mas20.txt")
exp = a[(~a.estimated) & a.B.notna() & (a.A >= 16)].reset_index(drop=True)
meas = {(r.Z, r.N) for r in exp.itertuples()}

# ---- measured Q-alpha from AME2020 rct1 ----
rows = []
for l in open("../data/rct1.txt"):
    if len(l) < 80 or not l[1:4].strip().isdigit():
        continue
    try:
        A, Z = int(l[1:4]), int(l[8:11])
    except ValueError:
        continue
    v, u = l[56:68].strip(), l[68:78].strip()
    if not v or "#" in v or v == "*":
        continue
    rows.append(dict(Z=Z, N=A - Z, A=A, Q=float(v) / 1000, dQ=float(u) / 1000))
qa = pd.DataFrame(rows)
# keep only decays that add information: parent or daughter mass not measured
qa = qa[[((r.Z, r.N) not in meas) or ((r.Z - 2, r.N - 2) not in meas) for r in qa.itertuples()]]
qa = qa[qa.Z >= 90].reset_index(drop=True)
print(f"measured Q-alpha adding new information: {len(qa)}  (Z range {qa.Z.min()}–{qa.Z.max()})")
print(qa[qa.Z >= 113].to_string())

# ---- baseline (liquid drop), same as before ----
X = design(exp.Z.values, exp.N.values)
coef, *_ = np.linalg.lstsq(X, exp.B.values, rcond=None)
base = lambda Z, N: design(np.atleast_1d(Z), np.atleast_1d(N)) @ coef

AMP2, LZ, LN, WN = 3.62 ** 2, 3.0, 3.36, 0.105  # from the earlier fit on masses only


def kern(P, Q):
    d2 = ((P[:, None, 0] - Q[None, :, 0]) / LZ) ** 2 + ((P[:, None, 1] - Q[None, :, 1]) / LN) ** 2
    K = AMP2 * np.exp(-0.5 * d2)
    same = (P[:, None, 0] == Q[None, :, 0]) & (P[:, None, 1] == Q[None, :, 1])
    return K + WN * same


class DiffGP:
    """Latent g(Z,N) = B - baseline. Observations: rows of A (sparse) applied to g at nodes U."""

    def __init__(self, mass_df, qa_df):
        nodes = {}
        def idx(z, n):
            return nodes.setdefault((z, n), len(nodes))
        obs = []  # (list of (node, coef), y, var)
        for r in mass_df.itertuples():
            obs.append(([(idx(r.Z, r.N), 1.0)], r.B - base(r.Z, r.N)[0], 1e-6))
        for r in qa_df.itertuples():
            p, d = (r.Z, r.N), (r.Z - 2, r.N - 2)
            y = (B_HE4 - r.Q) - (base(*p)[0] - base(*d)[0])
            obs.append(([(idx(*p), 1.0), (idx(*d), -1.0)], y, r.dQ ** 2))
        self.nodes = np.array(list(nodes.keys()), float)
        A = np.zeros((len(obs), len(nodes)))
        for i, (terms, _, _) in enumerate(obs):
            for j, c in terms:
                A[i, j] += c
        self.A = A
        self.y = np.array([o[1] for o in obs]); var = np.array([o[2] for o in obs])
        Kuu = kern(self.nodes, self.nodes)
        Koo = A @ Kuu @ A.T + np.diag(var)
        self.L = np.linalg.cholesky(Koo + 1e-8 * np.eye(len(Koo)))
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))

    def posterior(self, Z, N):
        G = np.column_stack([Z, N]).astype(float)
        Kgo = kern(G, self.nodes) @ self.A.T
        mu = Kgo @ self.alpha
        V = np.linalg.solve(self.L, Kgo.T)
        cov = kern(G, G) - V.T @ V
        return base(Z, N) + mu, cov

    def q_alpha(self, Z, N, n_samples=0, seed=0):
        """Q-alpha posterior (mean, sd) for parents (Z,N); optional joint samples."""
        Z = np.asarray(Z); N = np.asarray(N)
        PZ = np.concatenate([Z, Z - 2]); PN = np.concatenate([N, N - 2])
        mu, cov = self.posterior(PZ, PN)
        k = len(Z)
        D = np.hstack([-np.eye(k), np.eye(k)])  # Q = B_d + B_He - B_p
        qm = D @ mu + B_HE4
        qc = D @ cov @ D.T
        sd = np.sqrt(np.maximum(np.diag(qc), 0))
        if n_samples:
            rng = np.random.default_rng(seed)
            Ls = np.linalg.cholesky(qc + 1e-9 * np.eye(k))
            S = qm[None, :] + rng.standard_normal((n_samples, k)) @ Ls.T
            return qm, sd, S
        return qm, sd


if __name__ == "__main__":
    # ---------- fair test: hide all Q-alpha of Lv, Ts, Og (Z>=116), predict them ----------
    test = qa[qa.Z >= 116]
    res = {}
    for name, train_qa in [("masses only", qa.iloc[0:0]), ("masses + Qalpha up to Z=115", qa[qa.Z < 116])]:
        gp = DiffGP(exp, train_qa)
        qm, sd = gp.q_alpha(test.Z.values, test.N.values)
        err = qm - test.Q.values
        res[name] = dict(rms=float(np.sqrt(np.mean(err ** 2))), sigma=float(sd.mean()),
                         cover=float(np.mean(np.abs(err) < 2 * sd)),
                         rows=[dict(Z=int(z), A=int(z + n), meas=float(q), pred=float(m), sd=float(s))
                               for z, n, q, m, s in zip(test.Z, test.N, test.Q, qm, sd)])
        print(f"{name:28s} hidden Qα (Z>=116, n={len(test)}): RMS={res[name]['rms']:.2f} MeV  "
              f"σ≈{sd.mean():.2f}  within 2σ={res[name]['cover']:.0%}")
    json.dump(dict(test=res, n_qa=len(qa), z_range=[int(qa.Z.min()), int(qa.Z.max())]),
              open("../results/qa_test.json", "w"))
