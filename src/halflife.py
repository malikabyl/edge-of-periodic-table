"""Step 3: better alpha-decay half-life law + spontaneous fission.
Rule for every test: fit on Z < 104, judge on measured superheavies Z >= 104 the formula never saw."""
import json, re
import numpy as np
import pandas as pd
from load import load_nubase

nb = load_nubase("../data/nubase_4.mas20.txt")

# ---------- measured Q-alpha for every nucleus (AME2020 rct1) ----------
qrows = []
for l in open("../data/rct1.txt"):
    if len(l) < 80 or not l[1:4].strip().isdigit():
        continue
    try:
        A, Z = int(l[1:4]), int(l[8:11])
    except ValueError:
        continue
    v = l[56:68].strip()
    if v and "#" not in v and v != "*":
        qrows.append(dict(Z=Z, N=A - Z, Q=float(v) / 1000))
Qexp = pd.DataFrame(qrows)

# ---------- partial half-lives ----------
def branch(s, mode):
    m = re.search(r"(?:^|;)" + mode + r"\s*([=~])\s*([\d.]+(?:e[-+]?\d+)?)", s)
    return float(m.group(2)) / 100 if m else np.nan

nb["sf_frac"] = nb.decay.apply(lambda s: branch(s, "SF"))
ok = (~nb.T_est) & np.isfinite(nb["T"]) & (nb["T"] > 0)
al = nb[ok & (nb.alpha_frac > 0)].merge(Qexp, on=["Z", "N"])
al = al[(al.Z >= 82) & (al.Q > 0.5)].copy()
al["logT"] = np.log10(al["T"] / al.alpha_frac)
sf = nb[ok & (nb.sf_frac > 0) & (nb.Z >= 90)].copy()
sf["logT"] = np.log10(sf["T"] / sf.sf_frac)
print(f"alpha data: {len(al)} (Z>=104: {(al.Z >= 104).sum()}),  SF data: {len(sf)} (Z>=104: {(sf.Z >= 104).sum()})")

cls = lambda Z, N: (np.asarray(Z) % 2) * 2 + (np.asarray(N) % 2)  # 0 ee, 1 e-o(odd N), 2 o-e, 3 oo


# ---------- alpha formulas ----------
def f_vs(Z, N, Q):
    Z, N, Q = map(lambda v: np.asarray(v, float), (Z, N, Q))
    return np.column_stack([Z / np.sqrt(Q), 1 / np.sqrt(Q), Z, np.ones_like(Z), Z % 2, N % 2])

def f_royer(Z, N, Q):
    Z, N, Q = map(lambda v: np.asarray(v, float), (Z, N, Q)); A = Z + N
    return np.column_stack([np.ones_like(Z), A ** (1 / 6) * np.sqrt(Z), Z / np.sqrt(Q)])

def f_udl(Z, N, Q):
    Z, N, Q = map(lambda v: np.asarray(v, float), (Z, N, Q))
    Zd, Ad = Z - 2, Z + N - 4; mu = 4 * Ad / (4 + Ad)
    return np.column_stack([2 * Zd * np.sqrt(mu / Q), np.sqrt(mu * 2 * Zd * (Ad ** (1 / 3) + 4 ** (1 / 3))), np.ones_like(Z)])

class Law:
    def __init__(self, feat, per_class):
        self.feat, self.per_class = feat, per_class
    def fit(self, d):
        if self.per_class:
            self.c = {}
            for k in range(4):
                g = d[cls(d.Z, d.N) == k]
                self.c[k], *_ = np.linalg.lstsq(self.feat(g.Z, g.N, g.Q), g.logT, rcond=None)
        else:
            self.c, *_ = np.linalg.lstsq(self.feat(d.Z, d.N, d.Q), d.logT, rcond=None)
        return self
    def __call__(self, Z, N, Q):
        Z, N, Q = np.atleast_1d(Z), np.atleast_1d(N), np.maximum(np.atleast_1d(Q), 0.5)
        X = self.feat(Z, N, Q)
        if not self.per_class:
            return X @ self.c
        k = cls(Z, N)
        return np.array([X[i] @ self.c[k[i]] for i in range(len(Z))])

laws = {"Viola-Seaborg": (f_vs, False), "Viola-Seaborg per parity class": (lambda Z, N, Q: f_vs(Z, N, Q)[:, :4], True),
        "Royer (2000)": (f_royer, True), "Universal decay law UDL (2009)": (f_udl, True)}
train, test = al[al.Z < 104], al[al.Z >= 104]
alpha_res = {}
for name, (f, pc) in laws.items():
    L = Law(f, pc).fit(train)
    e_tr = L(train.Z.values, train.N.values, train.Q.values) - train.logT.values
    e_te = L(test.Z.values, test.N.values, test.Q.values) - test.logT.values
    alpha_res[name] = dict(train_rms=float(np.sqrt(np.mean(e_tr ** 2))), test_rms=float(np.sqrt(np.mean(e_te ** 2))),
                           test_bias=float(e_te.mean()), n_train=len(train), n_test=len(test))
    print(f"alpha {name:32s} train RMS {alpha_res[name]['train_rms']:.2f}  superheavy test RMS {alpha_res[name]['test_rms']:.2f}  bias {e_te.mean():+.2f}  (n={len(test)})")

best_alpha = min(alpha_res, key=lambda k: alpha_res[k]["test_rms"])
print("best alpha law:", best_alpha)

# ---------- spontaneous fission ----------
def f_sf(Z, N):
    Z, N = np.asarray(Z, float), np.asarray(N, float); A = Z + N
    x = Z ** 2 / A; I = (N - Z) / A
    return np.column_stack([np.ones_like(Z), x, x ** 2, I, I ** 2, (A % 2), ((Z % 2) * (N % 2))])

def f_sf_shell(Z, N):
    # same + a smooth bump for the deformed shell gaps N=152 and N=162 seen in actinides/transactinides
    Z, N = np.asarray(Z, float), np.asarray(N, float)
    return np.column_stack([f_sf(Z, N), np.exp(-((N - 152) / 4) ** 2), np.exp(-((N - 162) / 4) ** 2)])

sf_res, sf_models = {}, {}
trs, tes = sf[sf.Z < 104], sf[sf.Z >= 104]
for name, f in [("fission: Z^2/A + isospin", f_sf), ("fission: + shells N=152,162", f_sf_shell)]:
    c, *_ = np.linalg.lstsq(f(trs.Z, trs.N), trs.logT, rcond=None)
    sf_models[name] = (f, c)
    e_tr = f(trs.Z, trs.N) @ c - trs.logT.values
    e_te = f(tes.Z, tes.N) @ c - tes.logT.values
    sf_res[name] = dict(train_rms=float(np.sqrt(np.mean(e_tr ** 2))), test_rms=float(np.sqrt(np.mean(e_te ** 2))),
                        test_bias=float(e_te.mean()), n_train=len(trs), n_test=len(tes))
    print(f"SF {name:36s} train RMS {sf_res[name]['train_rms']:.2f}  superheavy test RMS {sf_res[name]['test_rms']:.2f}  bias {e_te.mean():+.2f} (n={len(tes)})")

# naive baseline for SF: predict the training mean
e = trs.logT.mean() - tes.logT.values
print("SF baseline (one constant)        test RMS", round(float(np.sqrt(np.mean(e ** 2))), 2))
sf_res["no formula (training mean)"] = dict(test_rms=float(np.sqrt(np.mean(e ** 2))), n_test=len(tes))

json.dump(dict(alpha=alpha_res, best_alpha=best_alpha, sf=sf_res), open("../results/halflife.json", "w"))
