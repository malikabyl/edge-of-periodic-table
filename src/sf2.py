import json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import halflife as H
bar = pd.read_csv('../data/derived/barriers.csv')
sf = H.sf.merge(bar, on=['Z','N'], how='left')

def X(Z, N, B=None, kind='base'):
    Z, N = np.asarray(Z, float), np.asarray(N, float); A = Z + N; x = Z**2/A
    cols = [np.ones(len(Z)), A % 2, (Z % 2)*(N % 2), x, x**2]
    if kind == 'iso': I = (N-Z)/A; cols += [I, I**2]
    if kind == 'bar': cols += [B]
    return np.column_stack(cols)

class SF:
    def __init__(self, kind='base', gp=False): self.kind, self.use_gp = kind, gp
    def fit(self, d):
        Xd = X(d.Z, d.N, d.Bf_bskg3.values, self.kind)
        self.c, *_ = np.linalg.lstsq(Xd, d.logT.values, rcond=None)
        r = d.logT.values - Xd @ self.c
        self.res_sd = float(np.std(r))
        if self.use_gp:
            k = C(2.0, (0.05, 100)) * RBF([4.0, 6.0], (1, 60)) + WhiteKernel(1.0, (1e-2, 20))
            self.gp = GaussianProcessRegressor(k, n_restarts_optimizer=3, random_state=0).fit(np.column_stack([d.Z, d.N]), r)
        return self
    def predict(self, Z, N, B=None):
        Z, N = np.atleast_1d(Z), np.atleast_1d(N)
        m = X(Z, N, B, self.kind) @ self.c
        if self.use_gp:
            g, s = self.gp.predict(np.column_stack([Z, N]), return_std=True); return m + g, s
        return m, np.full(len(Z), self.res_sd)

variants = {"Z^2/A + parity": dict(kind='base'), "+ isospin (step 3)": dict(kind='iso'),
            "+ BSkG3 barrier (theory)": dict(kind='bar'), "Z^2/A + parity + GP correction": dict(kind='base', gp=True)}
res = {}
for name, kw in variants.items():
    res[name] = {}
    for cut in [104, 106, 108, 110]:
        tr, te = sf[sf.Z < cut], sf[sf.Z >= cut]
        m = SF(**kw).fit(tr); p, s = m.predict(te.Z.values, te.N.values, te.Bf_bskg3.values)
        e = p - te.logT.values
        res[name][cut] = dict(n=len(te), rms=float(np.sqrt(np.mean(e**2))), cover=float(np.mean(np.abs(e) < 2*s)))
    print(f"{name:42s}", "  ".join(f"Z≥{c}: {r['rms']:.2f} (2σ {r['cover']:.0%})" for c, r in res[name].items()))
const = {cut: float(np.sqrt(((sf[sf.Z<cut].logT.mean()-sf[sf.Z>=cut].logT)**2).mean())) for cut in [104,106,108,110]}
corr = dict(bskg3=float(np.corrcoef(sf.dropna(subset=['Bf_bskg3']).Bf_bskg3, sf.dropna(subset=['Bf_bskg3']).logT)[0,1]),
            hfb=float(np.corrcoef(sf.dropna(subset=['Bf_hfb']).Bf_hfb, sf.dropna(subset=['Bf_hfb']).logT)[0,1]))
print("const", const, "corr", corr)
json.dump(dict(res=res, const=const, corr=corr, n=len(sf)), open('../results/sf2.json','w'))
