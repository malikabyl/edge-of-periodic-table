"""Fission as a Gaussian process (simple formulas fail), alpha law refit, end-to-end blind test, final predictions."""
import json, numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import halflife as H

al, sf, Law, f_vs = H.al, H.sf, H.Law, H.f_vs

class SFGP:
    def fit(self, d):
        Xp = np.column_stack([np.ones(len(d)), d.Z % 2, d.N % 2])
        self.p, *_ = np.linalg.lstsq(Xp, d.logT.values, rcond=None)
        r = d.logT.values - Xp @ self.p
        k = C(4.0, (0.1, 400)) * RBF([4.0, 6.0], (1, 60)) + WhiteKernel(1.0, (1e-2, 20))
        self.gp = GaussianProcessRegressor(k, n_restarts_optimizer=3, random_state=0).fit(np.column_stack([d.Z, d.N]), r)
        return self
    def predict(self, Z, N):
        Z, N = np.atleast_1d(Z), np.atleast_1d(N)
        m, s = self.gp.predict(np.column_stack([Z, N]), return_std=True)
        return np.column_stack([np.ones(len(Z)), Z % 2, N % 2]) @ self.p + m, s

# 1) fission GP: hide everything from Z>=cut
sf_curve = []
for cut in [106, 108, 110, 112]:
    tr, te = sf[sf.Z < cut], sf[sf.Z >= cut]
    g = SFGP().fit(tr); m, s = g.predict(te.Z.values, te.N.values)
    e = m - te.logT.values
    sf_curve.append(dict(cut=cut, n=len(te), rms=float(np.sqrt(np.mean(e**2))), sigma=float(s.mean()),
                         cover=float(np.mean(np.abs(e) < 2*s)), const=float(np.sqrt(np.mean((tr.logT.mean()-te.logT.values)**2)))))
    print("SF GP hide Z>=%d: n=%d RMS=%.2f sigma=%.2f within2σ=%.0f%% (constant: %.2f)" % (cut, len(te), sf_curve[-1]["rms"], s.mean(), 100*sf_curve[-1]["cover"], sf_curve[-1]["const"]))

# 2) end-to-end blind test on Lv, Ts, Og: everything trained on Z<116 only
qt = json.load(open("../results/qa_test.json"))["test"]["masses + Qalpha up to Z=115"]["rows"]
La = Law(f_vs, False).fit(al[al.Z < 116]); g = SFGP().fit(sf[sf.Z < 116])
a_rms = H.alpha_res["Viola-Seaborg"]["test_rms"]
rng = np.random.default_rng(0)
nbT = {(r.Z, r.A): np.log10(r.T) for r in H.nb.itertuples() if not r.T_est and np.isfinite(r.T)}
e2e = []
for r in qt:
    Z, A = r["Z"], r["A"]; N = A - Z
    Q = rng.normal(r["pred"], 0.25, 20000)
    la = La(np.full(20000, Z), np.full(20000, N), Q) + rng.normal(0, a_rms, 20000)
    ms, ss = g.predict(Z, N); ls = rng.normal(ms[0], ss[0], 20000)
    tot = -np.log10(10.0**-la + 10.0**-ls)
    meas = nbT.get((Z, A), np.nan)
    e2e.append(dict(Z=Z, A=A, meas=float(meas), pred=float(np.median(tot)), lo=float(np.percentile(tot, 16)),
                    hi=float(np.percentile(tot, 84)), p_alpha=float(np.mean(la < ls))))
    print(f"blind {Z}-{A}: measured 10^{meas:.2f} s, predicted 10^{np.median(tot):.2f} (68%: {np.percentile(tot,16):.2f}..{np.percentile(tot,84):.2f}), P(alpha first)={np.mean(la<ls):.2f}")
ee = np.array([x["pred"] - x["meas"] for x in e2e])
print("end-to-end RMS (dex):", round(float(np.sqrt(np.mean(ee**2))), 2), " inside 68% band:", np.mean([x["lo"] <= x["meas"] <= x["hi"] for x in e2e]))

# 3) final: everything trained on all data
La = Law(f_vs, False).fit(al); g = SFGP().fit(sf)
F = json.load(open("../results/qa_final.json"))
final = []
for t in F["targets"]:
    Z, N = t["Z"], t["N"]
    sQ = 0.25 if N < 184 else t["sQ"]
    Q = rng.normal(t["Q"], sQ, 40000)
    la = La(np.full(40000, Z), np.full(40000, N), Q) + rng.normal(0, a_rms, 40000)
    ms, ss = g.predict(Z, N); ls = rng.normal(ms[0], ss[0], 40000)
    tot = -np.log10(10.0**-la + 10.0**-ls)
    row = dict(label=t["label"], Z=Z, N=N, Q=t["Q"], sQ=sQ,
               a=[float(np.percentile(la, p)) for p in (16, 50, 84)],
               sf=[float(ms[0] - ss[0]), float(ms[0]), float(ms[0] + ss[0])],
               tot=[float(np.percentile(tot, p)) for p in (16, 50, 84)], p_alpha=float(np.mean(la < ls)),
               prev=t["cal"])
    final.append(row)
    print(f"{t['label']:42s} α 10^{row['a'][1]:.1f}  SF 10^{row['sf'][1]:.1f}±{ss[0]:.1f}  total 10^{row['tot'][1]:.1f} ({row['tot'][0]:.1f}..{row['tot'][2]:.1f})  P(α)={row['p_alpha']:.2f}")
json.dump(dict(alpha=H.alpha_res, sf_formulas=H.sf_res, sf_curve=sf_curve, e2e=e2e,
               e2e_rms=float(np.sqrt(np.mean(ee**2))), final=final), open("../results/decay_final.json", "w"))
