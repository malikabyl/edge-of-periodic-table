import json, numpy as np
from qalpha import DiffGP, exp, qa, nb
from predict_helpers import logT_alpha, VS_RMS

# 1) error vs extrapolation distance: hide Q-alpha of all Z>=cut, predict them
curve = []
for cut in [113, 114, 115, 116, 117, 118]:
    gp = DiffGP(exp, qa[qa.Z < cut]); t = qa[qa.Z >= cut]
    qm, sd = gp.q_alpha(t.Z.values, t.N.values)
    for z, n, q, m, s in zip(t.Z, t.N, t.Q, qm, sd):
        curve.append(dict(cut=cut, Z=int(z), A=int(z+n), dist=int(z - cut + 1), err=float(m - q), sd=float(s)))
c = {}
for r in curve: c.setdefault(r["dist"], []).append(r)
print("distance beyond last trained Z -> RMS error, model sigma, n")
for d in sorted(c):
    e = np.array([r["err"] for r in c[d]]); s = np.array([r["sd"] for r in c[d]])
    print(d, round(float(np.sqrt((e**2).mean())), 3), round(float(s.mean()), 2), len(e))

# 2) final model: all masses + all Q-alpha
gp = DiffGP(exp, qa)
targets = json.load(open("../results/predict.json"))["targets"]
out = []
for t in targets:
    z, n = t["Z"], t["N"]
    qm, sd, S = gp.q_alpha([z], [n], n_samples=4000, seed=1)
    Ts = logT_alpha(np.full(4000, z), np.full(4000, n), S[:, 0]) + np.random.default_rng(2).normal(0, VS_RMS, 4000)
    old = t["LDM+GP"]
    row = dict(Z=z, N=n, label=t["label"], Q=float(qm[0]), sQ=float(sd[0]),
               logT=float(np.median(Ts)), lo=float(np.percentile(Ts, 16)), hi=float(np.percentile(Ts, 84)),
               old=old)
    out.append(row)
    print(f"{t['label']:40s} Qα {old['Q']:.2f}±{old['sQ']:.2f} -> {qm[0]:.2f}±{sd[0]:.2f}   "
          f"logT {old['lo']:.1f}..{old['hi']:.1f} -> {row['lo']:.1f}..{row['hi']:.1f} (median {row['logT']:.1f})")
# 3) "calibrated" corridor: Q-alpha error as large as the model's REAL error on hidden chains (~0.25 MeV)
rng = np.random.default_rng(3)
for t in out:
    Q = rng.normal(t["Q"], 0.25, 20000)
    T = logT_alpha(np.full(20000, t["Z"]), np.full(20000, t["N"]), Q) + rng.normal(0, VS_RMS, 20000)
    t["cal"] = dict(sQ=0.25, logT=float(np.median(T)), lo=float(np.percentile(T, 16)), hi=float(np.percentile(T, 84)))
json.dump(dict(curve=curve, targets=out), open("../results/qa_final.json", "w"))
