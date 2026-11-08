"""Interpolation vs extrapolation tests for each model variant."""
import json
import numpy as np
from load import load_ame
from model import MassModel

a = load_ame("../data/mass_1.mas20.txt")
exp = a[(~a.estimated) & a.B.notna() & (a.A >= 16)].reset_index(drop=True)
Z, N, B = exp.Z.values, exp.N.values, exp.B.values

rng = np.random.default_rng(0)
rand_test = rng.random(len(exp)) < 0.1

# neutron-rich edge: for each Z, the 2 most neutron-rich measured isotopes
edge = np.zeros(len(exp), bool)
for z, g in exp.groupby("Z"):
    edge[g.sort_values("N").index[-2:]] = True

heavy = Z >= 99  # everything heavier than einsteinium

variants = {
    "LDM": dict(shell=None, use_gp=False),
    "LDM+GP": dict(shell=None, use_gp=True),
    "LDM+shell(Z114)": dict(shell="114", use_gp=False),
    "LDM+shell(Z114)+GP": dict(shell="114", use_gp=True),
    "LDM+shell(Z126)+GP": dict(shell="126", use_gp=True),
}

res = {}
for name, kw in variants.items():
    res[name] = {}
    for split, test in [("random", rand_test), ("n-rich edge", edge), ("Z>=99", heavy)]:
        m = MassModel(**kw).fit(Z[~test], N[~test], B[~test])
        p, s = m.predict(Z[test], N[test])
        err = p - B[test]
        cover = float(np.mean(np.abs(err) < 2 * s)) if kw["use_gp"] else None
        res[name][split] = dict(rms=float(np.sqrt(np.mean(err ** 2))), n=int(test.sum()),
                                mean_sigma=float(s.mean()), cover2s=cover)
        print(f"{name:22s} {split:12s} n={test.sum():4d} RMS={res[name][split]['rms']:.3f} MeV"
              f"  sigma={s.mean():.3f}  2σ-coverage={cover}")
json.dump(res, open("../results/eval.json", "w"), indent=1)
