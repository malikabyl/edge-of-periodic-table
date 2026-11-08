"""Train on all measured masses, predict the superheavy region, turn masses into alpha half-lives,
and test those half-lives against measured superheavy half-lives (which were never used in training)."""
import json, pickle
import numpy as np
import pandas as pd
from load import load_ame, load_nubase
from model import MassModel, q_alpha

a = load_ame("../data/mass_1.mas20.txt")
nb = load_nubase("../data/nubase_4.mas20.txt")
exp = a[(~a.estimated) & a.B.notna() & (a.A >= 16)]
Bexp = {(r.Z, r.N): r.B for r in exp.itertuples()}
Ball = {(r.Z, r.N): r.B for r in a.itertuples() if np.isfinite(r.B)}  # incl. AME '#' systematics

# ---------- 1. Viola–Seaborg alpha half-life law, fitted on measured data ----------
rows = []
for r in nb.itertuples():
    if r.Z < 84 or not np.isfinite(r.T) or r.T_est or not (r.alpha_frac > 0.01):
        continue
    p, dgt = (r.Z, r.N), (r.Z - 2, r.N - 2)
    if p in Bexp and dgt in Bexp:
        Q = q_alpha(Bexp[p], Bexp[dgt])
        if Q > 0:
            rows.append(dict(Z=r.Z, N=r.N, Q=Q, logT=np.log10(r.T / r.alpha_frac)))
vs = pd.DataFrame(rows)


def vs_design(Z, N, Q):
    Z = np.asarray(Z, float); N = np.asarray(N, float); Q = np.asarray(Q, float)
    return np.column_stack([Z / np.sqrt(Q), 1 / np.sqrt(Q), Z, np.ones_like(Z), Z % 2, N % 2])


vs_coef, *_ = np.linalg.lstsq(vs_design(vs.Z, vs.N, vs.Q), vs.logT, rcond=None)
vs_rms = float(np.sqrt(np.mean((vs_design(vs.Z, vs.N, vs.Q) @ vs_coef - vs.logT) ** 2)))
print(f"Viola–Seaborg fit on {len(vs)} measured alpha emitters: RMS = {vs_rms:.2f} in log10(T)")


def logT_alpha(Z, N, Q):
    Q = np.maximum(Q, 0.5)
    return vs_design(Z, N, Q) @ vs_coef


# ---------- 2. models trained on ALL measured masses ----------
variants = {"LDM+GP": dict(shell=None), "shell Z=114 +GP": dict(shell="114"), "shell Z=126 +GP": dict(shell="126")}
models = {k: MassModel(**v).fit(exp.Z.values, exp.N.values, exp.B.values) for k, v in variants.items()}
for k, m in models.items():
    print(k, m.gp.kernel_)

grid = [(z, n) for z in range(92, 127) for n in range(136, 201)]
gZ = np.array([g[0] for g in grid]); gN = np.array([g[1] for g in grid])
idx = {g: i for i, g in enumerate(grid)}

out = {"grid_Z": [92, 126], "grid_N": [136, 200], "variants": {}}
test_rows = []
for k, m in models.items():
    mu, sd = m.predict(gZ, gN)
    S = m.sample(gZ, gN, n=400, seed=1)  # joint posterior samples, correlated across nuclei
    cells = []
    for (z, n), i in idx.items():
        j = idx.get((z - 2, n - 2))
        if j is None:
            continue
        Qs = S[:, j] + 28.29566 - S[:, i]
        Ts = logT_alpha(np.full(len(Qs), z), np.full(len(Qs), n), Qs)
        cells.append(dict(Z=z, N=n, B=round(mu[i], 2), sB=round(sd[i], 2),
                          Q=round(float(np.median(Qs)), 2), sQ=round(float(np.std(Qs)), 2),
                          logT=round(float(np.median(Ts)), 2),
                          logT_lo=round(float(np.percentile(Ts, 16)), 2),
                          logT_hi=round(float(np.percentile(Ts, 84)), 2)))
    out["variants"][k] = cells
    cmap = {(c["Z"], c["N"]): c for c in cells}

    # ---------- 3. honest test: measured superheavy half-lives (Z>=104), never seen in training ----------
    for r in nb.itertuples():
        if r.Z < 104 or r.T_est or not np.isfinite(r.T) or not (r.alpha_frac > 0.5):
            continue
        c = cmap.get((r.Z, r.N))
        if c is None:
            continue
        # AME's own expert systematics as a reference
        ame_Q = q_alpha(Ball[(r.Z, r.N)], Ball[(r.Z - 2, r.N - 2)]) if (r.Z, r.N) in Ball and (r.Z - 2, r.N - 2) in Ball else np.nan
        test_rows.append(dict(model=k, Z=r.Z, N=r.N, A=r.A, logT_meas=np.log10(r.T / r.alpha_frac),
                              logT_pred=c["logT"], lo=c["logT_lo"], hi=c["logT_hi"], Q_pred=c["Q"],
                              Q_ame=ame_Q,
                              logT_ame=float(logT_alpha([r.Z], [r.N], [ame_Q])[0]) if np.isfinite(ame_Q) else np.nan))

tr = pd.DataFrame(test_rows)
summary = {}
for k, g in tr.groupby("model"):
    err = g.logT_pred - g.logT_meas
    inside = ((g.logT_meas >= g.lo - vs_rms) & (g.logT_meas <= g.hi + vs_rms)).mean()
    summary[k] = dict(n=len(g), rms=float(np.sqrt(np.mean(err ** 2))), bias=float(err.mean()),
                      within_band=float(inside))
    print(f"{k:18s} superheavy half-life test n={len(g)} RMS={summary[k]['rms']:.2f} dex  bias={err.mean():+.2f}  in band={inside:.0%}")
g = tr[tr.model == list(models)[0]].dropna(subset=["logT_ame"])
summary["AME2020 systematics"] = dict(n=len(g), rms=float(np.sqrt(np.mean((g.logT_ame - g.logT_meas) ** 2))),
                                      bias=float((g.logT_ame - g.logT_meas).mean()))
print("AME2020 expert systematics     RMS=%.2f dex" % summary["AME2020 systematics"]["rms"])

# targets of current experiments
targets = {(119, 176): "295-119 (Bk-249 + Ti-50, 4n)", (119, 177): "296-119 (Bk-249 + Ti-50, 3n)",
           (120, 175): "295-120 (Cf-249 + Ti-50, 4n)", (120, 176): "296-120 (Cf-249 + Ti-50, 3n)",
           (120, 178): "298-120 (Cm-248 + Cr-54, 4n)", (120, 179): "299-120 (Cm-248 + Cr-54, 3n)",
           (118, 176): "294-Og (known, reference)", (114, 184): "298-Fl (Z=114, N=184 — 'island' centre?)",
           (120, 184): "304-120 (N=184)"}
tg = []
for (z, n), lab in targets.items():
    row = dict(Z=z, N=n, label=lab)
    for k in models:
        c = {(c["Z"], c["N"]): c for c in out["variants"][k]}[(z, n)]
        row[k] = dict(Q=c["Q"], sQ=c["sQ"], logT=c["logT"], lo=c["logT_lo"], hi=c["logT_hi"], sB=c["sB"])
    tg.append(row)
    print(lab, {k: (row[k]["Q"], row[k]["sQ"], row[k]["logT"], row[k]["lo"], row[k]["hi"]) for k in models})

out.update(vs=dict(n=len(vs), rms=vs_rms), test=tr.round(3).to_dict("records"), summary=summary, targets=tg,
           measured=[dict(Z=int(z), N=int(n)) for (z, n) in Bexp if z >= 92 and n >= 136],
           eval=json.load(open("../results/eval.json")))
json.dump(out, open("../results/predict.json", "w"), default=float)
