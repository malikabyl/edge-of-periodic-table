"""Viola–Seaborg alpha half-life law fitted on measured data (shared by later steps)."""

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



VS_RMS = vs_rms
