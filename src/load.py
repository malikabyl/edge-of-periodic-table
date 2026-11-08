"""Parsers for AME2020 (masses) and NUBASE2020 (half-lives, decay modes)."""
import re
import numpy as np
import pandas as pd

M_N, M_H = 8071.31806, 7288.971064  # mass excess, keV


def _num(s):
    s = s.strip()
    if not s or s == "*":
        return np.nan, False
    est = "#" in s
    try:
        return float(s.replace("#", ".")), est
    except ValueError:
        return np.nan, est


def load_ame(path):
    rows = []
    with open(path) as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("1N-Z")) + 2
    for l in lines[start:]:
        if len(l) < 60:
            continue
        N, Z, A = int(l[4:9]), int(l[9:14]), int(l[14:19])
        el = l[20:23].strip()
        me, est = _num(l[28:42])
        dme, _ = _num(l[42:54])
        rows.append(dict(Z=Z, N=N, A=A, el=el, mass_excess=me, dmass=dme, estimated=est))
    d = pd.DataFrame(rows)
    # total binding energy, MeV
    d["B"] = (d.Z * M_H + d.N * M_N - d.mass_excess) / 1000.0
    d["dB"] = d.dmass / 1000.0
    return d


UNITS = {"ys": 1e-24, "zs": 1e-21, "as": 1e-18, "fs": 1e-15, "ps": 1e-12, "ns": 1e-9,
         "us": 1e-6, "ms": 1e-3, "s": 1, "m": 60, "h": 3600, "d": 86400,
         "y": 3.15576e7, "ky": 3.15576e10, "My": 3.15576e13, "Gy": 3.15576e16,
         "Ty": 3.15576e19, "Py": 3.15576e22, "Ey": 3.15576e25, "Zy": 3.15576e28, "Yy": 3.15576e31}


def load_nubase(path):
    rows = []
    for l in open(path):
        if l.startswith("#") or len(l) < 90:
            continue
        A = int(l[0:3]); zi = l[4:8]
        Z, iso = int(zi[:3]), int(zi[3])
        if iso != 0:
            continue
        t = l[69:78].strip(); unit = l[78:80].strip()
        est = "#" in t
        t = t.replace("#", "").replace("<", "").replace(">", "").replace("~", "")
        try:
            T = float(t) * UNITS[unit]
        except (ValueError, KeyError):
            T = np.inf if "stbl" in l[69:80] else np.nan
        br = l[119:].strip()
        m = re.search(r"(?:^|;)A\s*([=~])\s*([\d.]+(?:e[-+]?\d+)?)", br)
        alpha_frac = float(m.group(2)) / 100 if m else np.nan
        disc = l[114:118].strip()
        rows.append(dict(Z=Z, N=A - Z, A=A, T=T, T_est=est, decay=br,
                         alpha_frac=alpha_frac,
                         discovery=int(disc) if disc.isdigit() else np.nan))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    a = load_ame("../data/mass_1.mas20.txt")
    exp = a[~a.estimated & a.B.notna()]
    print(len(a), "nuclei;", len(exp), "measured")
    print("heaviest measured Z:", exp.Z.max(), exp[exp.Z >= 100][["el", "Z", "N", "B"]].to_string())
    print(a[(a.Z == 82) & (a.N == 126)][["el", "B", "estimated"]])
    nb = load_nubase("../data/nubase_4.mas20.txt")
    print(nb[(nb.Z >= 110)].head(20).to_string())
