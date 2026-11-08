"""Build the periodic table from scratch two ways:
1) from one rule (Madelung n+l filling order) - pure logic, no chemistry;
2) from data only (jumps in first ionization energy) - no theory.
Then compare with real ground-state configurations and extend to period 8.
"""
import json, re
import numpy as np
from mendeleev.fetch import fetch_table

L = "spdfghi"
CAP = lambda l: 2 * (2 * l + 1)

# ---------- 1. Madelung rule ----------
orbitals = sorted([(n, l) for n in range(1, 10) for l in range(0, n)], key=lambda o: (o[0] + o[1], o[0]))


def madelung_config(Z):
    conf, left = {}, Z
    for n, l in orbitals:
        if left <= 0:
            break
        k = min(CAP(l), left)
        conf[f"{n}{L[l]}"] = k
        left -= k
    return conf


def layout(Zmax=172):
    """Assign period / block / column (32-wide long form, 50-wide for period 8 g-block)."""
    out = []
    period, start_counts = 0, {}
    for Z in range(1, Zmax + 1):
        conf = madelung_config(Z)
        last = list(conf)[-1]
        n, l = int(last[:-1]), L.index(last[-1])
        k = conf[last]
        if l == 0 and k == 1:
            period += 1
        out.append(dict(Z=Z, period=period, block=L[l], k=k, last=last))
    # period lengths
    lengths = [sum(1 for e in out if e["period"] == p) for p in range(1, out[-1]["period"] + 1)]
    return out, lengths


elements, lengths = layout(172)
# block order inside a period (s, g, f, d, p) -> column offset
for e in elements:
    off = {"s": 0, "g": 2, "f": 2, "d": 2, "p": 2}[e["block"]]
    p = e["period"]
    # widths available in this period
    has_f = p >= 6; has_g = p >= 8
    if e["block"] == "f":
        off = 2 + (18 if has_g else 0)
    if e["block"] == "d":
        off = 2 + (18 if has_g else 0) + (14 if has_f else 0)
    if e["block"] == "p":
        off = 2 + (18 if has_g else 0) + (14 if has_f else 0) + (10 if p >= 4 else 0)
    if e["block"] == "g":
        off = 2
    e["col"] = off + e["k"] - 1

# ---------- compare with reality ----------
t = fetch_table("elements").set_index("atomic_number")
CORE = {"He": 2, "Ne": 10, "Ar": 18, "Kr": 36, "Xe": 54, "Rn": 86}


def parse_real(s, Z):
    conf = {}
    m = re.match(r"\[(\w+)\]\s*", s)
    if m:
        core = madelung_config(CORE[m.group(1)])  # noble-gas cores obey Madelung
        conf.update(core)
        s = s[m.end():]
    for tok in s.split():
        mm = re.match(r"(\d+)([spdfg])(\d*)", tok)
        conf[mm.group(1) + mm.group(2)] = int(mm.group(3) or 1)
    return conf


exceptions = []
for Z in range(1, 119):
    real = parse_real(t.loc[Z, "electronic_configuration"], Z)
    pred = madelung_config(Z)
    if {k: v for k, v in real.items() if v} != pred:
        diff = {o: (pred.get(o, 0), real.get(o, 0)) for o in set(pred) | set(real) if pred.get(o, 0) != real.get(o, 0)}
        exceptions.append(dict(Z=Z, sym=t.loc[Z, "symbol"], real=t.loc[Z, "electronic_configuration"],
                               diff={k: f"{a}->{b}" for k, (a, b) in sorted(diff.items())}))

# ---------- 2. data only: first ionization energy ----------
ie = fetch_table("ionizationenergies")
ie1 = ie[ie.ion_charge == 0].set_index("atomic_number").ionization_energy.sort_index()
ratio = ie1.shift(-1) / ie1
nobles = [int(z) for z, r in ratio.items() if r < 0.5]
bounds = [0] + nobles
data_lengths = [b - a for a, b in zip(bounds[:-1], bounds[1:])]

print("Madelung period lengths:", lengths)
print("2*k^2 check:", [2 * k * k for k in [1, 2, 2, 3, 3, 4, 4, 5, 5]])
print("Data-only noble gases (IE drop >2x):", nobles, "-> period lengths", data_lengths)
print(len(exceptions), "of 118 real configurations differ from Madelung:")
for e in exceptions:
    print(" ", e["Z"], e["sym"], e["real"], e["diff"])
grp3 = [e for e in elements if e["Z"] in (57, 71, 89, 103)]
print("group-3 check:", [(e["Z"], e["block"], e["col"]) for e in grp3])

sym = t.symbol.to_dict()
json.dump(dict(
    elements=[dict(e, sym=sym.get(e["Z"], ""), name=(t.loc[e["Z"], "name"] if e["Z"] in t.index else ""))
              for e in elements],
    lengths=lengths, nobles=nobles, data_lengths=data_lengths,
    ie1=[[int(z), float(v)] for z, v in ie1.items()],
    exceptions=exceptions), open("../results/periodic.json", "w"), indent=0, default=str)
