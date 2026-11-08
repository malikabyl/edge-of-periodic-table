"""Anomalous stars in JINAbase (metal-poor star abundances, Abohalima & Frebel 2018; 2021 update).

1) classic labels (r-I / r-II / s-rich) from simple cuts;
2) unsupervised anomaly search (IsolationForest) without labels -> does it rediscover famous stars?
3) a bridge to superheavies: do Ru-Ag track Eu in r-process stars (fission-fragment signature)?
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# solar photospheric log-eps (Asplund et al. 2009); Th, U meteoritic
SUN = dict(Fe=7.50, Mg=7.60, Ca=6.34, C=8.43, Sr=2.87, Y=2.21, Zr=2.58, Ru=1.75, Rh=0.91, Pd=1.57,
           Ag=0.94, Ba=2.18, La=1.10, Ce=1.58, Nd=1.42, Sm=0.96, Eu=0.52, Gd=1.07, Dy=1.10,
           Er=0.92, Os=1.40, Ir=1.38, Pt=1.62, Pb=1.75, Th=0.06, U=-0.54)

raw = pd.read_csv("../data/jina/JINAbase_2021.csv", low_memory=False)
raw = raw[raw["Fe/H"].notna()].copy()
raw["star"] = raw.Simbad_Identifier.fillna(raw.Name).astype(str)
raw["Name"] = raw.Name.fillna(raw.Simbad_Identifier).astype(str)
for el in SUN:
    s = raw[el].astype(str).str.strip()
    v = pd.to_numeric(s.str.replace("<", "", regex=False), errors="coerce")
    v[s.str.startswith("<")] = np.nan  # upper limits are not detections
    raw[f"[{el}/Fe]"] = v - SUN[el] - raw["Fe/H"]
# one star can appear in several papers: combine by median of each measurement
num = ["Fe/H"] + [f"[{el}/Fe]" for el in SUN]
d = raw.groupby("star").agg({**{c: "median" for c in num}, "Name": "first", "Ref": lambda x: ",".join(sorted(set(map(str, x))))})
d = d.reset_index()
d["[Ba/Eu]"] = d["[Ba/Fe]"] - d["[Eu/Fe]"]
d["logThEu"] = (d["[Th/Fe]"] + SUN["Th"]) - (d["[Eu/Fe]"] + SUN["Eu"])
print("unique stars:", len(d), " with Eu detected:", d["[Eu/Fe]"].notna().sum(),
      " Th:", d["[Th/Fe]"].notna().sum(), " U:", d["[U/Fe]"].notna().sum())

# ---- 1. classic labels ----
def label(r):
    eu, baeu, ba = r["[Eu/Fe]"], r["[Ba/Eu]"], r["[Ba/Fe]"]
    if pd.notna(ba) and ba > 1.0 and pd.notna(baeu) and baeu > 0.5:
        return "s-rich"
    if pd.notna(eu) and pd.notna(baeu) and baeu < 0:
        if eu > 0.7: return "r-II"
        if eu > 0.3: return "r-I"
    return "normal"
d["label"] = d.apply(label, axis=1)
print(d.label.value_counts().to_dict())

# ---- 2. unsupervised anomaly search ----
feat = ["[Mg/Fe]", "[Ca/Fe]", "[Sr/Fe]", "[Ba/Fe]", "[Eu/Fe]"]
X = d[feat].dropna()
iso = IsolationForest(n_estimators=500, contamination=0.05, random_state=0).fit(X)
d.loc[X.index, "anomaly"] = -iso.score_samples(X)
d.loc[X.index, "is_anom"] = iso.predict(X) == -1
an = d.loc[X.index]
print("stars in anomaly search:", len(X), " flagged:", int(an.is_anom.sum()))
print("label mix among flagged:", an[an.is_anom].label.value_counts().to_dict())
print("label mix overall      :", an.label.value_counts().to_dict())
famous = {"CS22892-052": "Sneden's star — first strongly r-enhanced star",
          "CS31082-001": "Cayrel's star — uranium detected",
          "HE1523-0901": "Frebel's star — uranium, age ~13 Gyr",
          "HE0107-5240": "one of the most iron-poor stars",
          "SMSSJ031300.36-670839.3": "most iron-poor star known",
          "HE1327-2326": "hyper iron-poor star"}
top = an.sort_values("anomaly", ascending=False)
top["rank"] = np.arange(1, len(top) + 1)
for k, v in famous.items():
    m = top[(top.Name.str.replace(" ", "").str.contains(k, regex=False))]
    print(f"{k:25s} rank={m['rank'].tolist()} of {len(top)}   ({v})")

# ---- 3. fission-fragment bridge ----
r = d[(d["[Eu/Fe]"] > 0.3) & (d["[Ba/Eu]"] < 0)].copy()
bridge = {}
for el in ["Sr", "Y", "Zr", "Ru", "Rh", "Pd", "Ag", "Ba", "La", "Dy", "Th"]:
    s = r[[f"[{el}/Fe]", "[Eu/Fe]"]].dropna()
    if len(s) < 8: continue
    slope, icpt = np.polyfit(s["[Eu/Fe]"], s[f"[{el}/Fe]"], 1)
    bridge[el] = dict(n=len(s), slope=float(slope), corr=float(s.corr().iloc[0, 1]))
    print(f"r-stars: [{el}/Fe] vs [Eu/Fe]  n={len(s):3d}  slope={slope:.2f}  corr={bridge[el]['corr']:.2f}")

cols = ["Name", "Fe/H", "[Mg/Fe]", "[Sr/Fe]", "[Ba/Fe]", "[Eu/Fe]", "[Ba/Eu]", "logThEu", "label", "anomaly", "is_anom", "Ref"]
out = dict(
    n_stars=int(len(d)), n_eu=int(d["[Eu/Fe]"].notna().sum()), n_search=int(len(X)),
    labels=d.label.value_counts().to_dict(),
    flagged_mix=an[an.is_anom].label.value_counts().to_dict(), overall_mix=an.label.value_counts().to_dict(),
    top=top.head(25)[cols + ["rank"]].round(2).to_dict("records"),
    famous={k: dict(note=v, rank=top[top.Name.str.replace(" ", "").str.contains(k, regex=False)]["rank"].tolist())
            for k, v in famous.items()},
    scatter=an[["Name", "Fe/H", "[Eu/Fe]", "[Ba/Eu]", "label", "is_anom"]].round(2).to_dict("records"),
    thorium=d[d.logThEu.notna()][["Name", "Fe/H", "logThEu", "[Eu/Fe]"]].round(2).to_dict("records"),
    bridge=bridge, feat=feat)
json.dump(out, open("../results/stars.json", "w"), default=lambda o: None if pd.isna(o) else bool(o) if isinstance(o, np.bool_) else str(o))
