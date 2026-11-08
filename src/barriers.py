"""Fission barrier heights from two theoretical tables shipped with TALYS: HFB-14 fission paths and BSkG3 (2023)."""
import glob, re, numpy as np, pandas as pd
rows = []
for f in glob.glob('../data/talys/fission/hfbpath/*.fis'):
    cur = None
    for l in open(f):
        m = re.match(r"# Z=\s*(\d+) A=\s*(\d+)", l)
        if m:
            if cur and cur[2]: rows.append(dict(Z=cur[0], A=cur[1], N=cur[1]-cur[0], Bf_hfb=max(cur[2])))
            cur = (int(m.group(1)), int(m.group(2)), []); continue
        if l.startswith('#') or not l.strip(): continue
        cur[2].append(float(l.split()[3]))
    if cur and cur[2]: rows.append(dict(Z=cur[0], A=cur[1], N=cur[1]-cur[0], Bf_hfb=max(cur[2])))
hfb = pd.DataFrame(rows)
brow = []
for l in open('../data/talys/fission/hfbpath_bskg3/barriers_bskg3.dat'):
    if l.startswith('#') or not l.strip():
        continue
    v = l.split(); Z, N = int(v[0]), int(v[1])
    brow.append(dict(Z=Z, N=N, Bf_bskg3=max(float(v[3]), float(v[7]), float(v[11]))))  # highest of inner/outer barriers
bsk = pd.DataFrame(brow)
bar = hfb.merge(bsk, on=['Z','N'], how='outer')
import os; os.makedirs('../data/derived', exist_ok=True)
bar.to_csv('../data/derived/barriers.csv', index=False)
if __name__ == '__main__':
    print(len(hfb), len(bsk), len(bar))
    for z, a in [(92,238),(94,240),(98,252),(100,256),(102,254),(104,256),(110,280),(112,284),(114,286),(114,298),(116,296)]:
        r = bar[(bar.Z==z)&(bar.N==a-z)]
        print(z, a, r[['Bf_hfb','Bf_bskg3']].values.tolist())
