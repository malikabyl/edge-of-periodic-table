# Edge of the Periodic Table: three open-data experiments

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24058024.svg)](https://doi.org/10.5281/zenodo.24058024)

Three questions about the far end of the periodic table, answered only with published open datasets and a few hundred lines of Python:

1. **Can the periodic table be rebuilt from scratch?** From a single filling rule, and separately from data alone.
2. **Can machine learning predict how long the undiscovered elements 119 and 120 will live,** and say honestly how sure it is?
3. **Do old stars carry traces of nuclei heavier than uranium?**

Every number below is reproduced by `run_all.sh`. The full illustrated report (in Russian) is `report/index.html`.

## Main conclusion

The periodic table does not need to be *revised*. It needs to be *extended*, and at its edge simple logic stops working and data take over.

- **The table's logic can be derived, up to the edge.** The Madelung (n+l) rule alone reproduces all seven periods (2, 8, 8, 18, 18, 32, 32). Jumps in the first ionization energy, with no theory at all, give the same boundaries. The rule fails for 22 of 118 real configurations, and probably breaks down in period 8.
- **ML is strong near data and honest about where data end.** A blind prediction of the half-lives of Lv, Ts and Og, trained only on nuclei up to Mc (Z ≤ 115), is off by a factor of ~2 to 3. For elements 119 and 120 the prediction became useful only after measured alpha-decay chains were added. On the "island of stability" (N = 184) the model still knows almost nothing.
- **Simple models extrapolate better than complex ones.** For both alpha decay and fission, the simplest formulas generalized best beyond the training data. Theoretical fission barriers did not help.
- **Stars may see further than accelerators.** In r-process-enhanced stars, Ru, Pd and Ag scale with Eu and Th (slope ≈ 1) while Sr, Y and Zr do not. This is partly consistent with the fission-fragment signature reported by Roederer et al. (2023), but our sample is small.

**Prediction for elements 119 and 120:** α-decay half-lives of about **1–5 ms for element 119** and **0.06–0.4 ms for element 120**, for the isotopes targeted by the current experiments at JINR (Dubna) and LBNL (Berkeley). α decay is expected to win over spontaneous fission with 70–100% probability, so the new elements should be identifiable through their α chains. This is a falsifiable forecast that can be checked once either experiment reports.

## Experiment 1: the periodic table from one rule

`src/periodic.py`

| What | Result |
|---|---|
| Period lengths from the Madelung rule | 2, 8, 8, 18, 18, 32, 32, 50 (= 2k², each repeated twice) |
| Noble gases found from ionization-energy drops (>2×) | Z = 2, 10, 18, 36, 54, 86, i.e. periods 2, 8, 8, 18, 18, 32 |
| Real ground states that differ from the rule | 22 of 118 (Cr, Cu, Nb–Ag, La, Ce, Gd, Pt, Au, Ac–Np, Cm, Lr, Ds, Rg) |
| Side findings | The rule puts He above Be, not Ne; it puts Lu/Lr in group 3; it predicts a g-block starting at Z = 121 |

## Experiment 2: superheavy nuclei

Base model: a 6-term liquid-drop formula (Bethe–Weizsäcker) plus a Gaussian-process correction trained on the 2550 measured AME2020 masses. Every test hides data the model has never seen.

**Step 1: masses** (`evaluate.py`, `predict.py`). RMS error of the binding energy, in MeV:

| Model | Inside data | Neutron-rich edge | Beyond the edge (Z ≥ 99) |
|---|---|---|---|
| Liquid drop | 2.77 | 5.34 | 2.33 |
| Liquid drop + GP | **0.46** | **0.92** | **1.00** |
| Liquid drop + shell term + GP | 0.46 | 0.89 | 1.86 |

A crude shell term hurts extrapolation. With masses alone, the 68% half-life band for 119/120 spans about 11 orders of magnitude.

**Step 2: measured α-decay energies** (`qalpha.py`, `qa_final.py`). Superheavy masses are not measured, but 69 Qα values are. Each one is a constraint on a *difference* of masses, B(parent) − B(daughter) = B(⁴He) − Qα, which a GP handles exactly as a linear functional. Test: hide all Qα of Lv, Ts and Og (Z = 116–118):

| Training data | RMS error of hidden Qα |
|---|---|
| masses only | 0.49 MeV |
| masses + Qα up to Z = 115 | **0.12 MeV** |

The error stays around 0.2 MeV even four elements beyond the training data. The model's own uncertainty (1.3–2.6 MeV) is about 10 times too cautious.

**Step 3: decay laws and fission** (`decay_final.py`). Formulas are fit on Z < 104 and judged on the measured superheavies (Z ≥ 104):

| α-decay law | Train RMS | Superheavy RMS (dex) |
|---|---|---|
| Viola–Seaborg | 0.67 | **0.73** |
| Viola–Seaborg per parity class | 0.62 | 0.95 |
| Royer (2000) | 0.64 | 0.99 |
| UDL (2009) | 0.64 | 1.02 |

The blind end-to-end test (masses, Qα, decay law and fission all trained on Z ≤ 115) predicts the half-lives of 7 Lv/Ts/Og isotopes within a factor of ~2 (RMS 0.29 dex with a GP fission model), and all 7 fall inside the 68% band.

**Step 4: theoretical fission barriers** (`barriers.py`, `sf2.py`). Barrier heights from HFB-14 and BSkG3 (via the TALYS structure database) barely correlate with the 104 measured fission half-lives (r = 0.06 and 0.17). Adding them makes held-out predictions worse. The best fission model is the simplest: fissility (Z²/A) plus parity, with an RMS of 1.3–3 dex on hidden superheavies against 7–10 dex for a constant. Two reasonable fission models disagree by 4–12 orders of magnitude for 119/120, but both say fission is much slower than α decay.

| Nucleus (reaction) | α half-life | Total, median | 68% band | P(α first) |
|---|---|---|---|---|
| ²⁹⁵119 (²⁴⁹Bk + ⁵⁰Ti, 4n) | 0.9 ms | 0.9 ms | 0.12–6.8 ms | 78–100% |
| ²⁹⁶119 (²⁴⁹Bk + ⁵⁰Ti, 3n) | 4.6 ms | 4.6 ms | 0.6–35 ms | 70–100% |
| ²⁹⁵120 (²⁴⁹Cf + ⁵⁰Ti, 4n) | 0.17 ms | 0.17 ms | 0.02–1.2 ms | 76–100% |
| ²⁹⁶120 (²⁴⁹Cf + ⁵⁰Ti, 3n) | 0.06 ms | 0.06 ms | 0.009–0.49 ms | 84–100% |
| ²⁹⁸120 (²⁴⁸Cm + ⁵⁴Cr, 4n) | 0.11 ms | 0.11 ms | 0.015–0.86 ms | 84–100% |
| ²⁹⁹120 (²⁴⁸Cm + ⁵⁴Cr, 3n) | 0.44 ms | 0.44 ms | 0.06–3.4 ms | 78–100% |
| ²⁹⁴Og (known; measured 0.7 ms) | 1.6 ms | 1.5 ms | 0.2–12 ms | 79–98% |

## Experiment 3: anomalous stars

`src/stars.py`, using JINAbase (5129 metal-poor stars after merging repeat measurements).

- **Unsupervised anomaly search** (Isolation Forest on [Mg, Ca, Sr, Ba, Eu/Fe], 891 stars). s-process-rich stars are 5× over-represented among the 45 flagged stars. r-II stars are *not*, because the database was built around them, so for this sample they are normal. An anomaly is always relative to the sample.
- **Fission-fragment bridge.** In r-process stars ([Eu/Fe] > 0.3) the slope against [Eu/Fe] is 0.9–1.1 for Ru, Pd and Ag, and 0.99 for Th, against 0.36–0.51 for Sr, Y and Zr. This is partly consistent with Roederer et al. (2023), who interpret the Ru–Ag correlation as fragments of nuclei with A > 260. Our simple slope cannot separate that from the main r-process (La and Dy also give a slope of ≈ 1), and only 15–24 stars have Ru–Ag measured.
- **Thorium.** 12 of 66 stars show an actinide boost (log Th/Eu ≥ −0.2).

## Limitations

- Fission is the least predictable part. Model selection for fission used the same held-out tests that report its error, so that error may be slightly optimistic.
- The GP hyperparameters were fit on masses only and are over-cautious for Qα. The "calibrated" bands use the empirical Qα error (0.25 MeV) instead.
- β⁺/EC decay is ignored, and no data exist near N = 184, so the island of stability remains an open question.
- AME2020/NUBASE2020 are from 2020–2021. Newer superheavy data are not included.

## Reproduce

```bash
pip install -r requirements.txt
./fetch_data.sh     # downloads and checksums all inputs into ./data
./run_all.sh        # ~30 min; writes results/*.json and report/index.html
```

Tested with Python 3.13, numpy 2.5, pandas 3.0, scikit-learn 1.9, mendeleev 1.3.

## Data sources

- **AME2020 / NUBASE2020**: W.J. Huang et al., Chin. Phys. C 45, 030002 (2021); M. Wang et al., Chin. Phys. C 45, 030003 (2021); F.G. Kondev et al., Chin. Phys. C 45, 030001 (2021). https://www-nds.iaea.org/amdc/
- **JINAbase**: A. Abohalima & A. Frebel, ApJS 238, 36 (2018); 2021 update by M. Mardini. https://github.com/Mohammad-Mardini/JINAbase-updated
- **Fission barriers** (HFB-14 paths, BSkG3) from the TALYS nuclear-structure database. https://github.com/psaxioti/talys-structure
- **Electron configurations and ionization energies**: the `mendeleev` Python package (NIST data).
- **Solar abundances**: M. Asplund et al., ARA&A 47, 481 (2009).
- **Fission fragments in stars**: I.U. Roederer et al., Science 382, 1177 (2023). https://doi.org/10.1126/science.adf1341

## Use of AI

This project was carried out with Claude (Anthropic), an AI assistant, under the direction of the repository author. The AI wrote the code, ran the analyses and drafted the text. The author chose the questions and the direction of each step, and is responsible for the content. AI tools are not authors. Please treat the results as a reproducible exploratory study, not peer-reviewed work. Corrections from nuclear physicists and astronomers are very welcome; please open an issue.

## How to cite

Malik (2026). *Edge of the Periodic Table: three open-data experiments*. Zenodo. https://doi.org/10.5281/zenodo.24058024

This DOI always points to the latest version. To cite exactly v1.0, use https://doi.org/10.5281/zenodo.24058025.

## License

Code: MIT (see `LICENSE`). Input datasets belong to their authors and keep their own terms; they are downloaded by `fetch_data.sh` and not redistributed here.
