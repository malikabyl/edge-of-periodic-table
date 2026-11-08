#!/usr/bin/env bash
# Reproduces every number in results/ and rebuilds report/index.html. Takes ~30 min on a laptop (most of it Gaussian-process fits).
set -euo pipefail
cd "$(dirname "$0")/src"
mkdir -p ../results
python3 periodic.py        # experiment 1: periodic table from one rule and from ionization energies
python3 stars.py           # experiment 3: anomalous stars, Eu correlations, Th/Eu
python3 evaluate.py        # experiment 2, step 1: hidden-nuclei tests of the mass models (slow)
python3 predict.py         # step 1: predictions on the superheavy grid + half-life test (slow)
python3 qalpha.py          # step 2: measured alpha-decay energies as mass differences
python3 qa_final.py        #         error vs extrapolation distance, new 119/120 predictions
python3 decay_final.py     # step 3: alpha-decay laws, fission (GP), blind test on Lv/Ts/Og
python3 barriers.py        # step 4: theoretical fission barriers (HFB-14, BSkG3)
python3 sf2.py             #         do the barriers help? held-out comparison
python3 decay_final2.py    #         final 119/120 predictions with the chosen fission model
python3 ../report/build.py
