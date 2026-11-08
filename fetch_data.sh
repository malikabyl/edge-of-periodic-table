#!/usr/bin/env bash
# Downloads every input dataset into ./data and checks it against the checksums used for the published results.
# The original AME2020/NUBASE2020 files live at https://www-nds.iaea.org/amdc/ ; here they are taken from
# public mirrors on GitHub (pinned commits) so that the download is scriptable.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/derived
OPENMC=93b147fa49d15e13c9d3cef7a7b0430a9b2ab86a
O2SCL=d68532ff31e59dac65844c3d387e1388cb71a770
curl -sSfL -o data/mass_1.mas20.txt  "https://raw.githubusercontent.com/openmc-dev/openmc/$OPENMC/openmc/data/mass_1.mas20.txt"
curl -sSfL -o data/nubase_4.mas20.txt "https://raw.githubusercontent.com/awsteiner/o2scl/$O2SCL/data/o2scl/nucmass/ame20/nubase_4.mas20.txt"
curl -sSfL -o data/rct1.txt           "https://raw.githubusercontent.com/awsteiner/o2scl/$O2SCL/data/o2scl/nucmass/ame20/rct1_20.txt"
# JINAbase (metal-poor star abundances), 2021 update
[ -d data/jina ] || git clone -q https://github.com/Mohammad-Mardini/JINAbase-updated.git data/jina
git -C data/jina checkout -q e034a8c0bea71b3a657046ae622d75750c661319
# TALYS nuclear-structure database: only the fission-barrier files
if [ ! -d data/talys ]; then
  git clone -q --filter=blob:none --sparse https://github.com/psaxioti/talys-structure.git data/talys
  git -C data/talys checkout -q 2b340f143767934cac3ac003423b02b5d2dd86ef
  git -C data/talys sparse-checkout set --no-cone '/fission/hfbpath/' '/fission/hfbpath_bskg3/barriers_bskg3.dat'
fi
md5sum -c <<'SUMS'
6d28b75833cf53c7cc230223f63da6f6  data/mass_1.mas20.txt
91e92411c7c609aa73b28136da61317f  data/nubase_4.mas20.txt
5b143f5c3f0233053aeb243645eb8656  data/rct1.txt
d87f942db9c8fe16650018e17afe3b64  data/jina/JINAbase_2021.csv
18243bc2776d601b2c9cc83017698e8b  data/talys/fission/hfbpath_bskg3/barriers_bskg3.dat
SUMS
echo "All data in place."
