#!/bin/bash
# Block-length sweep on the walking takes.
#
# Each take is run at 0.5x, 1.5x and 2x its room's rule block length - the
# blockLenMsSuggested of data/rooms/<room>_reference.mat: i.e. the smallest 500 ms
# multiple >= 2 x the longest-band T20 of the omni mic reference RIR measurement. The 1.0x point
# is the baseline run itself (results/pipeline/<take>.mat).
#
# One MATLAB per (take, block) so a failure costs one run; results carry a
# _blk<ms> tag so nothing overwrites the baseline; anything already on disk is
# skipped, so the script resumes. WPE does not depend on the block length and
# is reused from the baseline run's cache.
#
#   ./run_blocksweep.sh                   # every walking take of the four baseline rooms
#   ./run_blocksweep.sh <take> [<take>]   # just these
#
# everything scored afterwards with pipeline/summariseBlockSweep.m.
cd "$(dirname "$0")"
MATLAB="${MATLAB:-/Applications/MATLAB_R2025a.app/bin/matlab}"   # override: MATLAB=/path/to/matlab ./run_blocksweep.sh
FACTORS="0.5 1.5 2"
TAKES="$*"
[ -n "$TAKES" ] || TAKES="office_maleSpeech_fullMove office_femaleSpeech_fullMove office_maleSpeech_fullMove2
  corridor_maleSpeech_fullMove corridor_femaleSpeech_fullMove corridor_maleSpeech_fullMove2
  reverb_maleSpeech_fullMove reverb_femaleSpeech_fullMove
  meeting_maleSpeech_fullMove meeting_femaleSpeech_fullMove"
mkdir -p results/pipeline
for t in $TAKES; do
  room=${t%%_*}
  rule=$(python3 -c "import scipy.io as s; print(int(round(float(s.loadmat('data/rooms/${room}_reference.mat', squeeze_me=True)['blockLenMsSuggested']))))")
  for f in $FACTORS; do
    b=$(python3 -c "print(int(round($rule * $f)))")
    out="results/pipeline/${t}_blk${b}.mat"
    if [ -f "$out" ]; then echo "== $t @ ${b} ms (${f}x): already done"; continue; fi
    echo "== $t @ ${b} ms (${f}x): starting $(date +%H:%M:%S)"
    MW_DDUX_FORCE_ENABLE=false "$MATLAB" -batch \
      "cd('$(pwd)/pipeline'); runTake('$t','blockLenMs',$b,'tag','_blk${b}');" \
      >> results/pipeline/blocksweep.log 2>&1
    echo "== $t @ ${b} ms (${f}x): exit $? at $(date +%H:%M:%S)"
  done
done
echo "ALL DONE"
