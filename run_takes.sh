#!/bin/bash
# Run the pipeline one take per MATLAB process, so a failure costs one take
# rather than the batch and memory is released between takes. Takes that
# already have a result file are skipped, so this resumes.
#
# Before starting, check nothing else is already running these takes:
#     ps aux | grep '[M]ATLAB_maca64'
# (note the real command name is MATLAB_maca64, NOT MacOS/MATLAB -- a pgrep
# pattern anchored on the latter reports "not running" for a live process,
# which once led to two runs competing over the same takes.)
cd "$(dirname "$0")"
MATLAB="${MATLAB:-/Applications/MATLAB_R2025a.app/bin/matlab}"   # override: MATLAB=/path/to/matlab ./run_takes.sh ...
mkdir -p results/pipeline
for t in "$@"; do
  if [ -f "results/pipeline/$t.mat" ]; then echo "== $t: already done, skipping"; continue; fi
  echo "== $t: starting $(date +%H:%M:%S)"
  MW_DDUX_FORCE_ENABLE=false "$MATLAB" -batch \
    "cd('$(pwd)/pipeline'); runTake('$t');" >> results/pipeline/run.log 2>&1
  echo "== $t: exit $? at $(date +%H:%M:%S)"
done
echo "ALL DONE"
