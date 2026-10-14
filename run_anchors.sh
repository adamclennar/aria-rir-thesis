#!/bin/bash
# Front-end anchors (pipeline/frontEndAnchors.m) on every take with a protocol
# bearing (addOracleDoa): the composite-protocol takes, which have an in-take
# sweep, and the corridor static and head-only takes, whose reference pose
# faces the loudspeaker (config/rooms.json). Not the L-shape takes.
#
# One MATLAB process per take, JOBS takes at a time (the RLS keeps about one
# core busy, so a few run side by side; starts are staggered). A take whose
# MATLAB fails is retried once: MATLAB R2025a occasionally crashes on its own
# at start-up or exit. Finished conditions are cached in
# results/frontend/<take>/, so the script resumes. Each take logs to
# results/frontend/logs/<take>.log; MATLAB output goes straight to that file,
# never through a pipe (MATLAB's service host inherits the pipe and holds it
# open). Static and head-only takes are listed first.
#
#   ./run_anchors.sh                   # every take with a protocol bearing
#   ./run_anchors.sh <take> [<take>]   # just these
#   JOBS=2 ./run_anchors.sh            # a different number at a time (default 3)
cd "$(dirname "$0")"
MATLAB="${MATLAB:-/Applications/MATLAB_R2025a.app/bin/matlab}"   # override: MATLAB=/path/to/matlab ./run_anchors.sh
JOBS="${JOBS:-3}"
mkdir -p results/frontend/logs
TAKES="$*"
[ -n "$TAKES" ] || TAKES="office_maleSpeech_static office_femaleSpeech_static office_maleSpeech_headMove office_femaleSpeech_headMove
  reverb_maleSpeech_static reverb_femaleSpeech_static reverb_maleSpeech_headMove reverb_femaleSpeech_headMove
  meeting_maleSpeech_static meeting_femaleSpeech_static meeting_maleSpeech_headMove meeting_femaleSpeech_headMove
  smallmeeting_maleSpeech_static smallmeeting_femaleSpeech_static smallmeeting_maleSpeech_headMove smallmeeting_femaleSpeech_headMove
  corridor_maleSpeech_static corridor_femaleSpeech_static corridor_maleSpeech_headMove corridor_femaleSpeech_headMove
  office_maleSpeech_fullMove office_femaleSpeech_fullMove office_maleSpeech_fullMove2
  corridor_maleSpeech_fullMove corridor_femaleSpeech_fullMove corridor_maleSpeech_fullMove2
  reverb_maleSpeech_fullMove reverb_femaleSpeech_fullMove
  meeting_maleSpeech_fullMove meeting_femaleSpeech_fullMove meeting_maleSpeech_fullMove2
  smallmeeting_maleSpeech_fullMove smallmeeting_femaleSpeech_fullMove smallmeeting_maleSpeech_fullMove2 smallmeeting_femaleSpeech_fullMove2"

one() {   # one <take>: run it, and once more if MATLAB fails
  local t=$1 try rc
  for try in 1 2; do
    echo "== $t: starting $(date +%H:%M:%S) (try $try)"
    MW_DDUX_FORCE_ENABLE=false "$MATLAB" -batch \
      "cd('$(pwd)/pipeline'); frontEndAnchors('$t');" >> "results/frontend/logs/$t.log" 2>&1
    rc=$?
    echo "== $t: exit $rc at $(date +%H:%M:%S)"
    [ $rc -eq 0 ] && return
  done
}

worker() {   # worker <k>: every JOBS-th take, starting at the k-th
  local k=$1 t
  for t in $(printf '%s\n' $TAKES | awk -v k="$k" -v n="$JOBS" '(NR - 1) % n == k'); do one "$t"; done
}

for k in $(seq 0 $((JOBS - 1))); do
  (sleep $((k * 30)); worker "$k") &
done
wait
echo "ALL DONE"
