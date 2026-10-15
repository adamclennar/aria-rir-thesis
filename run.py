#!/usr/bin/env python3
"""Run the pipeline's MATLAB stages: build each job list from config/rooms.json, then run it.

    python run.py baseline   [selection]   the reference paper's four conditions per take
    python run.py rescore                  re-score every saved baseline result in results/pipeline/summary.csv
    python run.py blocksweep [selection] [--factors 0.5 1.5 2]
                                           run the runTake script at multiples of each room's rule block length, then
                                           summariseBlockSweep stored in results/blocksweep/blocksweep.csv
    python run.py anchors    [selection]   the front-end anchors stored in results/frontend/
    python run.py oracle     [selection]   the oracle front-end candidates: exportInputs, dereverb/oracle.py,
                                           evalDereverb. Stored in results/dereverb/
    python run.py jobs FILE                any job list (lines: <label> <MATLAB command>)

  selection: --rooms R [R ...]  --movement static headMove fullMove  --takes T [T ...]
  every command: --jobs N (default 3)  --dry-run (print the jobs, run nothing)

Takes come from config/rooms.json, minus each room's exclusions. The L-shape takes count only where
they are part of the SLAM/OptiTrack pose comparison (an OptiTrack recording linked in rooms.json),
and only for baseline and rescore: the room exists for that comparison. Each stage defaults to the
takes it needs:
  baseline    every take
  blocksweep  walking and static takes of the composite rooms
  anchors     takes with a protocol bearing: an in-take sweep, or faces_source_at_reference
  oracle      the same takes, where the wearer does not walk

A job whose output already exists is skipped, so every command resumes. MATLAB jobs run a few at a
time, one MATLAB process each (a crash costs that job only, memory is freed between jobs, and
several run side by side), and a job whose MATLAB fails is retried once: MATLAB R2025a
occasionally crashes on its own at start-up or exit. Each job's output goes straight to
results/logs/<stage>/<label>.log, never through a pipe: MATLAB's service host inherits the pipe
and keeps it open, so a caller reading it would never see it close.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import scipy.io as sio

ROOT = Path(__file__).resolve().parent
MATLAB = os.environ.get("MATLAB", "/Applications/MATLAB_R2025a.app/bin/matlab")
PIPE = ROOT / "results" / "pipeline"
FRONT = ROOT / "results" / "frontend"
DEREV = ROOT / "results" / "dereverb"
MOVEMENTS = ("static", "headMove", "fullMove")
ANCHORS = ("published", "wpe_oracle_steering", "nowpe_oracle_steering", "perfect_frontend",
           "informed_lvl_matched", "informed")
ORACLES = ("oracle_comb", "oracle_floor", "oracle_early_sub", "oracle_early", "oracle_fixed", "oracle_perfect_resid")
PASSTHROUGH = ("office_maleSpeech_static", "meeting_femaleSpeech_headMove")   # the harness check
_lock = threading.Lock()


# ---- takes 
def movement(take: str) -> str:
    return next((m for m in MOVEMENTS if f"_{m}" in take), "")


def all_takes() -> list[dict]:
    """Every take in rooms.json, minus exclusions: name, room, movement, and its flags."""
    rooms = json.loads((ROOT / "config" / "rooms.json").read_text())["rooms"]
    out = []
    for room, rc in rooms.items():
        for t, tc in rc["takes"].items():
            if t in rc.get("exclude", {}):
                continue
            out.append({"take": t, "room": room, "movement": movement(t),
                        "sweep": "sweep_span_s" in rc or "sweep_span_s" in tc,
                        "faces": bool(tc.get("faces_source_at_reference")),
                        "pose_pair": "optitrack_csv" in tc})
    return out


def select(a, default) -> list[str]:
    """Takes passing the stage's default filter and the command-line selection."""
    takes = [d for d in all_takes() if default(d)]
    if a.rooms:
        takes = [d for d in takes if d["room"] in a.rooms]
    if a.movement:
        takes = [d for d in takes if d["movement"] in a.movement]
    names = [d["take"] for d in takes]
    if a.takes:
        missing = set(a.takes) - set(names)
        if missing:
            sys.exit(f"not available for this stage: {' '.join(sorted(missing))}")
        names = [t for t in names if t in a.takes]
    return names


def composite(d) -> bool:
    return d["room"] != "lshape"


def has_bearing(d) -> bool:
    return composite(d) and (d["sweep"] or d["faces"])


def rule_block(room: str) -> int:
    m = sio.loadmat(ROOT / "data" / "rooms" / f"{room}_reference.mat", variable_names=["blockLenMsSuggested"],
                    squeeze_me=True)
    return int(round(float(m["blockLenMsSuggested"])))


# ---- running 
def say(msg: str) -> None:
    with _lock:
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def run_one(label: str, cmd: str, logs: Path, delay: float) -> int:
    time.sleep(delay)
    env = dict(os.environ, MW_DDUX_FORCE_ENABLE="false")
    rc = 1
    for attempt in (1, 2):
        say(f"== {label}: starting (try {attempt})")
        with open(logs / f"{label}.log", "a") as log:
            rc = subprocess.run([MATLAB, "-batch", f"cd('{ROOT / 'pipeline'}'); ensurePaths(); {cmd}"], stdout=log,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env).returncode
        say(f"== {label}: exit {rc}")
        if rc == 0:
            return 0
    return rc


def run_jobs(jobs: list[tuple[str, str]], stage: str, a) -> bool:
    """Run (label, MATLAB command) jobs, a.jobs at a time, starts staggered by 30 s; True if all succeeded."""
    say(f"{stage}: {len(jobs)} job(s)")
    if a.dry_run:
        for label, cmd in jobs:
            print(f"  {label}  {cmd}")
        return True
    if not jobs:
        return True
    logs = ROOT / "results" / "logs" / stage
    logs.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        rcs = list(ex.map(lambda ij: run_one(*ij[1], logs, 30 * ij[0] if ij[0] < a.jobs else 0), enumerate(jobs)))
    failed = [label for (label, _), rc in zip(jobs, rcs) if rc != 0]
    say(f"{stage}: {len(jobs) - len(failed)}/{len(jobs)} succeeded" + (f"; failed: {' '.join(failed)}" if failed else ""))
    return not failed


def q(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


# ---- stages 
def baseline(a) -> bool:
    takes = select(a, lambda d: composite(d) or d["pose_pair"])
    return run_jobs([(t, f"runTake({q(t)})") for t in takes if not (PIPE / f"{t}.mat").exists()], "baseline", a)


def rescore(a) -> bool:
    return run_jobs([("rescore", "rescore()")], "rescore", a)


def blocksweep(a) -> bool:
    takes = select(a, lambda d: composite(d) and d["movement"] in ("static", "fullMove"))
    jobs = []
    for t in takes:
        rule = rule_block(t.split("_")[0])
        for f in a.factors:
            b = int(round(rule * f))
            if not (PIPE / f"{t}_blk{b}.mat").exists():
                jobs.append((f"{t}_blk{b}", f"runTake({q(t)}, 'blockLenMs', {b}, 'tag', '_blk{b}')"))
    ok = run_jobs(jobs, "blocksweep", a)
    summarise = ("summariseBlockSweep(unique(regexprep({dir(fullfile('..','results','pipeline','*_blk*.mat')).name}, "
                 "'_blk\\d+\\.mat$', '')))")
    return run_jobs([("summariseBlockSweep", summarise)], "blocksweep", a) and ok


def anchors(a) -> bool:
    takes = select(a, has_bearing)
    todo = [t for t in takes if not all((FRONT / t / f"{c}.mat").exists() for c in ANCHORS)]
    return run_jobs([(t, f"frontEndAnchors({q(t)})") for t in todo], "anchors", a)


def oracle(a) -> bool:
    takes = select(a, lambda d: has_bearing(d) and d["movement"] in ("static", "headMove"))
    inp, cand = DEREV / "inputs", DEREV / "candidates"
    exports = [(f"export_{t}", f"exportInputs({q(t)}, 'array', false, 'refNone', false, 'received', true)")
               for t in takes if not ((inp / f"{t}_ref_wpe.wav").exists() and (inp / f"{t}_received.wav").exists())]
    if not run_jobs(exports, "oracle", a):
        return False
    need = [t for t in takes if not all((cand / f"{t}_{c}.wav").exists() for c in ORACLES)]
    say(f"oracle: candidates for {len(need)} take(s)")
    if need and not a.dry_run:
        for args in ([], ["--split"]):
            subprocess.run([sys.executable, str(ROOT / "dereverb" / "oracle.py"), *args, *need], check=True)
        for t in need:                                   # passthrough is evaluated on PASSTHROUGH only
            if t not in PASSTHROUGH:
                (cand / f"{t}_passthrough.wav").unlink(missing_ok=True)
    pairs = [(t, c) for c in ORACLES for t in takes] + [(t, "passthrough") for t in PASSTHROUGH if t in takes]
    evals = [(f"eval_{t}_{c}", f"evalDereverb({q(t)}, {q(cand / f'{t}_{c}.wav')})") for t, c in pairs
             if not (DEREV / "eval" / f"{t}_{c}.mat").exists()]
    return run_jobs(evals, "oracle", a)


def jobfile(a) -> bool:
    lines = [ln.split(maxsplit=1) for ln in a.file.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    return run_jobs([(l, c) for l, c in lines], a.file.stem, a)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    stages = {"baseline": baseline, "rescore": rescore, "blocksweep": blocksweep, "anchors": anchors,
              "oracle": oracle, "jobs": jobfile}
    for name in stages:
        p = sub.add_parser(name)
        p.add_argument("--jobs", type=int, default=3)
        p.add_argument("--dry-run", action="store_true")
        if name in ("baseline", "blocksweep", "anchors", "oracle"):
            p.add_argument("--rooms", nargs="+")
            p.add_argument("--movement", nargs="+", choices=MOVEMENTS)
            p.add_argument("--takes", nargs="+")
        if name == "blocksweep":
            p.add_argument("--factors", nargs="+", type=float, default=[0.5, 1.5, 2.0])
        if name == "jobs":
            p.add_argument("file", type=Path)
    a = ap.parse_args()
    return 0 if stages[a.stage](a) else 1


if __name__ == "__main__":
    raise SystemExit(main())
