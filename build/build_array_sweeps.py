#!/usr/bin/env python3
"""7-channel array RIRs from standalone sweep recordings (glasses worn at the reference) to obtain ground-truth data.

    python build/build_array_sweeps.py lshape

For rooms recorded before the composite protocol there is no sweep inside the
takes, so no in-take IRs. The L-shape room has instead six separate recordings
of the standard 20 s sweep made with the glasses worn on the mark, facing the
loudspeaker (same loudspeaker and mark as the takes): three with the OptiTrack
markers on the glasses, as during the takes, and three without.

Each is deconvolved exactly as build_take.array_ir_from_sweep does it - the
sweep located by itself, the same regularised deconvolution, 5 ms of pre-delay,
3 s long - so the result is directly comparable with the in-take IRs of the
other rooms. Absolute delay is NOT kept (the recording and the excitation are on
different clocks and the segment starts at the correlation peak), so these give
RT and DRR but not distance; distance comes from the tape (config distance_m).

Writes data/rooms/<room>_arraySweeps.mat:
    IRs [3 s x 7 x n]   names {n}   markers [n] (1 = OptiTrack markers on)
    drrCh0Db [n], onsetSpreadSmp [n], fs, info
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_reference import deconvolve          # noqa: E402
from source_align import _hpf, xcorr_lag         # noqa: E402
from vrs_audio import load_audio                 # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
from paths import CFG, RAW  # noqa: E402


def sweep_ir(x, exc, fs, ir_len_s=3.0, pre_s=0.005, tail_s=3.0):
    ch = int(np.argmax(np.abs(x).max(0)))
    a, r = xcorr_lag(_hpf(x[:, ch], fs), _hpf(exc, fs))
    b = a + len(exc) + int(tail_s * fs)
    if a < 0 or b > x.shape[0]:
        raise RuntimeError("sweep falls outside the recording")
    seg = x[a:b]
    irs = np.stack([deconvolve(seg[:, c], exc, fs) for c in range(seg.shape[1])], 1)
    p = int(np.argmax(np.abs(irs[: int(0.5 * fs), 0])))
    lo = p - int(pre_s * fs)
    out = np.zeros((int(ir_len_s * fs), irs.shape[1]))
    cut = irs[max(lo, 0):lo + len(out)]
    out[max(-lo, 0):max(-lo, 0) + len(cut)] = cut
    kk = int(np.argmax(np.abs(out[:, 0])))
    d0, d1 = max(0, kk - int(0.001 * fs)), kk + int(0.0025 * fs)
    drr = float(10 * np.log10(np.sum(out[d0:d1, 0] ** 2) / np.sum(out[d1:, 0] ** 2)))
    onset = [int(np.argmax(np.abs(out[:, c]) > 0.3 * np.abs(out[:, c]).max())) for c in range(out.shape[1])]
    return out, drr, max(onset) - min(onset), a / fs, r


def main(room):
    rc = CFG["rooms"][room]
    sw = rc["array_sweeps"]
    exc, fs = sf.read(RAW / sw["excitation"], always_2d=True)
    exc = exc[:, 0]
    names = sw["markers"] + sw["no_markers"]
    markers = [1] * len(sw["markers"]) + [0] * len(sw["no_markers"])
    IRs, drr, spread = [], [], []
    for n in names:
        x, _, fsx = load_audio(RAW / rc["aria_dir"] / sw["dir"] / n, ROOT / "data" / "cache")
        assert abs(fsx - fs) < 1
        out, d, s, at, r = sweep_ir(x, exc, fs)
        if s > 25:
            print(f"  WARNING {n}: per-mic onset spread {s} smp exceeds the array aperture (~20)")
        print(f"  {n:30s} sweep at {at:6.2f} s (corr {r:.2f}), ch0 DRR {d:+6.1f} dB, per-mic onset spread {s:2d} smp")
        IRs.append(out); drr.append(d); spread.append(s)
    out = {"IRs": np.stack(IRs, 2), "names": np.array(names, dtype=object), "markers": np.array(markers, float),
           "drrCh0Db": np.array(drr), "onsetSpreadSmp": np.array(spread, float), "fs": float(fs),
           "info": (f"Array-side RIRs for the {room} room from {len(names)} standalone sweep recordings, glasses "
                    f"worn on the mark facing the loudspeaker (markers=1: OptiTrack markers on the glasses, as in "
                    f"the takes). Deconvolved as build_take.array_ir_from_sweep. Absolute delay not kept. "
                    f"Built {time.strftime('%Y-%m-%d')} by aria_rir/build/build_array_sweeps.py.")}
    sio.savemat(ROOT / "data" / "rooms" / f"{room}_arraySweeps.mat", out, do_compression=False)
    m = np.array(markers, bool)
    print(f"[{room}] ch0 DRR markers {np.round(np.array(drr)[m], 1)} | no markers {np.round(np.array(drr)[~m], 1)}")


if __name__ == "__main__":
    for room in sys.argv[1:]:
        main(room)
