#!/usr/bin/env python3
"""One-line-per-take sanity table over the built .mat files.

    python build/check_takes.py [stems...]

Reports, per take: level (peak/rms during speech), the loudest channel
event outside the noise (mechanical contact shows as one channel far above
the others), speech-to-silence SNR above 100 Hz, pose ranges during the
speech, MPS quality, and the source-alignment numbers from meta.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]


def check(path: Path) -> None:
    d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    x, fs = d["arraySigs"], float(d["fs"])
    m = d["meta"]
    s0, s1 = [int(v * fs) for v in m.speech_span_s]
    sp = x[s0:s1]
    pk = np.abs(x).max(0)
    ipk = int(np.argmax(np.abs(x).max(1)))
    order = np.argsort(pk)[::-1]
    contact = pk[order[0]] / pk[order[1]]           # >4 (12 dB) = one channel alone
    fq, pq = welch(d["silenceData"], fs, nperseg=4096, axis=0)
    fa, pa = welch(sp, fs, nperseg=4096, axis=0)
    snr = 10 * np.log10(pa[fa >= 100].sum(0) / pq[fq >= 100].sum(0))
    ypr = d["rotationYawPitchRollRad"][s0:s1]
    pos = d["translationXYZCart"][s0:s1]
    q = d["mpsQuality"][s0:s1]
    yaw = np.degrees(ypr[:, 0])
    d_ = m.drive
    src = getattr(m, "placement_source", "speech (GCC-PHAT)")
    sw0 = getattr(m, "sweep", None)
    if src.startswith("in-take"):
        place = f"sweep  corr {sw0.lagCorrectionSamples:+5d} smp"
    else:
        place = (f"speech PHAT {d_.ref_window_peak_ratio:3.0f} spread {d_.ref_window_offset_spread:3.0f}"
                 f" ({getattr(d_, 'ref_window_used', 0)}/{getattr(d_, 'ref_window_n', 0)})")
    sw = getattr(m, "sweep", None)
    irtxt = f" | IRs DRR {sw.drrDb:+5.1f} onset-spread {sw.perMicOnsetSpread:3d}" if sw is not None else ""
    print(f"{path.stem:32s} peak {20 * np.log10(pk.max()):6.1f} dBFS (ch{order[0]} at {ipk / fs:5.1f} s"
          f"{', ALONE x%.0f' % contact if contact > 4 else ''}) "
          f"rms {20 * np.log10(sp.std()):6.1f}  SNR {snr.min():4.1f}-{snr.max():4.1f} dB | "
          f"yaw {yaw.min():+5.0f}..{yaw.max():+4.0f} pitch {np.degrees(ypr[:, 1]).mean():+4.0f} "
          f"r_max {np.hypot(pos[:, 0], pos[:, 1]).max():4.2f} m q<1 {100 * np.mean(q < 1):4.1f}% | "
          f"ref@{m.ref_window_centre_s:4.1f}s ({m.ref_window_yaw_spread_deg:2.0f} deg, {100 * m.ref_window_displacement_m:2.0f} cm) | "
          f"{place}" + irtxt)


def main() -> int:
    stems = sys.argv[1:]
    files = sorted(p for p in (ROOT / "data" / "rooms").glob("*.mat") if "reference" not in p.name
                   and (not stems or p.stem in stems))
    for p in files:
        check(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
