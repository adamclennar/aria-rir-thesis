#!/usr/bin/env python3
"""CHiME Aria ATFs (aria_atfs.pkl) -> anechoic ATF .mat files for the pipeline.

    python build/build_atf.py

Writes data/anechoic/ariaAtfsHorizontal.mat in the layout of the paper's
glassesOnKemarAnechoicIRs.mat --

    irs [332 x 7 x 48]      elevation-0 ring (the CHiME grid is quasi-uniform
    rotationAnglesDeg [48x1] on the sphere: 6 and 12 deg gaps on this ring)
    fs, info

data/anechoic/ariaAtfsHorizontalDense.mat: same layout with every direction
within |el| <= 3 deg (136 directions, 60 unique azimuths: the +/-3 deg rings fill
the 12 deg gaps, giving uniform 6 deg coverage; elevation ignored) -- for the grid-ceiling experiment, not for reproducing the paper;

and data/anechoic/ariaAtfsFullSphere.mat (irs [332 x 7 x 1674], aziDeg,
eleDeg) for the elevation-aware extension.

Two conventions are baked in here, both established in
CHiME_dataset/doa_eval/run_eval.py and aria_egocentric_audio/notes/methods.md:

  * Channel order. VRS audio channel k is device mic k; the pkl has mics 5
    and 6 (the two rear mics) swapped relative to that. Uncorrected, it
    cost ~50 deg median azimuth error. Permutation (0,1,2,3,4,6,5) applied.
  * Angle sign. estimateRirs.m line 72 turns the stored angles into a
    measurement grid with atfGridHorAziRad = -rotationAnglesDeg, because
    the paper rotated the head (KEMAR on a turntable) rather than the
    source. The CHiME angles are SOURCE azimuths in the +x-forward,
    +y-left frame (+90 = left). We therefore store rotationAnglesDeg =
    -sourceAzimuthDeg so that the unmodified pipeline recovers the source
    azimuth. Wrapped to (-180, 180].

The measurements were made on a HATS (the near-field
ATF "from the direction of the mouth" needs a mouth; rear-mic ILD of 15 dB
at +/-90 deg is head shadow, not frame diffraction), so like the paper's
KEMAR set they include the head.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
from paths import CFG  # noqa: E402
CH_PERM = [0, 1, 2, 3, 4, 6, 5]
FS = 48000.0


def main() -> int:
    d = pickle.load(open(CFG["chime_atf_pkl"], "rb"))
    ff = np.asarray(d["far_field"], dtype=float)[:, CH_PERM, :]   # taps x mics x dirs
    az, el, dist = np.asarray(d["directions"], dtype=float)
    az_w = (az + 180) % 360 - 180

    out = ROOT / "data" / "anechoic"
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d")

    ring = np.where(np.abs(el) < 1e-6)[0]
    ring = ring[np.argsort(az_w[ring])]
    sio.savemat(out / "ariaAtfsHorizontal.mat", {
        "irs": ff[:, :, ring],
        "rotationAnglesDeg": (-az_w[ring])[:, None],
        "fs": FS,
        "info": (f"CHiME MCAS Aria Gen 1 far-field ATFs, horizontal ring ({len(ring)} directions, "
                 f"{dist[ring[0]]:.2f} m), measured on a HATS. Channels reordered (0,1,2,3,4,6,5) to match "
                 f"VRS audio order. rotationAnglesDeg = -sourceAzimuthDeg (paper convention; "
                 f"estimateRirs.m negates it). Source azimuth +90 = wearer's left. Built {stamp} by aria_rir/build/build_atf.py."),
    }, oned_as="column")

    # |el| <= 3 deg treated as horizontal: 136 directions, uniform 6 deg azimuth coverage.
    # For the grid-ceiling experiment, not for reproducing the paper.
    dense = np.where(np.abs(el) <= 3 + 1e-6)[0]
    dense = dense[np.argsort(az_w[dense], kind="stable")]
    sio.savemat(out / "ariaAtfsHorizontalDense.mat", {
        "irs": ff[:, :, dense],
        "rotationAnglesDeg": (-az_w[dense])[:, None],
        "eleDeg": el[dense][:, None],
        "fs": FS,
        "info": (f"As ariaAtfsHorizontal.mat but all CHiME directions with |elevation| <= 3 deg "
                 f"({len(dense)} directions), elevation ignored. Built {stamp}."),
    }, oned_as="column")

    sio.savemat(out / "ariaAtfsFullSphere.mat", {
        "irs": ff, "aziDeg": az_w[:, None], "eleDeg": el[:, None], "fs": FS,
        "nearFieldMouthIr": np.asarray(d["near_field"], dtype=float)[:, CH_PERM],
        "info": (f"CHiME MCAS Aria Gen 1 far-field ATFs, full sphere ({ff.shape[2]} directions, 6 deg az x 3 deg el). "
                 f"Channels reordered (0,1,2,3,4,6,5). aziDeg is SOURCE azimuth (+90 = left), eleDeg positive up. Built {stamp}."),
    }, oned_as="column")

    print(f"horizontal ring: {len(ring)} dirs, azimuths {az_w[ring].min():.0f}..{az_w[ring].max():.0f} step {np.diff(az_w[ring])[0]:.0f}")
    ua = np.unique(np.round(az_w[dense], 3))
    print(f"dense ring (|el|<=3): {len(dense)} dirs, {len(ua)} unique azimuths, max gap {np.diff(np.r_[ua, ua[0] + 360]).max():.0f} deg")
    print(f"full sphere: {ff.shape}")
    # sanity: for a source on the left, the left rear mic (device mic 5) leads the right rear (mic 6)
    i = ring[np.argmin(np.abs(az_w[ring] - 90))]
    on = [int(np.argmax(np.abs(ff[:, c, i]) > 0.2 * np.abs(ff[:, c, i]).max())) for c in range(7)]
    print(f"source at +90 (left): onsets per mic {on} -> mic5 leads mic6 by {on[6] - on[5]} samples "
          f"({'OK' if on[5] < on[6] else 'WRONG ORDER'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
