#!/usr/bin/env python3
"""Fit the OptiTrack-to-glasses link from recordings with both OptiTrack and MPS SLAM.

    python build/fit_optitrack_link.py lshape L_shape_calibration_1 L_shape_calibration_2
    python build/fit_optitrack_link.py lshape L_shape_calibration_1 L_shape_calibration_2 --write

The link maps the OptiTrack rigid body on the glasses to the glasses' device frame: a rotation
R_rb_dev and a lever arm l_rb (body origin -> device origin, in the body frame), so that
    R_world_device = R_world_rb R_rb_dev,   p_device = p_rb + R_world_rb l_rb
(pose_optitrack.device_pose). Per recording:
  1. the clock offset and the rotation, jointly, from the two systems' angular velocities
     (pose_optitrack.fit_clock: omega_dev = R_dev_rb omega_rb, so R_rb_dev = R_dev_rb^T);
  2. the lever arm and the OptiTrack-to-MPS world transform from positions, by least squares
     (pose_optitrack.fit_lever), using only samples within 20 ms of a real OptiTrack frame.
The link is the mean over the recordings (the rotation projected back onto SO(3)). The fit needs
rotation about all three axes; recordings made for it (turning the head about every axis) constrain
it better than takes recorded for the acoustics. The takes the link is later evaluated on should
not be among the recordings it is fitted on.

Prints each recording's fit, how the fits agree, and how the mean compares with the link in
config/rooms.json. --write stores the mean there, with the recordings in 'fitted_on'.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pose_optitrack as po                 # noqa: E402
from pose_mps import load_closed_loop        # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
from paths import RAW                         # noqa: E402


def fit_one(rc: dict, stem: str) -> dict:
    mps = load_closed_loop(RAW / rc["aria_dir"] / f"mps_{stem}_vrs")
    opti = po.load_glasses(RAW / rc["optitrack_dir"] / f"{stem}.csv")
    clk = po.fit_clock(opti, mps)
    R_rb_dev = clk["R_dev_rb"].T
    lev = po.fit_lever(po.on_device_clock(opti, clk), mps, R_rb_dev)
    return {"stem": stem, "R_rb_dev": R_rb_dev, "l_rb": lev["l_rb"], "tau_s": clk["tau_s"],
            "sync_corr": clk["coarse_corr"], "rot_resid_deg": clk["rot_residual_deg"],
            "pos_rms_cm": 100 * lev["pos_rms_m"], "pos_p90_cm": 100 * lev["pos_p90_m"],
            "rot_rms_deg": lev["rot_rms_deg"], "coverage": lev["coverage"], "n": lev["n"], "trimmed": lev["trimmed"]}


def mean_rotation(Rs: list[np.ndarray]) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.mean(Rs, axis=0))
    return U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt


def angle_deg(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(A.T @ B).magnitude()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("room")
    ap.add_argument("recordings", nargs="+", help="recording stems with both an OptiTrack csv and MPS output")
    ap.add_argument("--write", action="store_true", help="store the mean link in config/rooms.json")
    a = ap.parse_args()
    cfg_path = ROOT / "config" / "rooms.json"
    cfg_text = cfg_path.read_text()
    cfg = json.loads(cfg_text)
    rc = cfg["rooms"][a.room]
    fits = [fit_one(rc, s) for s in a.recordings]
    for f in fits:
        print(f"{f['stem']}: clock {f['tau_s']:+.4f} s (sync corr {f['sync_corr']:.2f}, rotation residual "
              f"{f['rot_resid_deg']:.1f} deg) | lever {np.round(100 * f['l_rb'], 1)} cm | fit residual "
              f"{f['pos_rms_cm']:.1f} cm rms (p90 {f['pos_p90_cm']:.1f}), {f['rot_rms_deg']:.1f} deg rms | "
              f"{f['n']} samples ({100 * f['trimmed']:.1f} % trimmed), coverage {100 * f['coverage']:.0f} %")
    for i in range(len(fits)):
        for j in range(i + 1, len(fits)):
            print(f"  {fits[i]['stem']} vs {fits[j]['stem']}: rotation {angle_deg(fits[i]['R_rb_dev'], fits[j]['R_rb_dev']):.1f} deg, "
                  f"lever {100 * np.linalg.norm(fits[i]['l_rb'] - fits[j]['l_rb']):.1f} cm")
    R = mean_rotation([f["R_rb_dev"] for f in fits])
    l = np.mean([f["l_rb"] for f in fits], axis=0)
    old = rc.get("optitrack_link", {})
    if old:
        print(f"mean vs current link ({', '.join(old.get('fitted_on', []))}): rotation "
              f"{angle_deg(np.array(old['R_rb_dev']), R):.1f} deg, lever {100 * np.linalg.norm(np.array(old['l_rb_m']) - l):.1f} cm")
    if a.write:
        assert json.dumps(cfg, indent=2, ensure_ascii=False) == cfg_text, "rooms.json would be reformatted"
        link = rc.setdefault("optitrack_link", {})
        link["R_rb_dev"] = [[round(float(v), 6) for v in row] for row in R]
        link["l_rb_m"] = [round(float(v), 4) for v in l]
        link["fitted_on"] = list(a.recordings)
        if len(fits) == 1:
            how = (f"fitted on {fits[0]['stem']} (build/fit_optitrack_link.py; residual {fits[0]['pos_rms_cm']:.1f} cm "
                   f"rms, {fits[0]['rot_rms_deg']:.1f} deg rms)")
        else:
            spread = max(angle_deg(f['R_rb_dev'], R) for f in fits), max(100 * np.linalg.norm(f['l_rb'] - l) for f in fits)
            how = (f"mean of the fits on {', '.join(a.recordings)} (build/fit_optitrack_link.py), each within "
                   f"{spread[0]:.1f} deg and {spread[1]:.1f} cm of the mean")
        link["note"] = (f"hand-eye rotation (rigid body -> device) and lever arm (body origin -> device origin, body "
                        f"frame), {how}. A recording made for the calibration, so every take the link is evaluated "
                        f"on is out of sample. Fixed constants: the build never refits them; only the clock offset is "
                        f"fitted per take.")
        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        print("wrote the mean link to config/rooms.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
