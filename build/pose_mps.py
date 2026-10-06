"""Convert from Aria MPS closed-loop trajectory -> array pose in the paper's convention.

Frame chain:
    MPS gives T_world_device(t)          (world: gravity aligned, +z up)
    VRS calibration gives T_device_cpf   (fixed, per device)
    CPF -> array frame (array frame convention for the downstream pipeline) is a fixed axis permutation P:
        Aria CPF is +X left, +Y up, +Z forward; the array/ATF frame used by
        the CHiME ATFs and aria_egocentric_audio is +X forward, +Y left,
        +Z up (verified by TDOA sign, CHiME_dataset/doa_eval/run_eval.py).

    R_world_array = R_world_device @ R_device_cpf @ P.T
    p_world_array = p_world_device + R_world_device @ t_device_cpf

Reference frame written to the .mat:
The reference pose is the mean pose over the stillest 2 s window
(stillest_window), searched in:
  * the in-take sweep, on composite takes - the wearer standing on the
    mark facing the loudspeaker throughout it;
  * the speech, on older takes with no sweep - the stillest 2s moment.
  
Using a pose the wearer actually held makes the RIR-at-reference a real
pose. build_take places the source signal so its direct path lands at
10 ms at this pose.

estimateRirs.m subtracts referenceRotationAziEleRad from yaw but takes the
translation azimuth in world coordinates as-is, so it silently requires the
world x-axis to be the array's forward direction at the reference pose
(Deppisch's files: reference yaw ~ -1 deg, reference position ~ origin).
We therefore re-express everything in a frame W' whose origin is the array
position at the reference pose and whose x-axis is the horizontal forward
direction there; referencePos = 0 and reference yaw = 0 by construction.

Euler angles are scipy 'ZYX' intrinsic: yaw about +z (counter-clockwise
from above, so +90 deg = facing left, matching ATF azimuth), pitch about
the new y (positive = nose DOWN), roll about the new x. Only yaw is
needed by the downstream pipeline.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

CPF_TO_ARRAY = np.array([[0.0, 0.0, 1.0],
                         [1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0]])


@dataclass
class Pose:
    t_s: np.ndarray        # (n,) device-clock seconds
    R: np.ndarray          # (n,3,3) world <- frame
    p: np.ndarray          # (n,3) metres
    quality: np.ndarray    # (n,) MPS quality_score, 1 = good

    def sample(self, t_query: np.ndarray) -> "Pose":
        """SLERP/linear resample. Queries outside the track are clamped."""
        tq = np.clip(t_query, self.t_s[0], self.t_s[-1])
        p = np.stack([np.interp(tq, self.t_s, self.p[:, k]) for k in range(3)], 1)
        R = Slerp(self.t_s, Rotation.from_matrix(self.R))(tq).as_matrix()
        q = np.interp(tq, self.t_s, self.quality)
        return Pose(tq, R, p, q)

    def yaw_pitch_roll(self) -> np.ndarray:
        return Rotation.from_matrix(self.R).as_euler("ZYX")


def load_closed_loop(mps_dir: str | Path) -> Pose:
    f = Path(mps_dir) / "slam" / "closed_loop_trajectory.csv"
    if not f.exists():
        raise FileNotFoundError(f)
    t, p, q, qs = [], [], [], []
    with open(f, newline="") as fh:
        for row in csv.DictReader(fh):
            t.append(float(row["tracking_timestamp_us"]) * 1e-6)
            p.append([float(row[f"t{a}_world_device"]) for a in "xyz"])
            q.append([float(row[f"q{a}_world_device"]) for a in "xyzw"])
            qs.append(float(row.get("quality_score", 1.0)))
    t = np.asarray(t)
    order = np.argsort(t)
    return Pose(t[order], Rotation.from_quat(np.asarray(q)[order]).as_matrix(),
                np.asarray(p)[order], np.asarray(qs)[order])


def device_to_array(pose: Pose, R_device_cpf: np.ndarray, t_device_cpf: np.ndarray) -> Pose:
    R_da = R_device_cpf @ CPF_TO_ARRAY.T
    R = np.einsum("nij,jk->nik", pose.R, R_da)
    p = pose.p + np.einsum("nij,j->ni", pose.R, t_device_cpf)
    return Pose(pose.t_s, R, p, pose.quality)


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def stillest_window(pose: Pose, t0: float, t1: float, win_s: float = 2.0,
                    yaw_scale_deg: float = 30.0, disp_scale_m: float = 0.3) -> tuple[float, float, float]:
    """Centre time of the win_s window in [t0, t1] where the wearer moved
    least: minimises yaw spread / yaw_scale + displacement / disp_scale.
    Returns (centre_s, yaw_spread_deg, displacement_m)."""
    sel = (pose.t_s >= t0) & (pose.t_s <= t1)
    t, R, p = pose.t_s[sel], pose.R[sel], pose.p[sel]
    yaw = np.unwrap(np.arctan2(R[:, 1, 0], R[:, 0, 0]))
    n = int(win_s / np.median(np.diff(t)))
    best = None
    for a in range(0, len(t) - n, max(n // 8, 1)):
        ys = np.degrees(np.ptp(yaw[a:a + n]))
        ds = np.linalg.norm(p[a:a + n] - p[a], axis=1).max()
        c = ys / yaw_scale_deg + ds / disp_scale_m
        if best is None or c < best[0]:
            best = (c, a, ys, ds)
    _, a, ys, ds = best
    return float(t[a] + win_s / 2), float(ys), float(ds)


def to_reference_frame(pose: Pose, ref_mask: np.ndarray) -> tuple[Pose, dict]:
    """Re-express `pose` in W' (origin and horizontal forward of the mean
    pose over `ref_mask`). Returns (pose_in_Wprime, transform_info)."""
    fwd = pose.R[ref_mask][:, :, 0].mean(0)          # mean forward axis, world
    yaw_ref = float(np.arctan2(fwd[1], fwd[0]))
    p_ref = pose.p[ref_mask].mean(0)
    Rw = _rz(-yaw_ref)
    R = np.einsum("ij,njk->nik", Rw, pose.R)
    p = (pose.p - p_ref) @ Rw.T
    info = {"yaw_ref_world_rad": yaw_ref, "p_ref_world_m": p_ref.tolist(),
            "R_wprime_world": Rw.tolist()}
    return Pose(pose.t_s, R, p, pose.quality), info
