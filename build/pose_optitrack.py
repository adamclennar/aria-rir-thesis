"""OptiTrack ('glasses' rigid body) -> Aria device pose on the DEVICE clock.

The motion-aware model consumes the array pose relative to a reference pose,
so the OptiTrack world frame itself is irrelevant: build_take re-expresses
everything in W' (origin and horizontal forward at the reference pose). What
has to be known is how the tracked BODY relates to the DEVICE, and how the two
clocks relate:

    t_device = t_opti - opti_t0 + tau + mps_t0            (clock offset, per take)
    R_world_device(t) = R_world_rb(t) @ R_rb_dev           (hand-eye rotation)
    p_world_device(t) = p_world_rb(t) + R_world_rb(t) @ l_rb   (lever arm)

R_rb_dev and l_rb are physical constants of how the markers sit on the
glasses. They are fitted against MPS on OTHER takes (the head-moving ones,
which rotate the most) and only then applied to the take being built, so the
OptiTrack pose of that take owes nothing to its own SLAM trajectory beyond the
clock offset -- which can only come from a motion match, since the two
systems share no clock. tau is fitted by joint_align on angular velocity
(magnitude cross-correlation, then a joint rotation/offset solve) as in
aria_egocentric_audio/scripts/27_pose_slam_vs_optitrack.py.

The OptiTrack world must be gravity-aligned with +z up for W' to be right;
fit_link reports the tilt between its z and MPS's gravity-aligned z.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Rotation

from pose_mps import Pose

from optitrack_io import (  # noqa: E402
    PoseTrack, clean_angular_velocity, drop_frozen_frames, drop_implausible_rotation,
    estimate_clock_offset, joint_align, load_motive_csv)


def load_glasses(csv_path) -> PoseTrack:
    """'glasses' body in METRES (Motive exported centimetres), z-up as exported,
    with frozen and implausible-rotation frames removed (they leave holes)."""
    tr = load_motive_csv(str(csv_path), body_name="glasses", up_axis="z", strict=False)
    tr = PoseTrack(tr.t_s, tr.position * 0.01, tr.rotation)
    tr = drop_frozen_frames(tr, verbose=False)
    return drop_implausible_rotation(tr, verbose=False)


def gap_mask(t_query, t_orig, max_gap_s=0.02):
    """True where t_query lies within max_gap_s of a real mocap sample."""
    idx = np.clip(np.searchsorted(t_orig, t_query), 1, len(t_orig) - 1)
    return np.minimum(np.abs(t_query - t_orig[idx - 1]), np.abs(t_orig[idx] - t_query)) <= max_gap_s


def fit_clock(opti: PoseTrack, mps: Pose, smooth=15, max_offset_s=30.0) -> dict:
    """tau (zero-based mocap -> zero-based device time) and the rotation that
    maps body-frame angular velocity onto the device's: omega_dev = R omega_rb,
    i.e. R = R_dev_rb."""
    o0, m0 = float(opti.t_s[0]), float(mps.t_s[0])
    t_o, w_o = clean_angular_velocity(PoseTrack(opti.t_s - o0, opti.position, opti.rotation))
    mz = PoseTrack(mps.t_s - m0, mps.p, mps.R)
    t_m, w_m = mz.angular_velocity()
    w_o = uniform_filter1d(w_o, smooth, axis=0)
    w_m = uniform_filter1d(w_m, max(1, int(round(smooth * len(t_m) / len(t_o)))), axis=0)
    tau0, peak = estimate_clock_offset(t_m, w_m, t_o, w_o, max_offset_s=max_offset_s, step_s=0.002)
    al = joint_align(t_o, w_o, t_m, w_m, tau_init=tau0)
    return {"tau_s": float(al["tau_s"]), "R_dev_rb": np.asarray(al["R"]), "o0": o0, "m0": m0,
            "coarse_corr": float(peak), "rot_residual_deg": float(al["residual_deg"]),
            "tau_std_ms": float(al["tau_std_ms"]), "converged": bool(al["converged"])}


def on_device_clock(opti: PoseTrack, clk: dict) -> PoseTrack:
    return PoseTrack(opti.t_s - clk["o0"] + clk["tau_s"] + clk["m0"], opti.position, opti.rotation)


def fit_lever(opti_dev: PoseTrack, mps: Pose, R_rb_dev: np.ndarray) -> dict:
    """Lever arm l_rb and the OptiTrack->MPS world transform from positions.

    p_mps = R_W (p_rb + R_rb l) + c. R_W from orientations (Procrustes of
    R_mps (R_rb R_rb_dev)^T), then l and c by linear least squares. Only
    samples with a real mocap sample within 20 ms are used.
    """
    lo, hi = max(opti_dev.t_s[0], mps.t_s[0]), min(opti_dev.t_s[-1], mps.t_s[-1])
    t = np.arange(lo, hi, 1 / 120.0)
    t = t[gap_mask(t, opti_dev.t_s)]
    O, M = opti_dev.interp_to(t), mps.sample(t)
    A = np.einsum("nij,nkj->ik", M.R, np.einsum("nij,jk->nik", O.rotation, R_rb_dev))
    U, _, Vt = np.linalg.svd(A)
    R_W = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    # p_mps - R_W p_rb = R_W R_rb l + c   ->   [R_W R_rb | I] [l; c]
    lhs = np.concatenate([np.einsum("ij,njk->nik", R_W, O.rotation), np.tile(np.eye(3), (len(t), 1, 1))], 2)
    rhs = M.p - O.position @ R_W.T
    x, *_ = np.linalg.lstsq(lhs.reshape(-1, 6), rhs.reshape(-1), rcond=None)
    l, c = x[:3], x[3:]
    pred = (O.position + np.einsum("nij,j->ni", O.rotation, l)) @ R_W.T + c
    res = np.linalg.norm(pred - M.p, axis=1)
    R_pred = np.einsum("ij,njk->nik", R_W, np.einsum("nij,jk->nik", O.rotation, R_rb_dev))
    rot_err = Rotation.from_matrix(np.einsum("nji,njk->nik", R_pred, M.R)).magnitude()
    return {"l_rb": l, "R_W": R_W, "c": c, "n": len(t),
            "pos_rms_m": float(np.sqrt(np.mean(res ** 2))), "pos_p90_m": float(np.percentile(res, 90)),
            "rot_rms_deg": float(np.degrees(np.sqrt(np.mean(rot_err ** 2)))),
            "tilt_deg": float(np.degrees(np.arccos(np.clip(R_W[2, 2], -1, 1)))),
            "coverage": float(len(t) / max(len(np.arange(lo, hi, 1 / 120.0)), 1))}


def device_pose(opti_dev: PoseTrack, R_rb_dev: np.ndarray, l_rb: np.ndarray) -> Pose:
    """Device pose in the OptiTrack world (z-up), on the device clock.
    quality = 1 at real mocap samples (the Pose field MPS uses for its score)."""
    R = np.einsum("nij,jk->nik", opti_dev.rotation, R_rb_dev)
    p = opti_dev.position + np.einsum("nij,j->ni", opti_dev.rotation, l_rb)
    return Pose(opti_dev.t_s.copy(), R, p, np.ones(len(opti_dev.t_s)))
