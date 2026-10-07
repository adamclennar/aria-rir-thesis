"""Place the source signal on a moving take from TRACKED GEOMETRY.

Why: placement needs the direct-path lag at the reference pose to within half
a millisecond. A measured lag is (clock offset + latency) + travel time, and on
a take where the wearer never stood still the travel time changes throughout
every measurement window, so a window measures a blur -- or, worse, latches
onto a reflection (lshape_maleSpeech_fullMove2: 4.1 ms out). OptiTrack tracks
the loudspeaker as well as the glasses, so the travel time to mic 0 is KNOWN at
every sample:

    take[n] ~ drive(n - K - D(n)),   D(n) = |mic0(t_n) - source| * fs / c

Warping the drive by D(n) leaves one constant, K, which a GCC-PHAT over the
whole speech finds sharply (the direct path adds coherently under the warp;
reflections, whose path changes differently, do not). The lag at the
reference pose is then K + D(n_ref).

Validation tests the GEOMETRY, not just the bookkeeping:
  * split halves: K from the first and second half of the speech must agree;
  * per-second residuals: the measured lag in each 1 s window, after the warp,
    must sit at the same K (i.e. OptiTrack predicts the delay as it changes).
A constant error in the assumed source point or latency cancels (it moves K,
not the residuals); an error in c, the mic position or the clock offset
between OptiTrack and the device does not, and shows up in both checks.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from pose_mps import Pose
from source_align import _hpf, phat_lag


def mic_position_device(vrs_path, label="mic0") -> np.ndarray:
    """CAD position of a microphone in the device frame (metres). Aria Gen 1
    microphones carry no factory extrinsics, so the CAD value is the one."""
    from projectaria_tools.core import data_provider
    c = data_provider.create_vrs_data_provider(str(vrs_path)).get_device_calibration()
    return np.asarray(c.get_transform_device_sensor(label, True).translation(), float).reshape(3)


def speaker_point(csv_path, bodies=("speaker_side", "speaker_top")) -> np.ndarray:
    """Loudspeaker reference point in the OptiTrack frame: the mean of the
    tracked bodies' median positions (metres). Not the acoustic centre -- a
    constant offset cancels, see module docstring."""
    from optitrack_io import load_motive_csv
    pts = [np.median(load_motive_csv(str(csv_path), body_name=b, up_axis="z", strict=False).position, 0) * 0.01
           for b in bodies]
    return np.mean(pts, 0)


def delay_samples(dev: Pose, valid_t: np.ndarray, mic_dev: np.ndarray, src: np.ndarray,
                  t_query: np.ndarray, fs: float, c: float, max_gap_s: float = 0.02):
    """D at each t_query (samples) and a validity mask (a real mocap sample
    within max_gap_s). Evaluated on a 1 kHz grid, then interpolated."""
    g = np.arange(t_query[0], t_query[-1] + 1e-3, 1e-3)
    gq = np.clip(g, dev.t_s[0], dev.t_s[-1])
    p = np.stack([np.interp(gq, dev.t_s, dev.p[:, k]) for k in range(3)], 1)
    R = Slerp(dev.t_s, Rotation.from_matrix(dev.R))(gq).as_matrix()
    L = np.linalg.norm(p + np.einsum("nij,j->ni", R, mic_dev) - src, axis=1)
    idx = np.clip(np.searchsorted(valid_t, g), 1, len(valid_t) - 1)
    ok = np.minimum(np.abs(g - valid_t[idx - 1]), np.abs(valid_t[idx] - g)) <= max_gap_s
    ok &= (g >= valid_t[0]) & (g <= valid_t[-1])
    D = np.interp(t_query, g, L * fs / c)
    valid = np.interp(t_query, g, ok.astype(float)) > 0.999
    return D, valid, np.interp(t_query, g, L)


def _warp(y, n, D, K0):
    """u[n] = y(n - D(n) - K0), fractional, zero outside y."""
    q = n - D - K0
    out = np.interp(q, np.arange(len(y)), y, left=0.0, right=0.0)
    return out


def fit_K(take_ch, drive, D, valid, span, K0, fs, win_s=1.0):
    """K by whole-span motion-compensated GCC-PHAT, plus the two checks.

    take_ch : one Aria channel (the one the placement check uses); D/valid on
    the take's sample grid; span = (n0, n1) take samples of the speech."""
    xt, yd = _hpf(take_ch, fs), _hpf(drive, fs)
    n0, n1 = span
    n = np.arange(n0, n1)
    m = valid[n0:n1]
    u = _warp(yd, n.astype(float), D[n0:n1], K0) * m
    x = xt[n0:n1] * m
    d, pk = phat_lag(x, u, fs)
    K = K0 + d
    half = (n1 - n0) // 2
    dh = [phat_lag(x[a:b], u[a:b], fs) for a, b in ((0, half), (half, n1 - n0))]
    # per-second residuals against the SAME warped signal re-centred on K
    u2 = _warp(yd, n.astype(float), D[n0:n1], K) * m
    w = int(win_s * fs); res = []
    for a in range(0, len(x) - w, w):
        if m[a:a + w].mean() < 0.9:
            continue
        k, r = phat_lag(x[a:a + w], u2[a:a + w], fs)
        res.append((a / fs, k, r))
    res = np.array(res) if res else np.zeros((0, 3))
    good = res[res[:, 2] >= 6] if len(res) else res
    return {"K": int(round(K)), "peak_ratio": float(pk),
            "K_half1": int(round(K0 + dh[0][0])), "K_half2": int(round(K0 + dh[1][0])),
            "half_peak_ratios": [float(dh[0][1]), float(dh[1][1])],
            "valid_fraction": float(m.mean()),
            "residual_windows": int(len(res)), "residual_windows_sharp": int(len(good)),
            "residual_median_smp": float(np.median(good[:, 1])) if len(good) else float("nan"),
            "residual_within_24_pct": float(100 * np.mean(np.abs(good[:, 1]) <= 24)) if len(good) else 0.0,
            "residual_iqr_smp": float(np.subtract(*np.percentile(good[:, 1], [75, 25]))) if len(good) else float("nan"),
            "residuals": res.tolist()}


# --------------------------------------------------------------- take glue

def tracked_geometry(rc: dict, tc: dict, raw: Path, vrs: Path, mps_device: Pose) -> dict:
    """Everything a take needs from OptiTrack, with the link constants FIXED
    from config (never refitted here) and only the clock offset fitted."""
    import pose_optitrack as po
    link = rc["optitrack_link"]
    csv = raw / rc["optitrack_dir"] / tc["optitrack_csv"]
    opti = po.load_glasses(csv)
    clk = po.fit_clock(opti, mps_device)
    dev = po.device_pose(po.on_device_clock(opti, clk), np.array(link["R_rb_dev"]), np.array(link["l_rb_m"]))
    # OptiTrack world -> MPS world, only to fill OptiTrack gaps in the
    # OptiTrack-pose version (W' makes the world frame itself irrelevant).
    t = dev.t_s[(dev.t_s >= mps_device.t_s[0]) & (dev.t_s <= mps_device.t_s[-1])]
    M, O = mps_device.sample(t), dev.sample(t)
    U, _, Vt = np.linalg.svd(np.einsum("nij,nkj->ik", M.R, O.R))
    R_W = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    c = np.mean(M.p - O.p @ R_W.T, 0)
    res = np.linalg.norm(O.p @ R_W.T + c - M.p, axis=1)
    rot = Rotation.from_matrix(np.einsum("nji,njk->nik", np.einsum("ij,njk->nik", R_W, O.R), M.R)).magnitude()
    return {"dev": dev, "clk": clk, "csv": csv.name, "R_W": R_W, "c": c,
            "mic0": mic_position_device(vrs), "src": speaker_point(csv, tuple(link["speaker_bodies"])),
            "c_sound": float(link["speed_of_sound_m_s"]),
            "vs_mps_pos_median_cm": float(100 * np.median(res)), "vs_mps_pos_p90_cm": float(100 * np.percentile(res, 90)),
            "vs_mps_rot_median_deg": float(np.degrees(np.median(rot))),
            "vs_mps_rot_p90_deg": float(np.degrees(np.percentile(rot, 90)))}


def hybrid_pose(geo: dict, mps_device: Pose, max_gap_s: float = 0.02) -> Pose:
    """OptiTrack device pose on the MPS 1 kHz grid; samples with no mocap
    within max_gap_s are filled from MPS mapped into the OptiTrack world.
    quality = 1 where OptiTrack, 0 where filled."""
    import pose_optitrack as po
    g = mps_device.t_s
    ok = po.gap_mask(g, geo["dev"].t_s, max_gap_s) & (g >= geo["dev"].t_s[0]) & (g <= geo["dev"].t_s[-1])
    O = geo["dev"].sample(g)
    R = np.einsum("ji,njk->nik", geo["R_W"], mps_device.R)
    p = (mps_device.p - geo["c"]) @ geo["R_W"]
    R[ok], p[ok] = O.R[ok], O.p[ok]
    return Pose(g.copy(), R, p, ok.astype(float))


def place_from_geometry(al: dict, geo: dict, take_ch: np.ndarray, t_take_s: np.ndarray,
                        t_ref_s: float, fs: float) -> dict:
    """Set al's placement from the geometry fit; returns the fit diagnostics."""
    D, valid, L = delay_samples(geo["dev"], geo["dev"].t_s, geo["mic0"], geo["src"], t_take_s, fs, geo["c_sound"])
    n0 = int((al["speech_start_take_s"] + 1.0) * fs)
    n1 = int((al["speech_end_take_s"] - 1.0) * fs)
    fit = fit_K(take_ch, al["src"], D, valid, (n0, n1), al["lag_coarse"], fs)
    n_ref = int(np.argmin(np.abs(t_take_s - t_ref_s)))
    if not valid[n_ref]:
        raise RuntimeError("no OptiTrack sample within 20 ms of the reference instant; "
                           "the reference delay would be interpolated across a dropout")
    al["lag_direct"] = int(round(fit["K"] + D[n_ref]))
    al["lag_place"] = al["lag_direct"] - al["pre_delay_samples"]
    al["ref_window_centre_take_s"] = float(t_take_s[n_ref] - t_take_s[0])
    sp = al["speech_span_src_s"]
    al["speech_start_take_s"] = (al["lag_direct"] + sp[0] * fs) / fs
    al["speech_end_take_s"] = (al["lag_direct"] + sp[1] * fs) / fs
    fit.update({"D_ref_smp": float(D[n_ref]), "distance_ref_m": float(L[n_ref]),
                "distance_speech_range_m": [float(L[n0:n1][valid[n0:n1]].min()), float(L[n0:n1][valid[n0:n1]].max())],
                "delay_swing_smp": float(np.ptp(D[n0:n1][valid[n0:n1]])),
                "D": D, "valid": valid})
    al["placement_source"] = ("OptiTrack geometry: drive warped by the tracked mic-0-to-loudspeaker delay, "
                              "constant fitted by GCC-PHAT over the whole speech")
    return fit


def validate_geometry(array_ch, source_sig, D_rel, valid, span, pre, fs):
    """Where does sourceSig's direct path land AT THE REFERENCE POSE? The
    whole-speech, motion-compensated counterpart of validate_placement:
    sourceSig is warped by the tracked delay change relative to the
    reference, so a correct placement puts the peak at `pre` throughout."""
    n0, n1 = span
    n = np.arange(n0, n1).astype(float)
    m = valid[n0:n1]
    u = np.interp(n - D_rel[n0:n1], np.arange(len(source_sig)), source_sig, left=0.0, right=0.0) * m
    # phat_lag(x, y): y[j] ~ x[j + k]. With array[n] ~ u[n - pre], x = array
    # and y = u give k = +pre for a correct placement.
    k, pk = phat_lag(array_ch[n0:n1] * m, u, fs)
    return int(k), float(pk)
