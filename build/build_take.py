#!/usr/bin/env python3
"""Build Deppisch-format take files from Aria VRS + MPS + drive signals.

    python build/build_take.py                      # every take in config/rooms.json
    python build/build_take.py office_maleSpeech_headMove corridor_femaleSpeech_static
    python build/build_take.py --room corridor

Composite-protocol rooms (clicks -> sweep -> speech in one played file, the
wearer standing still on the mark from the clicks until a few seconds into
the speech) are handled by three config spans: `speech_span_s` (the speech
inside the drive file, used for alignment), `source_span_s` (the portion kept
as sourceSig, so the take is trimmed the same way as the non-composite rooms)
and `sweep_span_s`, which yields a 7-channel array RIR in the `IRs` field --
the array-side sweep, recorded inside the take and therefore in the same MPS
frame as the pose.

Output per take: data/rooms/<take>.mat with exactly the fields
estimateRirs.m reads --

    arraySigs [N x 7]      Aria audio, trimmed to the drive-signal span
    sourceSig [N x 1]      loudspeaker drive, aligned (no clock-rate correction)
    fs                     48000
    silenceData [M x 7]    quietest 2 s of the take outside [drive start, speech end + 1 s]
    fAxNoise, noisePower   8192-point block-averaged |FFT|^2 of silenceData
    referencePos [1 x 3]   zeros: the frame origin IS the reference pose
    referenceRotationAziEleRad [1 x 2]  [0, pitch at reference]
    rotationYawPitchRollRad [N x 3]     array pose per audio sample
    translationXYZCart [N x 3]          metres, frame W' (see pose_mps.py)
    info                   provenance string

plus fields the paper did not have: sourceSigSm57 (near-field reference
mic, same placement), mpsQuality [N x 1], and a `meta` struct with the
alignment/frame numbers. A JSON sidecar and a QA figure go to results/build/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noise import noise_power, pick_silence          # noqa: E402
from pose_mps import device_to_array, load_closed_loop, stillest_window, to_reference_frame  # noqa: E402
from source_align import align_source, phat_lag, place, place_at, xcorr_lag  # noqa: E402
from vrs_audio import device_to_cpf, load_audio        # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
from paths import CFG, RAW  # noqa: E402

# Room-level settings a take may override (see build_take's rc.update).
PER_TAKE_KEYS = ("drive", "sm57", "speech_span_s", "source_span_s", "sweep_span_s")


def read_mono(path: Path) -> tuple[np.ndarray, float]:
    x, fs = sf.read(path, always_2d=True)
    return x[:, 0].astype(np.float64), float(fs)     # all GT files are dual mono


def build_take(room: str, stem: str, out_dir: Path, qa_dir: Path, cache_dir: Path) -> dict:
    rc, tc = dict(CFG["rooms"][room]), CFG["rooms"][room]["takes"][stem]
    # A take may override any of the room-level recording settings. The corridor
    # needs it: its static and head-moving takes played the old 66 s speech-only
    # file, while the walking takes were re-recorded with the 118 s composite
    # (clicks -> sweep -> speech), so protocol is a property of the TAKE there,
    # not of the room.
    rc.update({k: v for k, v in tc.items() if k in PER_TAKE_KEYS})
    vrs = RAW / rc["aria_dir"] / tc["vrs"]
    mps = RAW / rc["aria_dir"] / (tc.get("mps") or f"mps_{Path(tc['vrs']).stem}_vrs")
    t0 = time.time()
    print(f"\n== {stem}  ({vrs.name})")

    # -- audio -------------------------------------------------------------
    x, ts_ns, fs = load_audio(vrs, cache_dir)
    fs = 48000.0 if abs(fs - 48000) < 1 else fs
    print(f"  audio {x.shape[0] / fs:.1f} s, peak {20 * np.log10(np.abs(x).max()):.1f} dBFS  [{time.time() - t0:.0f} s]")

    # -- source-side signals: drive (what was played) and SM57 (at the speaker)
    drive, fsd = read_mono(RAW / rc["drive"][tc["speaker"]])
    assert fsd == fs
    # The SM57 at the loudspeaker is optional: it is an ALTERNATIVE informed
    # reference (doseResponse.m), not an input the pipeline needs. The corridor
    # walking takes have none -- the composite near-field recording was only
    # made in the reverb room -- and borrowing another room's would put that
    # room's early reflections into a corridor take's reference.
    sm57 = None
    if rc.get("sm57") and rc["sm57"].get(tc["speaker"]):
        sm57, fss = read_mono(RAW / rc["sm57"][tc["speaker"]])
        assert fss == fs
    k_rough, _ = xcorr_lag(x[:, 0], drive[: int(20 * fs)])
    span = slice(max(k_rough, 0), max(k_rough, 0) + len(drive))
    ch = int(np.argmax(np.sqrt(np.mean(x[span] ** 2, 0))))   # loudest channel
    speech_span = tuple(rc.get("speech_span_s", (4.0, 64.0)))
    al = align_source(x[:, ch], drive, fs, speech_span_s=speech_span)
    print(f"  drive: direct path at {al['lag_direct'] / fs:.3f} s, ref-window PHAT peak/std {al['ref_window_peak_ratio']:.0f}, "
          f"speech-wide PHAT inliers {al['phat_inlier_windows']}/{al['phat_n_windows']} spread {al['phat_inlier_spread_samples']:.0f} smp"
          f", coarse corr {al['corr_coarse']:.2f}")
    # The SM57 sits on the laptop clock with the drive: its offset to the drive
    # is a fixed chain latency, measured file-to-file by PHAT (a 1000-sigma
    # peak), so it is placed from the drive's lag rather than aligned to the
    # Aria on its own (which can latch onto a reflection on moving takes).
    k_sm, r_sm = (0, float("nan"))
    if sm57 is not None:
        k_sm, r_sm = phat_lag(drive, sm57, fs)        # sm57[j] ~ drive[j + k_sm]
        print(f"  sm57: file starts {k_sm / fs:+.3f} s into the playback (PHAT peak/std {r_sm:.0f})")
    else:
        print("  sm57: none recorded for this take; sourceSigSm57 omitted")
    # -- pose --------------------------------------------------------------
    R_dc, t_dc = device_to_cpf(vrs)
    pose_raw = load_closed_loop(mps)
    # The MPS folder must belong to THIS recording. Both are on the device
    # clock, so a genuine pair overlaps almost completely; a folder from a
    # different take does not overlap at all. Meeting_room_male_full_moving
    # shipped paired with a trajectory 21 s after its audio ended, which would
    # have produced a take built on someone else's head movements.
    a0, a1 = ts_ns[0] / 1e9, ts_ns[-1] / 1e9
    overlap = min(a1, pose_raw.t_s[-1]) - max(a0, pose_raw.t_s[0])
    if overlap < 0.9 * (a1 - a0):
        raise RuntimeError(
            f"{mps.name} covers {pose_raw.t_s[0]:.0f}-{pose_raw.t_s[-1]:.0f} s but the audio is "
            f"{a0:.0f}-{a1:.0f} s (overlap {overlap:.0f} s of {a1 - a0:.0f} s) -- this MPS output "
            f"is from a different recording")
    pose = device_to_array(pose_raw, R_dc, t_dc)
    # OptiTrack (L-shape only): tracked geometry for placement and/or pose.
    # The link constants are fixed in config; only the clock offset is fitted.
    geo = geo_fit = None
    if tc.get("placement") == "optitrack_geometry" or tc.get("pose_source") == "optitrack":
        import geometry_placement as gp
        geo = gp.tracked_geometry(rc, tc, RAW, vrs, pose_raw)
        print(f"  OptiTrack {geo['csv']}: clock offset {geo['clk']['tau_s']:+.4f} s (sync corr "
              f"{geo['clk']['coarse_corr']:.2f}); vs MPS median {geo['vs_mps_pos_median_cm']:.1f} cm / "
              f"{geo['vs_mps_rot_median_deg']:.1f} deg (p90 {geo['vs_mps_pos_p90_cm']:.1f} cm / "
              f"{geo['vs_mps_rot_p90_deg']:.1f} deg)")
    t_sp0 = ts_ns[0] / 1e9 + al["speech_start_take_s"]
    t_sp1 = ts_ns[0] / 1e9 + al["speech_end_take_s"]
    irs = sweep_info = None
    if "sweep_span_s" in rc:
        # Composite take. The sweep gives the REFERENCE POSE (the wearer stands
        # on the mark facing the loudspeaker throughout it) and the array RIR.
        # The speech is placed from the SPEECH, as in the non-composite rooms,
        # and the placement is validated below either way.
        #
        # (An earlier comment here claimed the sweep could not be used to place
        # the speech, on a measurement of "+49.4 / +10.9 / -0.2 ms" across three
        # takes. That measurement was wrong -- it came from circular tooling.
        # Measured properly over all 12 composite takes the speech sits at a
        # mean of -0.05 ms from the sweep with a 0.38 ms spread, i.e. they ARE
        # one rigid block, as a single played file must be. Placing from the
        # speech is still the right choice: it needs no assumption about the
        # file's internal layout. But the sweep would work too.)
        irs, sweep_info = array_ir_from_sweep(x, drive, rc["sweep_span_s"], fs)
        sw0 = ts_ns[0] / 1e9 + (al["lag_place"] + rc["sweep_span_s"][0] * fs) / fs
        sw1 = ts_ns[0] / 1e9 + (al["lag_place"] + rc["sweep_span_s"][1] * fs) / fs
        t_ref, yaw_spread, disp = stillest_window(pose, sw0, sw1)
        place_at(al, min(max(t_sp0, t_sp0 + 2.5), t_sp1 - 2.5) - ts_ns[0] / 1e9, fs)
        al["placement_source"] = "speech (GCC-PHAT); reference pose and IRs from the in-take sweep"
    else:
        # Keep 2.5 s of margin so every placement window (up to 3 s, offset
        # +/-0.5 s) stays inside the speech; a window straddling the silence
        # before it finds no direct path at all.
        if "ref_time_s" in tc:
            # A fixed reference instant (seconds from the start of the
            # recording) instead of the stillest window: on old-protocol
            # walking takes the stillest moment can have the wearer facing
            # away from the loudspeaker. The 2 s window is centred on it.
            t_ref = ts_ns[0] / 1e9 + float(tc["ref_time_s"])
            w = (pose.t_s >= t_ref - 1.0) & (pose.t_s < t_ref + 1.0)
            yw = np.unwrap(np.arctan2(pose.R[w, 1, 0], pose.R[w, 0, 0]))
            yaw_spread = float(np.degrees(np.ptp(yw)))
            disp = float(np.linalg.norm(pose.p[w] - pose.p[w][0], axis=1).max())
        else:
            t_ref, yaw_spread, disp = stillest_window(pose, t_sp0 + 2.5, t_sp1 - 2.5)
        if tc.get("placement") == "optitrack_geometry":
            # The wearer never stands still, so a window measures a blur (or a
            # reflection). The tracked mic-0-to-loudspeaker delay removes the
            # motion; see geometry_placement.py. t_ref is still chosen from
            # the MPS pose, so the SLAM- and OptiTrack-pose versions of a take
            # share the same reference instant and the same placement.
            geo_fit = gp.place_from_geometry(al, geo, x[:, 0], ts_ns / 1e9, t_ref, fs)
            print(f"  geometry placement: whole-speech peak/std {geo_fit['peak_ratio']:.0f}, halves agree to "
                  f"{geo_fit['K_half2'] - geo_fit['K_half1']:+d} smp, per-second residuals IQR "
                  f"{geo_fit['residual_iqr_smp']:.0f} smp ({geo_fit['residual_within_24_pct']:.0f}% within +/-24, "
                  f"{geo_fit['residual_windows_sharp']}/{geo_fit['residual_windows']} windows); distance "
                  f"{geo_fit['distance_speech_range_m'][0]:.2f}-{geo_fit['distance_speech_range_m'][1]:.2f} m, "
                  f"{geo_fit['distance_ref_m']:.2f} m at the reference")
        else:
            place_at(al, t_ref - ts_ns[0] / 1e9, fs)
            al["placement_source"] = "speech (GCC-PHAT)"
    if tc.get("pose_source") == "optitrack":
        pose = device_to_array(gp.hybrid_pose(geo, pose_raw), R_dc, t_dc)
    ref_mask = (pose.t_s >= t_ref - 1.0) & (pose.t_s < t_ref + 1.0)
    if ref_mask.sum() < 100:
        raise RuntimeError("reference window has no MPS samples")
    pose_w, frame = to_reference_frame(pose, ref_mask)

    # -- trim to the source span and place the sources --------------------
    # source_span_s selects the part of the drive file kept as sourceSig. For a
    # composite file that is the speech portion only, so every room's take file
    # has the same structure; the sweep is extracted separately below.
    s0, s1 = (0, len(drive))
    if "source_span_s" in rc:
        s0 = int(round(rc["source_span_s"][0] * fs))
        s1 = int(round(rc["source_span_s"][1] * fs))
        s0, s1 = max(s0, 0), min(s1, len(drive))
    a = max(al["lag_place"] + s0, 0)
    b = min(al["lag_place"] + s1, x.shape[0])
    if a != al["lag_place"] + s0 or b != al["lag_place"] + s1:
        print("  WARNING: source span extends beyond the recording; trimmed")
    arraySigs = x[a:b]
    n = arraySigs.shape[0]
    sourceSig = place(drive[s0:s1], al["lag_place"] + s0 - a, n)
    # SM57 -> drive index space first (it may not start with the playback: on
    # the composite takes the DAW armed it 28 s in), then take the same span.
    sourceSigSm57 = None
    if sm57 is not None:
        sm57_on_drive = place(sm57, k_sm, len(drive))
        sourceSigSm57 = place(sm57_on_drive[s0:s1], al["lag_place"] + s0 - a, n)
    meta_speech_start = al["speech_start_take_s"] - a / fs
    meta_speech_end = al["speech_end_take_s"] - a / fs
    t_audio = ts_ns[a:b] / 1e9
    if t_audio[0] < pose.t_s[0] - 0.5 or t_audio[-1] > pose.t_s[-1] + 0.5:
        print(f"  WARNING: audio span {t_audio[0]:.1f}-{t_audio[-1]:.1f} s exceeds MPS "
              f"{pose.t_s[0]:.1f}-{pose.t_s[-1]:.1f} s; pose clamped at the ends")
    print(f"  reference window at {t_ref - ts_ns[0] / 1e9 - a / fs:.1f} s of the trimmed take: yaw spread {yaw_spread:.0f} deg, "
          f"moved {disp:.2f} m; placement from {al['placement_source']}"
          + ("" if (sweep_info is not None or geo_fit is not None) else
             f", PHAT peak/std {al['ref_window_peak_ratio']:.0f}, spread {al['ref_window_offset_spread']:.0f} smp "
             f"over {al['ref_window_used']}/{al['ref_window_n']} usable windows"))

    # 1 kHz -> audio rate: interpolate angles (yaw unwrapped) and position
    ypr = pose_w.yaw_pitch_roll()
    ypr[:, 0] = np.unwrap(ypr[:, 0])
    tq = np.clip(t_audio, pose_w.t_s[0], pose_w.t_s[-1])
    ypr_a = np.stack([np.interp(tq, pose_w.t_s, ypr[:, k]) for k in range(3)], 1)
    ypr_a[:, 0] = np.angle(np.exp(1j * ypr_a[:, 0]))
    pos_a = np.stack([np.interp(tq, pose_w.t_s, pose_w.p[:, k]) for k in range(3)], 1)
    qual_a = np.interp(np.clip(t_audio, pose_raw.t_s[0], pose_raw.t_s[-1]), pose_raw.t_s, pose_raw.quality)
    opti_a = (np.interp(tq, pose_w.t_s, pose_w.quality) > 0.999).astype(float) \
        if tc.get("pose_source") == "optitrack" else None
    ref_ypr = np.stack([np.angle(np.exp(1j * ypr[ref_mask, 0])).mean(), ypr[ref_mask, 1].mean(), ypr[ref_mask, 2].mean()])
    print(f"  pose: yaw range {np.degrees(ypr_a[:, 0].min()):+.0f}..{np.degrees(ypr_a[:, 0].max()):+.0f} deg, "
          f"max horizontal excursion {np.hypot(pos_a[:, 0], pos_a[:, 1]).max():.2f} m, "
          f"MPS quality<1 on {100 * np.mean(qual_a < 1):.1f}% of samples"
          + ("" if opti_a is None else
             f"; POSE FROM OPTITRACK on {100 * opti_a.mean():.1f}% of samples "
             f"({100 * opti_a[max(int(meta_speech_start * fs), 0):max(min(int(meta_speech_end * fs), n), 1)].mean():.1f}% of the speech kept), "
             f"MPS-filled elsewhere"))

    if sweep_info is not None:
        i_s = int(np.argmin(np.abs(pose_w.t_s - sum(sweep_info["takeSpanS"]) / 2 - ts_ns[0] / 1e9)))
        i_r = int(np.argmin(np.abs(pose_w.t_s - t_ref)))
        ys = np.arctan2(pose_w.R[i_s, 1, 0], pose_w.R[i_s, 0, 0])
        yr = np.arctan2(pose_w.R[i_r, 1, 0], pose_w.R[i_r, 0, 0])
        sweep_info["yawDiffDeg"] = float(np.degrees(np.angle(np.exp(1j * (ys - yr)))))
        sweep_info["posDiffM"] = float(np.linalg.norm(pose_w.p[i_s] - pose_w.p[i_r]))
        print(f"  IRs from the in-take sweep: DRR {sweep_info['drrDb']:+.1f} dB, per-mic onset spread "
              f"{max(sweep_info['perMicOnsetSamples']) - min(sweep_info['perMicOnsetSamples'])} smp, "
              f"sweep vs reference pose {sweep_info['yawDiffDeg']:+.1f} deg / {100 * sweep_info['posDiffM']:.0f} cm")

    # -- validate the placement we are about to write -----------------------
    # Measure where sourceSig's direct path actually lands in arraySigs. It
    # must sit at pre_delay. Checked at two search widths: a genuine lock
    # gives the same answer at both, while a spurious PHAT peak moves with the
    # window. This is the check that catches a mis-placed source before it
    # reaches the pipeline, where it shows up only as a collapsed RLS.
    vlag, vpk, vstable = validate_placement(arraySigs[:, 0], sourceSig, fs,
                                            al["ref_window_centre_take_s"] - a / fs)
    off = vlag - al["pre_delay_samples"]
    ok = abs(off) <= 24 and vstable and vpk >= 8
    if geo_fit is not None:
        # On a walking take the 6 s window above spans a changing distance, so
        # it is reported for context only. The gate is the motion-compensated
        # check over the whole speech, plus the two geometry checks.
        print(f"  window check (context only -- it blurs over the walk): {off:+d} smp, peak/std {vpk:.0f}")
        pre = al["pre_delay_samples"]
        D_rel = geo_fit["D"][a:b] - geo_fit["D_ref_smp"]
        # clamped to the take: a take trimmed to a sub-span (source_span_s) can
        # start or end mid-speech
        span = (max(int((meta_speech_start + 1.0) * fs), 0), min(int((meta_speech_end - 1.0) * fs), n))
        vlag, vpk = gp.validate_geometry(arraySigs[:, 0], sourceSig, D_rel, geo_fit["valid"][a:b], span, pre, fs)
        off = vlag - pre
        halves = geo_fit["K_half2"] - geo_fit["K_half1"]
        ok = (abs(off) <= 24 and vpk >= 8 and abs(halves) <= 24
              and geo_fit["residual_within_24_pct"] >= 80)
        vstable = abs(halves) <= 24
        geo_fit["validation_offset_smp"], geo_fit["validation_peak_ratio"] = int(off), float(vpk)
    print(f"  placement check: direct path at {vlag / fs * 1e3:+.2f} ms "
          f"({off:+d} smp from the {al['pre_delay_samples'] / fs * 1e3:.0f} ms target), "
          f"peak/std {vpk:.0f}, stable: {vstable} -> {'OK' if ok else 'FAILED'}")
    if not ok:
        raise RuntimeError(f"{stem}: sourceSig placement failed validation "
                           f"(direct path {off:+d} samples from target, peak/std {vpk:.0f}, "
                           f"stable={vstable}) -- refusing to write a take the pipeline cannot use")

    # -- noise -------------------------------------------------------------
    silence, i_sil = pick_silence(x, fs, (a, a + int((meta_speech_end + 1.0) * fs)))
    fAx, nP = noise_power(silence, fs)
    print(f"  silence from t={i_sil / fs:.1f} s, rms {20 * np.log10(silence.std()):.1f} dBFS")

    # -- write -------------------------------------------------------------
    info = (f"Aria Gen 1 glasses, 7 mics, {room} room, take {tc['vrs']}. Loudspeaker: Genelec, "
            f"EARS {tc['speaker']} speech (60 s). Pose from "
            + ("OptiTrack (Motive 'glasses' body mapped to the device by a hand-eye rotation and lever "
               "arm fitted on the room's head-moving takes; gaps filled from MPS, see poseIsOptitrack)"
               if tc.get("pose_source") == "optitrack" else "Aria MPS closed-loop SLAM")
            + ", expressed in "
            f"the array frame (+x forward, +y left, +z up) relative to the mean pose over "
            + ("the in-take sweep, during which the wearer stood on the mark facing the loudspeaker"
               if sweep_info is not None else "the stillest 2 s of the speech")
            + f" (meta.ref_window_centre_s). sourceSig = loudspeaker drive signal; sourceSigSm57 = SM57 at "
            f"the loudspeaker; both placed from the {al['placement_source']} so their direct-path arrival at "
            f"the array, at the reference pose, is at {al['pre_delay_samples'] / fs * 1e3:.0f} ms. "
            f"Built {time.strftime('%Y-%m-%d')} by aria_rir/build/build_take.py.")
    meta = {"vrs": tc["vrs"], "mps_dir": mps.name, "speaker": tc["speaker"],
            "trim_start_sample": int(a), "trim_end_sample": int(b),
            "speech_span_s": [al["speech_start_take_s"] - a / fs, al["speech_end_take_s"] - a / fs],
            "pre_delay_samples": al["pre_delay_samples"],
            "drive": {k: v for k, v in al.items() if k not in ("src", "take_hpf", "src_hpf")},
            "placement_source": al["placement_source"],
            "ref_window_centre_s": (t_ref - ts_ns[0] / 1e9) - a / fs,
            "ref_window_yaw_spread_deg": yaw_spread, "ref_window_displacement_m": disp,
            "sm57_lag_vs_drive_samples": int(k_sm), "sm57_phat_peak_ratio": r_sm,
            "align_channel": ch, "silence_start_s": i_sil / fs,
            "mps_quality_below1_fraction": float(np.mean(qual_a < 1)),
            "mps_quality_min": float(qual_a.min()),
            "R_device_cpf": R_dc.tolist(), "t_device_cpf": t_dc.tolist(),
            "pose_source": tc.get("pose_source", "mps"),
            **({"optitrack": {"csv": geo["csv"], "clock_offset_s": geo["clk"]["tau_s"],
                              "sync_corr": geo["clk"]["coarse_corr"],
                              "R_rb_dev": rc["optitrack_link"]["R_rb_dev"], "l_rb_m": rc["optitrack_link"]["l_rb_m"],
                              "speaker_point_m": geo["src"].tolist(), "mic0_device_m": geo["mic0"].tolist(),
                              "vs_mps": {k: geo[k] for k in geo if k.startswith("vs_mps")},
                              **({"pose_fraction_take": float(opti_a.mean())} if opti_a is not None else {}),
                              **({"placement": {k: v for k, v in geo_fit.items() if k not in ("D", "valid")}}
                                 if geo_fit is not None else {})}} if geo is not None else {}),
            **({"sweep": sweep_info} if irs is not None else {}), **frame}
    out = {"arraySigs": arraySigs, "sourceSig": sourceSig[:, None],
           **({"sourceSigSm57": sourceSigSm57[:, None]} if sourceSigSm57 is not None else {}),
           "fs": float(fs), "silenceData": silence, "fAxNoise": fAx[:, None], "noisePower": nP[:, None],
           "referencePos": np.zeros((1, 3)), "referenceRotationAziEleRad": np.array([[0.0, ref_ypr[1]]]),
           **({"IRs": irs} if irs is not None else {}),
           "rotationYawPitchRollRad": ypr_a, "translationXYZCart": pos_a, "mpsQuality": qual_a[:, None],
           **({"poseIsOptitrack": opti_a[:, None]} if opti_a is not None else {}),
           "info": info, "meta": meta}
    out_dir.mkdir(parents=True, exist_ok=True)
    sio.savemat(out_dir / f"{stem}.mat", out, do_compression=False, oned_as="column")
    qa_dir.mkdir(parents=True, exist_ok=True)
    json.dump(meta, open(qa_dir / f"{stem}.json", "w"), indent=2)
    qa_figure(stem, arraySigs, sourceSig, fs, ypr_a, pos_a, qual_a, meta, qa_dir / f"{stem}.png")
    print(f"  wrote {stem}.mat ({n / fs:.1f} s)  [{time.time() - t0:.0f} s]")
    return meta


def validate_placement(take_ch, src, fs, centre_s, target_win_s=6.0):
    """Where does src's direct path actually land in take_ch? Returns
    (lag_samples, peak/std, stable) where `stable` means two search widths
    agree -- a real lock does, a noise peak does not.

    Centred on the window the placement was MEASURED over, and only 6 s wide:
    on a walking take the direct-path delay genuinely moves (0.5 m is 70
    samples), so a 20 s window centred elsewhere in the take measures the
    wearer's motion as if it were a placement error."""
    from source_align import _hpf, phat_window_lags
    xt, ys = _hpf(take_ch, fs), _hpf(src, fs)
    # Keep the window (plus the widest search) inside the signal: a reference
    # near either end of the take -- e.g. one trimmed to a sub-span -- would
    # otherwise run off it and return NaN. Changes nothing for a reference
    # further than ~3.2 s from both ends.
    edge = target_win_s / 2 + 8000 / fs
    centre = min(max(centre_s, edge), len(take_ch) / fs - edge)
    out = [phat_window_lags(xt, ys, 0, fs, [centre], win_s=target_win_s, search=s)[0]
           for s in (4000, 8000)]
    # "Stable" = the two widths find the same peak to within 2 samples (0.04 ms).
    # Exact equality was too strict: a direct path falling between two samples
    # gives two equal-height neighbours, and which one wins can flip with the
    # search width (reverb_maleSpeech_headMove: +478 vs +479, both 14.6). The
    # test exists to catch a spurious peak that moves by hundreds of samples.
    return int(out[0][1]), float(max(o[2] for o in out)), bool(abs(out[0][1] - out[1][1]) <= 2)


def direct_peak(ir, fs, search_s=0.002):
    """The direct path's sample in a deconvolved sweep IR. The sweep is located by cross-correlation with the
    recording, so its direct path sits at the start of the IR (sample 0 or 1 on every composite take); the
    search covers only the first search_s, ahead of the earliest reflection (the floor echo, 2.5 ms or more
    behind). Searched over the first 0.5 s, the strongest sample was a reflection on two takes
    (office_femaleSpeech_static at 8.1 ms, reverb_femaleSpeech_static at 11.4 ms), which put the cut after
    the direct path and spread the per-mic onsets over ~200 samples. (Band-passing the IR before the search
    does not work: the filter's edge padding moves a peak at sample 0 by 15-20 samples.)"""
    return int(np.argmax(np.abs(ir[: int(search_s * fs)])))


def array_ir_from_sweep(x, drive, sweep_span_s, fs, ir_len_s=3.0, pre_s=0.005, tail_s=3.0):
    """7-channel array RIR from the sweep played inside the take.

    The wearer stands on the mark facing the loudspeaker for the whole sweep,
    so this is the array-side reference the omni-on-a-stand cannot be: same
    microphones, same head, same position, same MPS frame as the take.

    It does not place the speech; each signal is located by itself. (An
    earlier note here said the sweep and speech do not sit at a fixed offset
    in the played file. That was a tooling error: measured properly over all
    12 composite takes they agree to -0.05 ms, spread 0.38 ms.)
    """
    from build_reference import deconvolve
    from source_align import _hpf, xcorr_lag
    k0 = int(round(sweep_span_s[0] * fs))
    k1 = int(round(sweep_span_s[1] * fs))
    exc = drive[k0:k1]
    # Locate the sweep BY THE SWEEP rather than from the speech's lag: the
    # deconvolution needs the segment aligned to its own excitation to within
    # a sample, which no inherited offset guarantees. (The smearing first
    # blamed on a non-rigid sweep/speech offset -- DRR 10 dB low, T20 doubled
    # in the office -- came from a double-counted refinement, since fixed.)
    a, _ = xcorr_lag(_hpf(x[:, 0], fs), _hpf(exc, fs))
    b = a + len(exc) + int(tail_s * fs)
    if a < 0 or b > x.shape[0]:
        raise RuntimeError("sweep span falls outside the recording")
    seg = x[a:b]
    irs = np.stack([deconvolve(seg[:, c], exc, fs) for c in range(seg.shape[1])], 1)
    p = direct_peak(irs[:, 0], fs)

    lo = p - int(pre_s * fs)
    out = np.zeros((int(ir_len_s * fs), irs.shape[1]))
    cut = irs[max(lo, 0):lo + len(out)]
    out[max(-lo, 0):max(-lo, 0) + len(cut)] = cut
    kk = int(np.argmax(np.abs(out[:, 0])))
    d0, d1 = max(0, kk - int(0.001 * fs)), kk + int(0.0025 * fs)
    drr = float(10 * np.log10(np.sum(out[d0:d1, 0] ** 2) / np.sum(out[d1:, 0] ** 2)))
    onset = [int(np.argmax(np.abs(out[:, c]) > 0.3 * np.abs(out[:, c]).max())) for c in range(out.shape[1])]
    spread = max(onset) - min(onset)
    if spread > 25:
        print(f"  WARNING: per-mic onset spread {spread} samples exceeds the array's "
              f"physical aperture (~20); the sweep deconvolution is suspect")
    info = {"spanS": list(sweep_span_s), "drrDb": drr, "peakMs": kk / fs * 1e3,
            "sweepLagSamples": int(a), "sweepPeakSamples": int(p),
            "perMicOnsetSpread": int(spread),
            "perMicOnsetSamples": onset,
            "takeSpanS": [a / fs, b / fs]}
    return out, info


def qa_figure(stem, arr, src, fs, ypr, pos, qual, meta, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(arr.shape[0]) / fs
    d = max(1, int(fs / 100))
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0, 0].plot(t[::d], arr[::d, 0], lw=0.3, label="mic 0")
    pk = np.abs(arr[:, 0]).max()
    ax[0, 0].plot(t[::d], src[::d] * pk / (np.abs(src).max() + 1e-12) - 2.2 * pk, lw=0.3, label="sourceSig (scaled, offset)")
    ax[0, 0].axvspan(*meta["speech_span_s"], color="g", alpha=0.08)
    ax[0, 0].set(title=f"{stem}: audio (peak {20 * np.log10(np.abs(arr).max()):.1f} dBFS, PHAT spread {meta['drive']['phat_inlier_spread_samples']:.0f} smp)", xlabel="s")
    ax[0, 0].legend(loc="upper right", fontsize=8)
    ax[0, 1].plot(t[::d], np.degrees(ypr[::d, 0]), label="yaw")
    ax[0, 1].plot(t[::d], np.degrees(ypr[::d, 1]), label="pitch", alpha=0.6)
    ax[0, 1].plot(t[::d], np.degrees(ypr[::d, 2]), label="roll", alpha=0.6)
    ax[0, 1].plot(t[::d], 100 * qual[::d] - 200, "k", lw=0.8, label="MPS quality (x100 - 200)")
    ax[0, 1].axvspan(*meta["speech_span_s"], color="g", alpha=0.08)
    if meta["ref_window_centre_s"] > 0:      # on composite takes the reference is the
        ax[0, 1].axvspan(meta["ref_window_centre_s"] - 1, meta["ref_window_centre_s"] + 1,
                         color="r", alpha=0.25, label="reference window")   # sweep, before the trim
    ax[0, 1].set(title="array orientation (W' frame)", xlabel="s", ylabel="deg"); ax[0, 1].legend(fontsize=8)
    ax[1, 0].plot(pos[::d, 0], pos[::d, 1], lw=0.8)
    s = slice(None, None, int(fs))
    ax[1, 0].quiver(pos[s, 0], pos[s, 1], np.cos(ypr[s, 0]), np.sin(ypr[s, 0]), scale=25, width=0.003, color="r")
    ax[1, 0].plot(0, 0, "ko"); ax[1, 0].set(title="top view: position and forward axis (1/s)", xlabel="x fwd [m]", ylabel="y left [m]"); ax[1, 0].axis("equal")
    ax[1, 1].plot(t[::d], pos[::d, 0], label="x"); ax[1, 1].plot(t[::d], pos[::d, 1], label="y"); ax[1, 1].plot(t[::d], pos[::d, 2], label="z")
    ax[1, 1].set(title="translation from reference", xlabel="s", ylabel="m"); ax[1, 1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("takes", nargs="*", help="take stems; default all")
    ap.add_argument("--room", help="restrict to one room")
    ap.add_argument("--out", default=str(ROOT / "data" / "rooms"))
    args = ap.parse_args()

    todo = [(room, stem) for room, rc in CFG["rooms"].items() for stem in rc["takes"]
            if (not args.room or room == args.room) and (not args.takes or stem in args.takes)
            and stem not in rc.get("exclude", {})]
    summary, failed = {}, []
    for room, stem in todo:
        try:
            summary[stem] = build_take(room, stem, Path(args.out), ROOT / "results" / "build", ROOT / "data" / "cache")
        except FileNotFoundError as e:
            print(f"\n== {stem}: SKIPPED, missing {e}")
            failed.append((stem, f"missing {e}"))
        except RuntimeError as e:
            print(f"\n== {stem}: FAILED -- {e}")
            failed.append((stem, str(e)))
    json.dump(summary, open(ROOT / "results" / "build" / "summary.json", "w"), indent=2)
    if failed:
        print(f"\n{len(failed)} take(s) not written:")
        for stem, why in failed:
            print(f"  {stem}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
