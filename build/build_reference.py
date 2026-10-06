#!/usr/bin/env python3
"""Reference RIR per room from the omni-mic sweep recordings.

    python build/build_reference.py            # all rooms
    python build/build_reference.py corridor

Writes data/rooms/<room>_reference.mat --

    IRs [N x 1]          3 s room impulse response, mean of the sweeps,
                         direct peak at 5 ms
    IRsPerSweep [N x S]  the individual sweeps, aligned
    fs, info
    fBands [7 x 1]       octave centres 125 Hz .. 8 kHz
    rt20Bands [7 x 1]    T20 per band (s) from Schroeder integration
    drrDb                direct-to-reverberant ratio at the listening position
    distanceM            source-receiver distance from the direct-path delay
    blockLenMsSuggested  smallest multiple of 500 ms >= 2 * max(rt20)
                         (RIR window = block/2 must cover the longest RT)

Not the paper's reference: theirs was an order-10 EMA at the wearer's
position, ours is a single omni at the same position, on a stand where the
wearer stands. So RT *and* DRR transfer -- the measured DRR (office +0.6,
corridor -2.5, reverb -6.3 dB) is what a listener-position mic reads, not a
near-field one. Two caveats when comparing DRR with an estimate: a free-field
omni is not the array's order-0 channel on a head (they agree at low
frequency and diverge where the head shadows), and the stand sat just under
shoulder height (~1.30 m) rather than at head height (~1.62 m). The height
offset is 1.8 cm of path at 2 m -- nothing for distance or direct level --
but it moves the floor-bounce comb notch from 110 to 130 Hz, so the 125 Hz
band is indicative only.

The source-receiver distance also follows: the deconvolution peak minus the
playback chain's electronic latency (CHAIN_MS, calibrated below) is the
propagation delay. Reported as `distanceM`, and checked against the
tape-measured `distance_m` in config/rooms.json when that is present.

Deconvolution is regularised spectral division against the played sweep
file (the exact WAV that was played, so the loudspeaker is inside the IR
just as it is inside the Aria-estimated RIRs).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import soundfile as sf
from scipy.signal import butter, sosfilt

ROOT = Path(__file__).resolve().parents[1]
from paths import CFG, RAW  # noqa: E402
F_BANDS = np.array([125, 250, 500, 1000, 2000, 4000, 8000], dtype=float)
IR_LEN_S, PRE_S = 3.0, 0.005
# Electronic latency of the playback/record chain, drive file -> loudspeaker
# cone (DAC, amp, driver, minus whatever the DAW compensated). CALIBRATED:
# each room's deconvolution peak minus its tape-measured source distance gives
# 1.09 / 1.17 / 1.17 ms (office / corridor / reverb) -- a 0.08 ms spread, i.e.
# 2.7 cm. Do not confuse it with the 2.58 ms lag between the drive file and the
# near-field SM57 recording: that mic sits 0.49 m from the cone, so it carries
# 1.44 ms of propagation on top of this.
CHAIN_MS = 1.14
C = 343.0


def read_mono(path):
    x, fs = sf.read(path, always_2d=True)
    return x[:, 0].astype(np.float64), float(fs)


def deconvolve(y, x, fs, band=(50.0, 16000.0)):
    """h = y * x^-1 by regularised division; tiny loading in band, full outside."""
    N = 1 << int(np.ceil(np.log2(len(x) + len(y))))
    X, Y = np.fft.rfft(x, N), np.fft.rfft(y, N)
    f = np.fft.rfftfreq(N, 1 / fs)
    pmax = np.abs(X).max() ** 2
    eps = np.where((f >= band[0]) & (f <= band[1]), 1e-5 * pmax, pmax)
    return np.fft.irfft(Y * np.conj(X) / (np.abs(X) ** 2 + eps), N)


def cut_at_peak(h, fs):
    k = peak_sample(h, fs)
    a = k - int(PRE_S * fs)
    out = np.zeros(int(IR_LEN_S * fs))
    seg = h[max(a, 0):a + len(out)]
    out[max(-a, 0):max(-a, 0) + len(seg)] = seg
    return out


def peak_sample(h, fs):
    return int(np.argmax(np.abs(h[:int(0.5 * fs)])))


def drr_db(ir, fs, pre_ms=1.0, post_ms=2.5):
    """Direct-to-reverberant ratio: energy in [peak-1 ms, peak+2.5 ms] over
    everything after it. Same window as the pipeline's getDrr."""
    k = int(np.argmax(np.abs(ir)))
    a, b = max(0, k - int(pre_ms / 1000 * fs)), k + int(post_ms / 1000 * fs)
    return float(10 * np.log10(np.sum(ir[a:b] ** 2) / np.sum(ir[b:] ** 2)))


def octave_sos(fc, fs):
    return butter(3, [fc / np.sqrt(2) / (fs / 2), min(fc * np.sqrt(2) / (fs / 2), 0.99)], "band", output="sos")


def t20(h, fs, fc):
    """T20 from the Schroeder curve, truncated where the decay meets the
    noise floor (estimated from the last 10 % of the IR) plus 10 dB."""
    hb = sosfilt(octave_sos(fc, fs), h)
    e = hb ** 2
    n_tail = len(e) // 10
    noise = e[-n_tail:].mean()
    w = int(0.02 * fs)
    sm = 10 * np.log10(np.convolve(e, np.ones(w) / w, "same") + 1e-30)
    thresh = 10 * np.log10(noise + 1e-30) + 10
    k0 = int(np.argmax(sm))
    above = np.where(sm[k0:] < thresh)[0]
    k_tr = k0 + (int(above[0]) if above.size else len(sm) - k0)
    edc = np.cumsum(e[:k_tr][::-1])[::-1]
    edc = 10 * np.log10(edc / edc[0] + 1e-30)
    t = np.arange(len(edc)) / fs
    i5, i25 = np.searchsorted(-edc, 5), np.searchsorted(-edc, 25)
    if i25 >= len(edc) or i25 - i5 < 10:
        return np.nan, float(sm.max() - thresh)
    p = np.polyfit(t[i5:i25], edc[i5:i25], 1)
    return float(-60 / p[0]), float(sm.max() - thresh)


def build_room(room: str, out_dir: Path) -> dict:
    rc = CFG["rooms"][room]
    x, fs = read_mono(RAW / rc["sweep_source"])
    irs, dists = [], []
    for name in rc["sweeps"]:
        y, fsy = read_mono(RAW / rc["gt_dir"] / name)
        assert fsy == fs
        h = deconvolve(y, x, fs)
        dists.append((peak_sample(h, fs) / fs * 1000 - CHAIN_MS) / 1000 * C)
        irs.append(cut_at_peak(h, fs))
    irs = np.stack(irs, 1)
    dist = float(np.mean(dists))
    dist_meas = rc.get("distance_m")
    w = int(0.05 * fs)
    cons = [np.corrcoef(irs[:w, 0], irs[:w, j])[0, 1] for j in range(1, irs.shape[1])]
    ir = irs.mean(1)

    rt, margin = zip(*[t20(ir, fs, fc) for fc in F_BANDS])
    rt = np.array(rt)
    blk = int(np.ceil(2 * np.nanmax(rt) / 0.5) * 500)
    drr = drr_db(ir, fs)
    print(f"== {room}: {irs.shape[1]} sweeps, first-50 ms consistency {np.min(cons):.3f}; "
          f"T20 " + " ".join(f"{int(f)}:{r:.2f}" for f, r in zip(F_BANDS, rt)) +
          f" s (decay range above noise {min(margin):.0f}-{max(margin):.0f} dB) -> blockLenMs >= {blk}; "
          f"DRR {drr:+.1f} dB, distance {dist:.2f} m" +
          (f" (measured {dist_meas:.2f} m, err {100 * (dist - dist_meas):+.0f} cm)" if dist_meas else " (not measured)"))

    info = (f"Omni microphone at the listening position, {room} room, mean of {irs.shape[1]} exponential sweeps "
            f"(50 Hz-16 kHz, 20 s), direct peak at {PRE_S * 1e3:.0f} ms. Regularised spectral "
            f"deconvolution against the played file. DRR {drr:+.1f} dB; source-receiver distance {dist:.2f} m "
            f"from the direct-path delay minus the {CHAIN_MS} ms playback chain latency. "
            f"Built {time.strftime('%Y-%m-%d')} by aria_rir/build/build_reference.py.")
    out_dir.mkdir(parents=True, exist_ok=True)
    sio.savemat(out_dir / f"{room}_reference.mat", {
        "IRs": ir[:, None], "IRsPerSweep": irs, "fs": fs, "info": info,
        "fBands": F_BANDS[:, None], "rt20Bands": rt[:, None], "blockLenMsSuggested": float(blk),
        "drrDb": drr, "distanceM": dist, "distanceMeasuredM": float(dist_meas) if dist_meas else np.nan,
    }, oned_as="column")
    return {"rt20": rt.tolist(), "blockLenMsSuggested": blk, "sweep_consistency": cons,
            "drrDb": drr, "distanceM": dist, "distanceMeasuredM": dist_meas,
            "distancePerSweep": dists}


def main() -> int:
    rooms = sys.argv[1:] or [r for r in CFG["rooms"] if CFG["rooms"][r].get("sweeps")]
    res = {r: build_room(r, ROOT / "data" / "rooms") for r in rooms}
    (ROOT / "results" / "build").mkdir(parents=True, exist_ok=True)
    json.dump(res, open(ROOT / "results" / "build" / "reference.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
