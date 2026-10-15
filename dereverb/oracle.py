#!/usr/bin/env python3
"""Oracle early-reflection candidates: what removing early reflections from the
pseudo-reference would do, measured with the source known (RQ3, step 1).

    python dereverb/oracle.py [<take> ...]      # default: results/dereverb/takes_step1.txt

Reads, from results/dereverb/inputs/ (pipeline/exportInputs.m, with 'received', true):
  <take>_ref_wpe.wav    today's pseudo-reference (WPE, beamformer steered at the protocol bearing)
  <take>_received.wav   the source as it reaches the moving array -- INFORMED

and estimates the reference's impulse response from the received source as
scoreRefSig does (long-window cross-spectral estimate, 100 Hz-8 kHz). Writes to
results/dereverb/candidates/:

  <take>_oracle_early_sub.wav  every early reflection removed by SUBTRACTION: the early
                           part of the reference's response (2.5-20 ms after the direct
                           path) applied to the received source and taken away. Removes
                           exactly the early reflections' contribution and nothing else;
                           the clean answer to 'what if they were not there'
  <take>_oracle_early.wav  the same by INVERSE filtering: the reference filtered by
                           G_direct / G_early (first 20 ms vs the direct path, -1..+2.5 ms),
                           regularised, unchanged outside 100 Hz-8 kHz. Where the early
                           reflections nearly cancel the direct path G_early has notches,
                           and the inverse boosts there (up to +25 dB) everything in the
                           reference that is not a filtered copy of the source: fragile,
                           kept to show it
  <take>_oracle_floor.wav  only the dominant echo 2.5-8 ms after the direct path (the
                           floor reflection) removed, as measured: the same filter with
                           G_early replaced by the direct path plus a +-0.75 ms window
                           around the echo, so the echo goes whatever its shape
  <take>_oracle_comb.wav   the same echo removed with the simple model a practical
                           canceller would use: the reference as d(t) + a d(t - tau),
                           tau and a measured from its response, d recovered recursively,
                           d(t) = y(t) - a d(t - tau)
  <take>_passthrough.wav   the reference unchanged: a check that the harness reproduces
                           frontEndAnchors' wpe_oracle_steering

With --split, instead (the split of today's gap between the reference's fixed and non-fixed parts):

  <take>_oracle_fixed.wav         the FIXED part only: today's whole measured response (from 1 ms
                                  before the direct path to 1.35 s after) applied to the received
                                  source, in 100 Hz-8 kHz; the reference itself outside. Colouring,
                                  early and late reflections kept, nothing that varies over time
  <take>_oracle_perfect_resid.wav the perfect front end (the received source, delayed to the direct
                                  path) plus the NON-FIXED part of today's reference: the reference
                                  minus its fixed part, in 100 Hz-8 kHz (time-varying processing,
                                  noise, and the estimate's own error). The source part is scaled
                                  to the fixed part's in-band power, so the residual keeps today's
                                  ratio to it: the reference's direct path is spread over a few ms
                                  by its colouring, so a least-squares gain onto a flat direct path
                                  would leave the source part ~14 dB weaker than today's and the
                                  residual far stronger than it is

Every candidate keeps the reference's scale (exportInputs' peak normalisation) and
alignment, so evalDereverb's 'auto' input gain applies. oracle_params.csv records the
echo found and each signal's early residual (2.5-20 ms re the direct path, dB).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "results" / "dereverb" / "inputs"
OUT = ROOT / "results" / "dereverb" / "candidates"
BAND = (100.0, 8000.0)
NFFT_IR = 2 ** 17          # 2.7 s: longer than any band's decay, as in scoreRefSig
NF_FILT = 8192             # correction filter length (170 ms)


def band_window(f: np.ndarray) -> np.ndarray:
    """Raised-cosine band-pass with half-octave edges (scoreRefSig's)."""
    lo, hi = BAND
    w = np.zeros_like(f)
    w[(f >= lo) & (f <= hi)] = 1
    m = (f >= lo / np.sqrt(2)) & (f < lo)
    w[m] = 0.5 - 0.5 * np.cos(np.pi * np.log2(f[m] / (lo / np.sqrt(2))) / 0.5)
    m = (f > hi) & (f <= hi * np.sqrt(2))
    w[m] = 0.5 + 0.5 * np.cos(np.pi * np.log2(f[m] / hi) / 0.5)
    return w


def bulk_lag(x: np.ndarray, y: np.ndarray, fs: float) -> int:
    """Lag (samples) of y behind x, within +-50 ms."""
    c = signal.correlate(y - y.mean(), x - x.mean(), mode="full", method="fft")
    mid = len(x) - 1
    m = int(round(0.05 * fs))
    return int(np.argmax(np.abs(c[mid - m: mid + m + 1]))) - m


def impulse_response(x: np.ndarray, y: np.ndarray, fs: float, with_lag: bool = False):
    """Response from x to y, band-limited; returns (h, index of the direct peak), plus the bulk lag
    if asked. The bulk lag is removed before the estimate, so in h lag 0 sits at the centre and
    the direct peak near it; the response at h index i is at lag (i - centre) + bulk lag."""
    lag = bulk_lag(x, y, fs)
    xs, ys = (x[: len(x) - lag], y[lag:]) if lag >= 0 else (x[-lag:], y[: len(y) + lag])
    f, pxy = signal.csd(xs, ys, fs, window="hann", nperseg=NFFT_IR, noverlap=NFFT_IR // 2)
    _, pxx = signal.welch(xs, fs, window="hann", nperseg=NFFT_IR, noverlap=NFFT_IR // 2)
    h = np.roll(np.fft.irfft(pxy / pxx * band_window(f), NFFT_IR), NFFT_IR // 2)
    c0 = NFFT_IR // 2
    w = int(round(0.010 * fs))
    pk = c0 - w + int(np.argmax(np.abs(h[c0 - w: c0 + w + 1])))
    return (h, pk, lag) if with_lag else (h, pk)


def oracle_early_sub(x: np.ndarray, y: np.ndarray, h: np.ndarray, pk: int, lag: int, fs: float) -> np.ndarray:
    """y minus the early reflections' contribution: h's 2.5-20 ms (tapered over 0.25 ms at each end)
    applied to the received source x at its true lag."""
    a, b = pk + int(round(0.0025 * fs)), pk + int(round(0.020 * fs))
    e = h[a:b] * signal.windows.tukey(b - a, 2 * 0.00025 * fs / (b - a))
    d0 = a - NFFT_IR // 2 + lag                    # delay of e's first tap, samples
    conv = signal.fftconvolve(x, e, mode="full")
    c = np.zeros(len(y))
    c[d0:] = conv[: len(y) - d0]
    return y - c


def early_db(h: np.ndarray, pk: int, fs: float) -> float:
    e = lambda a, b: float(np.sum(h[pk + int(round(a * fs)): pk + int(round(b * fs))] ** 2))
    return 10 * np.log10(e(0.0025, 0.020) / e(-0.001, 0.0025))


def correct(y: np.ndarray, h: np.ndarray, pk: int, fs: float, keep_ms: list[tuple[float, float]]) -> np.ndarray:
    """y filtered by G_direct / G_kept, regularised and left unchanged outside the band: G_direct is
    the response's direct path (-1..+2.5 ms), G_kept the direct path plus the windows in keep_ms
    (ms after the direct peak, Hann-tapered), all referenced to the direct peak. Removes from y
    exactly what those windows of its response contain."""
    i0, i1 = pk - int(round(0.001 * fs)), pk + int(round(0.0025 * fs))
    g = np.zeros(NF_FILT); gd = np.zeros(NF_FILT)
    gd[: i1 - i0] = h[i0:i1]
    g[: i1 - i0] = h[i0:i1]
    for a_ms, b_ms in keep_ms:
        j0, j1 = pk + int(round(a_ms * 1e-3 * fs)), pk + int(round(b_ms * 1e-3 * fs))
        g[j0 - i0: j1 - i0] += h[j0:j1] * signal.windows.tukey(j1 - j0, 0.5)
    G, Gd = np.fft.rfft(g), np.fft.rfft(gd)
    return _apply(y, G, Gd, fs)


def oracle_early(y: np.ndarray, h: np.ndarray, pk: int, fs: float) -> np.ndarray:
    """Every early reflection removed: G_kept covers 2.5-20 ms (an untapered join to the direct path)."""
    i0, i1, i2 = pk - int(round(0.001 * fs)), pk + int(round(0.0025 * fs)), pk + int(round(0.020 * fs))
    g = np.zeros(NF_FILT); gd = np.zeros(NF_FILT)
    g[: i2 - i0] = h[i0:i2]
    gd[: i1 - i0] = h[i0:i1]
    G, Gd = np.fft.rfft(g), np.fft.rfft(gd)
    return _apply(y, G, Gd, fs)


def oracle_floor(y: np.ndarray, h: np.ndarray, pk: int, tau: int, fs: float) -> np.ndarray:
    """Only the measured echo (+-0.75 ms around tau) removed."""
    t = tau / fs * 1e3
    return correct(y, h, pk, fs, [(t - 0.75, t + 0.75)])


def _apply(y: np.ndarray, G: np.ndarray, Gd: np.ndarray, fs: float) -> np.ndarray:
    f = np.fft.rfftfreq(NF_FILT, 1 / fs)
    w = band_window(f)
    eps = 1e-3 * np.mean(np.abs(G[w > 0.5]) ** 2)
    C = w * (Gd * np.conj(G) / (np.abs(G) ** 2 + eps)) + (1 - w)
    c = np.roll(np.fft.irfft(C, NF_FILT), NF_FILT // 2)       # zero lag at the centre
    z = signal.fftconvolve(y, c, mode="full")
    return z[NF_FILT // 2: NF_FILT // 2 + len(y)]


def floor_echo(h: np.ndarray, pk: int, fs: float) -> tuple[int, float]:
    """The echo 2.5-8 ms after the direct path with the largest least-squares amplitude against the
    direct-path segment (-1..+2.5 ms): (tau in samples, a)."""
    a0, a1 = int(round(0.001 * fs)), int(round(0.0025 * fs))
    gd = h[pk - a0: pk + a1]
    best = (0, 0.0)
    for tau in range(int(round(0.0025 * fs)), int(round(0.008 * fs)) + 1):
        seg = h[pk + tau - a0: pk + tau + a1]
        a = float(seg @ gd / (gd @ gd))
        if abs(a) > abs(best[1]):
            best = (tau, a)
    return best


def oracle_comb(y: np.ndarray, tau: int, a: float) -> np.ndarray:
    den = np.zeros(tau + 1); den[0] = 1.0; den[tau] = a
    return signal.lfilter([1.0], den, y)


def band_split(y: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """y's parts inside and outside 100 Hz-8 kHz (band_window), zero-phase."""
    Y = np.fft.rfft(y)
    w = band_window(np.fft.rfftfreq(len(y), 1 / fs))
    return np.fft.irfft(Y * w, len(y)), np.fft.irfft(Y * (1 - w), len(y))


def apply_at_lag(x: np.ndarray, e: np.ndarray, d0: int, n: int) -> np.ndarray:
    """e applied to x with e's first tap at delay d0 (samples, either sign), length n."""
    conv = signal.fftconvolve(x, e, mode="full")
    c = np.zeros(n)
    if d0 >= 0:
        c[d0:] = conv[: n - d0]
    else:
        c[:] = conv[-d0: -d0 + n]
    return c


def split_candidates(x: np.ndarray, y: np.ndarray, h: np.ndarray, pk: int, lag: int, fs: float) -> dict:
    n = len(y)
    a0 = pk - int(round(0.001 * fs))
    fixed = apply_at_lag(x, h[a0:], a0 - NFFT_IR // 2 + lag, n)   # h is band-limited: in-band only
    y_in, y_out = band_split(y, fs)
    resid_in = band_split(y - fixed, fs)[0]
    d_dir = pk - NFFT_IR // 2 + lag                                  # delay of the direct path, y re x
    b = apply_at_lag(x, np.array([1.0]), d_dir, n)
    k = np.sqrt(np.mean(fixed ** 2) / np.mean(band_split(b, fs)[0] ** 2))
    return {"oracle_fixed": fixed + y_out, "oracle_perfect_resid": k * b + resid_in}


def mid_coherence(x: np.ndarray, z: np.ndarray, fs: float) -> float:
    """Mean coherence of z with x over 350-1400 Hz (the 500 Hz and 1 kHz octaves), after the bulk lag."""
    lag = bulk_lag(x, z, fs)
    xs, zs = (x[: len(x) - lag], z[lag:]) if lag >= 0 else (x[-lag:], z[: len(z) + lag])
    f, c = signal.coherence(zs, xs, fs, window="hann", nperseg=8192, noverlap=4096)
    return float(np.mean(c[(f >= 354) & (f < 1414)]))


def main_split(takes: list[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in takes:
        y, fs = sf.read(INP / f"{t}_ref_wpe.wav", dtype="float64")
        x, _ = sf.read(INP / f"{t}_received.wav", dtype="float64")
        n = min(len(x), len(y)); x, y = x[:n], y[:n]
        h, pk, lag = impulse_response(x, y, fs, with_lag=True)
        r = {"take": t, "coh_mid_ref": round(mid_coherence(x, y, fs), 3), "early_db_ref": round(early_db(h, pk, fs), 2)}
        for name, z in split_candidates(x, y, h, pk, lag, fs).items():
            sf.write(OUT / f"{t}_{name}.wav", z.astype(np.float32), int(fs), subtype="FLOAT")
            hz, pz = impulse_response(x, z, fs)
            r[f"coh_mid_{name}"] = round(mid_coherence(x, z, fs), 3)
            r[f"early_db_{name}"] = round(early_db(hz, pz, fs), 2)
        rows.append(r)
        print(f"[{t}] mid-band coherence {r['coh_mid_ref']:.2f} -> fixed {r['coh_mid_oracle_fixed']:.2f}, "
              f"perfect+resid {r['coh_mid_oracle_perfect_resid']:.2f}; early residual {r['early_db_ref']:+.1f} -> "
              f"fixed {r['early_db_oracle_fixed']:+.1f}, perfect+resid {r['early_db_oracle_perfect_resid']:+.1f} dB",
              flush=True)
    with open(OUT / "oracle_params_split.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return 0


def main(takes: list[str], only_sub: bool = False) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in takes:
        y, fs = sf.read(INP / f"{t}_ref_wpe.wav", dtype="float64")
        x, fs2 = sf.read(INP / f"{t}_received.wav", dtype="float64")
        assert fs == fs2
        n = min(len(x), len(y)); x, y = x[:n], y[:n]
        h, pk, lag = impulse_response(x, y, fs, with_lag=True)
        tau, a = floor_echo(h, pk, fs)
        cands = {"oracle_early_sub": oracle_early_sub(x, y, h, pk, lag, fs)}
        if not only_sub:
            cands |= {"oracle_early": oracle_early(y, h, pk, fs), "oracle_floor": oracle_floor(y, h, pk, tau, fs),
                      "oracle_comb": oracle_comb(y, tau, a), "passthrough": y}
        r = {"take": t, "tau_ms": round(tau / fs * 1e3, 3), "a": round(a, 4), "early_db_ref": round(early_db(h, pk, fs), 2)}
        for name, z in cands.items():
            if np.max(np.abs(z)) > 1.0:
                print(f"[{t}] {name}: peak {np.max(np.abs(z)):.2f} > 1 (kept as float)")
            sf.write(OUT / f"{t}_{name}.wav", z.astype(np.float32), int(fs), subtype="FLOAT")
            if name != "passthrough":
                hz, pz = impulse_response(x, z, fs)
                r[f"early_db_{name}"] = round(early_db(hz, pz, fs), 2)
        rows.append(r)
        print(f"[{t}] echo at {r['tau_ms']:.2f} ms, a = {a:+.3f}; early residual {r['early_db_ref']:+.1f} dB -> " +
              ", ".join(f"{r[k]:+.1f} ({k[9:]})" for k in r if k.startswith("early_db_oracle")), flush=True)
    with open(OUT / ("oracle_params_sub.csv" if only_sub else "oracle_params.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return 0


if __name__ == "__main__":
    only_sub = "--only-sub" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")] or \
        (ROOT / "results" / "dereverb" / "takes_step1.txt").read_text().split()
    raise SystemExit(main_split(args) if "--split" in sys.argv else main(args, only_sub))
