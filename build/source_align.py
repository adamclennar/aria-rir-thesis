"""Place a source-side signal on the Aria clock -> sourceSig.

Two source-side signals exist per take: the loudspeaker drive file and the 
SM57 recording near-field to the loudspeaker.
Both were recorded & played on the laptop's audio clock, not the Aria's, so
for each we need its lag within the take.

Method:
1. Coarse lag: plain FFT cross-correlation over the whole take.
2. Direct-path lag: GCC-PHAT (150-6000 Hz) in 8 s windows across the
   speech. Plain correlation is energy-weighted and in a reverberant room
   flips between the direct path and a strong early reflection (seen on
   the office: two peaks 205 samples apart); PHAT whitening makes the
   direct path the sharpest peak in every window.
3. Clock rate: not corrected. Both devices sample at a nominal 48 kHz from
   independent crystals, which differ but only by a few ppm, so a
   common rate is assumed.
4. Placement: the source is placed so that its DIRECT-PATH ARRIVAL at the
   array sits at `pre_delay_ms` (default 10 ms) in sourceSig time. The
   absolute latency of the playback chain is not recoverable without a
   shared clock, so the RIR's direct peak is put at a known, causal delay
   by construction. The measured lag is kept in the .mat meta.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def _hpf(x: np.ndarray, fs: float, fc: float = 100.0) -> np.ndarray:
    return sosfiltfilt(butter(4, fc / (fs / 2), "high", output="sos"), x, axis=0)


def xcorr_lag(x: np.ndarray, y: np.ndarray) -> tuple[int, float]:
    """Lag k such that y[j] ~ x[k + j]. Returns (k, normalised peak corr)."""
    n = len(x) + len(y) - 1
    N = 1 << int(np.ceil(np.log2(n)))
    c = np.fft.irfft(np.fft.rfft(x, N) * np.conj(np.fft.rfft(y, N)), N)
    lags = np.arange(-len(y) + 1, len(x))
    c = np.concatenate([c[-(len(y) - 1):], c[:len(x)]])
    k = int(lags[np.argmax(c)])
    lo, hi = max(0, k), min(len(x), k + len(y))
    xs, ys = x[lo:hi], y[lo - k:hi - k]
    r = float(np.dot(xs, ys) / (np.linalg.norm(xs) * np.linalg.norm(ys) + 1e-20))
    return k, r


def phat_window_lags(x: np.ndarray, y: np.ndarray, k0: int, fs: float, centres_s,
                     win_s: float = 8.0, search: int = 4000, band=(150.0, 6000.0)):
    """Per-window GCC-PHAT lag offset of y (source) in x (take) around k0.
    Returns array of (centre_s, offset_samples, peak-to-std ratio)."""
    w = int(win_s * fs)
    out = []
    for c in centres_s:
        a = int(c * fs - w // 2)
        ys = y[a:a + w]
        xs = x[k0 + a - search:k0 + a + w + search]
        if len(ys) < w or len(xs) < w + 2 * search:
            out.append((c, np.nan, 0.0)); continue
        n = len(xs) + len(ys) - 1
        N = 1 << int(np.ceil(np.log2(n)))
        G = np.fft.rfft(xs, N) * np.conj(np.fft.rfft(ys, N))
        G /= np.abs(G) + 1e-12
        f = np.fft.rfftfreq(N, 1 / fs)
        G[(f < band[0]) | (f > band[1])] = 0
        cc = np.fft.irfft(G, N)[:2 * search + 1]
        i = int(np.argmax(cc))
        out.append((c, i - search, float(cc[i] / (cc.std() + 1e-20))))
    return np.array(out)


def phat_lag(x: np.ndarray, y: np.ndarray, fs: float, band=(150.0, 6000.0)) -> tuple[int, float]:
    """Whole-signal GCC-PHAT lag k such that y[j] ~ x[k + j]; (k, peak/std)."""
    N = 1 << int(np.ceil(np.log2(len(x) + len(y))))
    G = np.fft.rfft(_hpf(x, fs), N) * np.conj(np.fft.rfft(_hpf(y, fs), N))
    G /= np.abs(G) + 1e-12
    f = np.fft.rfftfreq(N, 1 / fs)
    G[(f < band[0]) | (f > band[1])] = 0
    c = np.fft.irfft(G, N)
    k = int(np.argmax(c))
    return (k if k < N // 2 else k - N), float(c.max() / (c.std() + 1e-20))


def align_source(take: np.ndarray, src: np.ndarray, fs: float,
                 speech_span_s: tuple[float, float] = (4.0, 64.0),
                 pre_delay_ms: float = 10.0, max_dev: float = 30.0,
                 ref_window_s: float = 2.0) -> dict:
    """Align `src` (drive or SM57) to `take` (one Aria channel). The source
    is used at its recorded rate: no clock-rate correction (see above).

    ref_window_s : length of the placement window. The placement lag is
        measured over the window the pose reference is taken from (see
        place_at), so the RIR's direct path sits at pre_delay_ms AT THE
        REFERENCE POSE; as the wearer moves the delay changes and the
        motion-aware model is what accounts for it.

    Returns lag_direct (int: src[0] arrives at the array at take[lag_direct]),
    lag_place (= lag_direct - pre-delay), the source signal, and
    diagnostics."""
    xt, ys = _hpf(take, fs), _hpf(src, fs)
    k0, r0 = xcorr_lag(xt, ys)
    # Direct-path lag in 12 windows across the speech: a diagnostic only. Its
    # spread shows how far the direct-path delay moves over the take (the
    # wearer's motion, plus any clock drift); nothing is fitted to it.
    centres = np.linspace(speech_span_s[0] + 5, speech_span_s[1] - 5, 12)
    wl = phat_window_lags(xt, ys, k0, fs, centres)
    off = wl[:, 1]
    ok = ~np.isnan(off) & (np.abs(off - np.nanmedian(off)) <= max_dev)
    # provisional placement from the first ref_window_s of speech; build_take
    # re-measures it with place_at() once the reference window is chosen
    ref = phat_window_lags(xt, ys, k0, fs, [speech_span_s[0] + ref_window_s / 2],
                           win_s=ref_window_s, search=8000)
    lag_direct = int(round(k0 + ref[0, 1]))
    pre = int(round(pre_delay_ms / 1000 * fs))

    return {"lag_direct": lag_direct, "lag_place": lag_direct - pre, "pre_delay_samples": pre,
            "lag_coarse": k0, "take_hpf": xt, "src_hpf": ys, "src": src, "corr_coarse": r0,
            "ref_window_peak_ratio": float(ref[0, 2]),
            "phat_offsets": off.tolist(), "phat_peak_ratio": wl[:, 2].tolist(),
            "phat_inlier_windows": int(ok.sum()), "phat_n_windows": int(len(off)),
            "phat_inlier_spread_samples": float(np.ptp(off[ok])) if ok.any() else float("nan"),
            "phat_median_minus_ref_samples": float(np.nanmedian(off) - ref[0, 1]),
            "speech_span_src_s": list(speech_span_s),
            "speech_start_take_s": (lag_direct + speech_span_s[0] * fs) / fs,
            "speech_end_take_s": (lag_direct + speech_span_s[1] * fs) / fs}


def place_at(al: dict, centre_take_s: float, fs: float) -> dict:
    """Re-measure the placement lag around take time `centre_take_s` (the
    chosen reference window) and update `al`.

    One window can latch onto a reflection when the wearer is moving (seen
    on an office walking take: +462 samples from the 2 s window, ~+70 from
    every other length and offset), so the lag is the median over a set of
    window lengths (0.5-3 s) and 1 s windows offset by +/-0.5 s; the spread
    of that set is reported."""
    centre_src_s = (centre_take_s * fs - al["lag_coarse"]) / fs
    rows = [phat_window_lags(al["take_hpf"], al["src_hpf"], al["lag_coarse"], fs, [centre_src_s],
                             win_s=w, search=8000)[0] for w in (0.5, 1.0, 2.0, 3.0)]
    rows += [phat_window_lags(al["take_hpf"], al["src_hpf"], al["lag_coarse"], fs, [centre_src_s + dc],
                              win_s=1.0, search=8000)[0] for dc in (-0.5, 0.5)]
    rows = np.array(rows)
    # Drop windows that found no sharp direct path. A window reaching back into
    # pre-speech silence correlates noise against noise and returns a lag that
    # is not merely imprecise but arbitrary -- on the reverberant room's moving
    # takes those pushed the window-to-window spread to ~10000 samples.
    good = rows[rows[:, 2] >= 6.0]
    if len(good) < 3:
        good = rows
    off = np.median(good[:, 1])
    al["lag_direct"] = int(round(al["lag_coarse"] + off))
    al["lag_place"] = al["lag_direct"] - al["pre_delay_samples"]
    al["ref_window_peak_ratio"] = float(np.median(good[:, 2]))
    al["ref_window_offsets"] = rows[:, 1].tolist()
    al["ref_window_offset_spread"] = float(np.ptp(good[:, 1]))
    al["ref_window_used"] = int(len(good))
    al["ref_window_n"] = int(len(rows))
    al["ref_window_centre_take_s"] = float(centre_take_s)
    sp = al["speech_span_src_s"]
    al["speech_start_take_s"] = (al["lag_direct"] + sp[0] * fs) / fs
    al["speech_end_take_s"] = (al["lag_direct"] + sp[1] * fs) / fs
    return al


def place(sig: np.ndarray, lag: int, n_out: int) -> np.ndarray:
    """Zero-padded copy of sig starting at index `lag` in an n_out vector."""
    out = np.zeros(n_out)
    a, b = max(lag, 0), min(lag + len(sig), n_out)
    if b > a:
        out[a:b] = sig[a - lag:b - lag]
    return out
