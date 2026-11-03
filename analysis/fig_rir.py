#!/usr/bin/env python3
"""Background figures: a measured room impulse response and its parts; its DRR and its T20.

    python analysis/fig_rir.py [<room>]        # default: meeting

The room's omni reference at the listening position (data/rooms/<room>_reference.mat, the mean of
the omni sweeps). The meeting room is the default: the cleanest of the references, with distinct
early reflections, a straight decay at 1 kHz and about 85 dB between the peak and the noise floor.
Two figures, each computed the way the scoring computes it. The annotations only name the parts: the
equations are in the thesis text, and this room's values are printed, not drawn:
  impulse response   the first 100 ms, band-passed 100 Hz - 8 kHz (as for DRR): the direct sound,
                     the early reflections and the start of the late reverberation. The early/late
                     boundary is 50 ms, the convention of ISO 3382-1's speech measures (C50)
  DRR window         the paper's direct-sound window, -0.5 to +1 ms around the direct peak (getDrr:
                     4th-order 100 Hz - 8 kHz band-pass, zero phase; the direct peak is the earliest
                     of the three largest peaks in the first 20 ms); everything after it, up to the
                     end of the decay (Lundeby), is the reverberant part
  EDC and T20        the 1 kHz octave band: the Schroeder energy decay curve, integrated back from
                     where the decay meets the noise floor (Lundeby), plus the ISO 3382-1 correction
                     for the energy after that point (the noise level times the late decay's
                     T / (6 ln 10)), as ita_roomacoustics' default 'cutWithCorrection' does; and the
                     line fitted between -5 and -25 dB, extended to a 60 dB fall. Its T20 is within
                     1 ms of the scoring's (getT20Ita).
Writes results/background/fig_rir.png (the impulse response) and fig_rir_measures.png (DRR beside
the EDC and T20).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.io as sio  # noqa: E402
from scipy.signal import butter, find_peaks, sosfiltfilt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "background"
BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, MUTED, GRID, GREY = "#0b0b0b", "#52514e", "#e4e3df", "#b9b7b0"
PRE_MS, POST_MS = 0.5, 1.0           # the paper's direct-sound window (drrWindows.m)
EARLY_MS = 50.0                      # early/late boundary (ISO 3382-1, C50)
FC = 1000.0                          # the octave band drawn for T20


def direct_peak(x: np.ndarray, fs: float) -> int:
    """getDrr's direct peak: the earliest of the three largest peaks of |x| in the first 20 ms."""
    a = np.abs(x[:round(0.02 * fs)])
    pk, _ = find_peaks(a, distance=10)
    return int(pk[np.argsort(a[pk])[::-1][:3]].min())


def lundeby(x: np.ndarray, fs: float) -> tuple[int, float, float]:
    """Where the decay meets the noise floor (after Lundeby et al. 1995, simplified): a line fitted to
    the 20 ms level of the decay, intersected with the noise level, iterated with the noise taken
    from after the intersection. Returns the sample, the noise's mean square and the late decay's
    reverberation time (s, from the last line)."""
    e = x ** 2
    w = round(0.02 * fs)
    n = len(e) // w
    lv = 10 * np.log10(e[:n * w].reshape(n, w).mean(1) + 1e-300)
    ref = lv.max()
    lv -= ref
    tc = (np.arange(n) + 0.5) * w / fs
    after = np.arange(n) > np.argmax(lv)
    noise = 10 * np.log10(e[-len(e) // 10:].mean() + 1e-300) - ref
    for i in range(5):
        fit = after & (lv > noise + (5 if i else 10)) & (lv < (-5 if i else 0))
        p = np.polyfit(tc[fit], lv[fit], 1)
        k = int((noise - p[1]) / p[0] * fs)
        noise = 10 * np.log10(e[min(k + round(0.1 * fs), len(e) - len(e) // 10):].mean() + 1e-300) - ref
    return min(k, len(x)), 10 ** ((noise + ref) / 10), -60 / p[0]


def style(ax) -> None:
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def main() -> int:
    room = sys.argv[1] if len(sys.argv) > 1 else "meeting"
    R = sio.loadmat(ROOT / "data" / "rooms" / f"{room}_reference.mat", squeeze_me=True)
    h, fs = np.asarray(R["IRs"], float), float(R["fs"])

    # DRR, as getDrr computes it
    hb = sosfiltfilt(butter(4, [100 / (fs / 2), 8000 / (fs / 2)], "band", output="sos"), h)
    t0 = direct_peak(hb, fs)
    end, _, _ = lundeby(hb, fs)
    pre, post = round(PRE_MS / 1000 * fs), round(POST_MS / 1000 * fs)
    drr = 10 * np.log10(np.sum(hb[t0 - pre:t0 + post + 1] ** 2) / np.sum(hb[t0 + post + 1:end] ** 2))
    tms = (np.arange(len(hb)) - t0) / fs * 1000          # ms re the direct peak
    y = hb / np.abs(hb[t0])

    # T20 in one octave band: Schroeder EDC to the Lundeby point with the ISO 3382-1 correction, line
    # fitted on -5..-25 dB
    s = sosfiltfilt(butter(3, [FC / np.sqrt(2) / (fs / 2), FC * np.sqrt(2) / (fs / 2)], "band", output="sos"), h)
    s = s[t0:]                                           # from the direct sound
    kb, noise_ms, t_late = lundeby(s, fs)
    corr = noise_ms * t_late / (6 * np.log(10)) * fs     # the energy after the cut, decaying at the late rate
    edc = np.cumsum(s[:kb][::-1] ** 2)[::-1] + corr
    edc = 10 * np.log10(edc / edc[0])
    te = np.arange(kb) / fs
    i5, i25 = int(np.argmax(edc <= -5)), int(np.argmax(edc <= -25))
    slope, icpt = np.polyfit(te[i5:i25], edc[i5:i25], 1)
    t20 = -60 / slope
    lvl = 10 * np.log10(s ** 2 / np.max(s ** 2) + 1e-300)

    OUT.mkdir(parents=True, exist_ok=True)

    # --- figure 1, the impulse response: direct sound, early reflections, late reverberation
    fig, ax = plt.subplots(figsize=(10.4, 3.55), layout="constrained")
    lo, hi = -2.0, 100.0
    m = (tms >= lo) & (tms <= hi)
    for a, b, col in ((-PRE_MS, POST_MS, BLUE), (POST_MS, EARLY_MS, ORANGE), (EARLY_MS, hi, GREEN)):
        ax.axvspan(a, b, color=col, alpha=0.13, lw=0, zorder=0)
    ax.plot(tms[m], y[m], color=INK, lw=0.8, zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-1.0, 1.32)
    ax.text(1.6, 1.0, "direct sound", fontsize=9, color=INK, va="center")
    ax.text((POST_MS + EARLY_MS) / 2, 1.16, "early reflections", fontsize=9, color=INK, ha="center")
    ax.text((EARLY_MS + hi) / 2, 1.16, "late reverberation (the tail)", fontsize=9, color=INK, ha="center")
    # the two strongest early reflections, picked from the response
    a = np.abs(hb)
    win = (tms > 2) & (tms < EARLY_MS)
    refl, _ = find_peaks(np.where(win, a, 0), distance=round(0.002 * fs))
    top = sorted(refl[np.argsort(a[refl])[::-1][:2]])
    tx = tms[top[-1]] + 4
    for k in top:
        ax.annotate("", xy=(tms[k], y[k] + (0.04 if y[k] > 0 else -0.04)), xytext=(tx, 0.62),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9, shrinkA=0, shrinkB=1))
    ax.text(tx + 0.5, 0.62, "discrete reflections", fontsize=8.5, color=MUTED, va="center")
    ax.set_xlabel("Time re the direct sound (ms)", fontsize=9, color=MUTED)
    ax.set_ylabel("Amplitude re direct peak", fontsize=9, color=MUTED)
    style(ax)
    path1 = OUT / "fig_rir.png"
    fig.savefig(path1, dpi=300)
    plt.close(fig)

    # --- figure 2: DRR beside the EDC and its T20 fit
    fig, (ax_b, ax_c) = plt.subplots(1, 2, figsize=(10.4, 4.3), width_ratios=[1, 1.55], layout="constrained")

    lo, hi = -2.0, 12.0
    m = (tms >= lo) & (tms <= hi)
    ax_b.axvspan(-PRE_MS, POST_MS, color=BLUE, alpha=0.18, lw=0, zorder=0)
    ax_b.axvspan(POST_MS, hi, color=GREY, alpha=0.25, lw=0, zorder=0)
    ax_b.plot(tms[m], y[m], color=INK, lw=0.9, zorder=2)
    ax_b.set_xlim(lo, hi)
    ax_b.set_ylim(-1.0, 1.55)
    ax_b.set_xticks([-PRE_MS, 0, POST_MS, 4, 8, 12], ["-0.5", "", "+1", "4", "8", "12"])
    ax_b.text(0.25, 1.38, "direct", fontsize=9, color=INK, ha="center")
    ax_b.text((POST_MS + hi) / 2, 1.3, "reverberant part\n(to the end of the decay)",
              fontsize=9, color=INK, ha="center", va="center")
    ax_b.set_xlabel("Time re the direct peak (ms)", fontsize=9, color=MUTED)
    ax_b.set_ylabel("Amplitude re direct peak", fontsize=9, color=MUTED)
    ax_b.set_title("Direct-to-reverberant ratio", loc="left", fontsize=10, color=INK)
    style(ax_b)

    ax_c.plot(np.arange(len(s)) / fs, lvl, color=GREY, lw=0.4, alpha=0.8, zorder=1)
    ax_c.plot(te, edc, color=INK, lw=2, zorder=3)
    tl = np.array([(-5 - icpt) / slope, (-65 - icpt) / slope])      # the line's -5 and -65 dB: a 60 dB fall
    ax_c.plot(te[[i5, i25]], edc[[i5, i25]], color=PURPLE, lw=2.6, zorder=4)
    ax_c.plot(tl, slope * tl + icpt, color=PURPLE, lw=1.3, ls=(0, (4, 3)), zorder=4)
    ax_c.scatter(te[[i5, i25]], edc[[i5, i25]], s=40, color=PURPLE, edgecolor="white", linewidth=1.2, zorder=5)
    for lev in (-5, -25):
        ax_c.axhline(lev, color=MUTED, lw=0.7, ls=":", zorder=2)
        ax_c.text(1.0, lev, f" {lev} dB", transform=ax_c.get_yaxis_transform(), fontsize=8, color=MUTED,
                  va="center", ha="left")
    ax_c.text(te[i25] + 0.03, edc[i25] + 2.5, "line fitted\nfrom -5 to -25 dB", fontsize=8.5, color=INK,
              va="bottom")
    # T20: the time the fitted line takes to fall 60 dB
    for x in tl:
        ax_c.plot([x, x], [slope * x + icpt, -73], color=MUTED, lw=0.7, ls=":", zorder=2)
    ax_c.annotate("", xy=(tl[1], -73), xytext=(tl[0], -73),
                  arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1, shrinkA=0, shrinkB=0))
    ax_c.text(tl.mean(), -76, r"$T_{20} = 60\,\mathrm{dB}\ /\ |\mathrm{slope\ of\ the\ fitted\ line}|$", fontsize=10,
              color=INK, ha="center", va="top",
              bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.85))
    k1 = int(np.argmin(np.abs(te - 1.2)))
    ax_c.annotate("energy decay curve", xy=(te[k1], edc[k1]), xytext=(1.1, -52), fontsize=9, color=INK,
                  arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, shrinkB=2))
    ax_c.annotate("squared impulse response", xy=(0.58, -62), xytext=(0.8, -33), fontsize=9, color=MUTED,
                  arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8, shrinkA=2, shrinkB=0))
    ax_c.text(1.56, -80, "noise floor", fontsize=8.5, color=MUTED, ha="center")
    ax_c.set_xlim(0, 1.7)
    ax_c.set_ylim(-100, 3)
    ax_c.set_xlabel("Time re the direct sound (s)", fontsize=9, color=MUTED)
    ax_c.set_ylabel("Level (dB)", fontsize=9, color=MUTED)
    ax_c.set_title("Energy decay curve and reverberation time (one octave band)", loc="left", fontsize=10,
                   color=INK)
    style(ax_c)
    path2 = OUT / "fig_rir_measures.png"
    fig.savefig(path2, dpi=300)
    plt.close(fig)
    print(f"{room}: direct peak at {t0 / fs * 1e3:.2f} ms, decay ends {(end - t0) / fs:.2f} s, DRR {drr:+.2f} dB "
          f"(paper's window); {FC:.0f} Hz T20 {t20:.3f} s, EDC to {kb / fs:.2f} s\n"
          f"wrote {path1.relative_to(ROOT)}, {path2.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
