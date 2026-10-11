#!/usr/bin/env python3
"""RQ1 reproduction summary: the paper's Table 2, Fig. 7 and Fig. 8 on this data.

    python analysis/reproduction.py

Reads results/pipeline/summary.csv (written by pipeline/rescore.m) and writes
results/reproduction/:

  table2_drr.csv       mean and sd of |DRR error| per method and movement,
                       under the paper's direct-sound window and the repo's,
                       beside the paper's Table 2, plus the held-out room
  t20_per_band.csv     Fig. 7 data: mean and sd of |relative T20 error| per
                       octave band, plus the signed mean (the bias)
  rms_per_band.csv     Fig. 8 data: mean and sd of the tonal-balance (octave-
                       band RMS) error AT THE MICROPHONES (scoreRir.m: each
                       estimate decoded to the 7 microphones, spectral shape
                       of the reverberant part against the measured one, |error|
                       averaged over the microphones), plus the signed mean
  per_room.csv         mean |T20 error| and |DRR error| per room, held-out
                       room last
  fig7_t20.png         Fig. 7 layout: head-only movement (top), walking (bottom)
  fig8_rms.png         Fig. 8 layout, same panels, microphone-domain error
  fig_drr.png          mean |DRR error| per method (paper window) beside the
                       paper's Table 2
  summary.md           all of the above on one page, with the takes used

Takes: the four composite-protocol rooms only. The L-shape takes are left out:
they are acoustically the corridor and exist for the SLAM/OptiTrack pose
comparison (README, "Rooms"), so including them would count the corridor
twice. Movement is read from the take name; the paper's "combined movement" is
this dataset's walking (fullMove). Static takes have no counterpart in the
paper and appear in the tables only.

The held-out room (small meeting room, rooms.json "held_out") was recorded
after everything else was settled. It is kept out of the means compared with
the paper and the figures, and reported on its own: a column of Table 2 and a
row of the per-room table.

The paper gives Figs. 7 and 8 as plots without tabulated values, so only
Table 2 is compared number for number; the figures reproduce its layout for a
side-by-side reading. Fig. 8 deliberately departs from the paper's definition
(the omni CH channel): on this 7-mic array, under movement, the omni channel of
the RLS estimate and of the CH-encoded reference disagree where the microphone
signals agree, which made the informed estimate look worse than the blind one.
The paper-style value is still in summary.csv as rmsOmniErrMeanDb. The paper's
single-channel baseline was not run here.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "pipeline" / "summary.csv"
OUT = ROOT / "results" / "reproduction"

ROOMS = ("office", "corridor", "reverb", "meeting")
HELD_OUT = ("smallmeeting",)
BANDS = (125, 250, 500, 1000, 2000, 4000, 8000)
MOVEMENTS = {"headMove": "Head-only movement", "fullMove": "Walking (paper: combined movement)",
             "static": "Static (not in the paper)"}
METHODS = {"informed_motionAware": "Informed (MA)", "blind_motionAware": "Blind (MA)",
           "blind_omni": "Blind (Omni)", "informed_omni": "Informed (Omni)"}
PAPER_METHODS = ("informed_motionAware", "blind_motionAware", "blind_omni")   # the paper's three
# Deppisch et al. (2026), Table 2: mean and sd of |DRR error| in dB
PAPER_TABLE2 = {
    ("headMove", "informed_motionAware"): (1.0, 0.9), ("headMove", "blind_motionAware"): (1.6, 1.3),
    ("headMove", "blind_omni"): (4.9, 2.1), ("headMove", "baseline"): (24.9, 2.3),
    ("fullMove", "informed_motionAware"): (1.6, 1.3), ("fullMove", "blind_motionAware"): (3.3, 0.8),
    ("fullMove", "blind_omni"): (6.0, 2.5), ("fullMove", "baseline"): (24.8, 1.4),
}
# categorical slots 1-3 of the dataviz reference palette (light mode), fixed order
COLOURS = {"informed_motionAware": "#2a78d6", "blind_motionAware": "#eb6834", "blind_omni": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PAPER_BAR, OURS_BAR = "#cfcdc6", "#52514e"      # neutrals: in the DRR figure colour marks the source, not the method


def movement(take: str) -> str | None:
    for m in MOVEMENTS:
        if f"_{m}" in take:
            return m
    return None


def load(rooms: tuple[str, ...] = ROOMS) -> dict:
    """take -> condition -> row, for the given rooms."""
    if not SUMMARY.exists():
        raise SystemExit(f"{SUMMARY} not found: run  matlab -batch \"cd pipeline; rescore()\"  first")
    rows = list(csv.DictReader(open(SUMMARY)))
    need = ["drrErrPaper", "rmsErrMeanDb", "t20Err_125", "rmsErr_125", "rmsAbsErr_125"]
    missing = [c for c in need if c not in rows[0]]
    if missing:
        raise SystemExit(f"summary.csv lacks {missing}: re-run rescore() with the current scoreRir")
    data = defaultdict(dict)
    for r in rows:
        if r["take"].split("_")[0] in rooms and movement(r["take"]):
            data[r["take"]][r["condition"]] = r
    return data


def takes_for(data: dict, mov: str) -> list[str]:
    return sorted(t for t in data if movement(t) == mov)


def values(data: dict, takes: list[str], cond: str, col: str, absolute: bool = True) -> np.ndarray:
    v = np.array([float(data[t][cond][col]) for t in takes if cond in data[t]], float)
    return np.abs(v) if absolute else v


def mean_sd(v: np.ndarray) -> tuple[float, float]:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else float("nan")


def fmt(m: float, s: float, unit: str = "") -> str:
    if not np.isfinite(m):
        return "--"
    return f"{m:.1f} ± {s:.1f}{unit}" if np.isfinite(s) else f"{m:.1f}{unit}"


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow([f"{x:.4f}" if isinstance(x, float) else x for x in r])


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def table2(data: dict, held: dict) -> tuple[list, str]:
    rows, md = [], []
    for mov in ("headMove", "fullMove", "static"):
        T, H = takes_for(data, mov), takes_for(held, mov)
        if not T:
            continue
        for cond in (*PAPER_METHODS, "informed_omni", "baseline"):
            p = PAPER_TABLE2.get((mov, cond), (float("nan"), float("nan")))
            if cond == "baseline":
                if not np.isfinite(p[0]):
                    continue
                rows.append([mov, cond, 0, p[0], p[1], "", "", "", "", 0, "", ""])
                md.append([MOVEMENTS[mov].split(" (")[0], "Baseline (single mic)", fmt(*p, " dB"), "not run", "not run",
                           "0", "not run", "0"])
                continue
            pw = mean_sd(values(data, T, cond, "drrErrPaper"))
            rw = mean_sd(values(data, T, cond, "drrErr"))
            n = int(np.isfinite(values(data, T, cond, "drrErrPaper")).sum())
            hw = mean_sd(values(held, H, cond, "drrErrPaper"))
            nh = int(np.isfinite(values(held, H, cond, "drrErrPaper")).sum())
            rows.append([mov, cond, n, p[0], p[1], pw[0], pw[1], rw[0], rw[1], nh, hw[0], hw[1]])
            md.append([MOVEMENTS[mov].split(" (")[0], METHODS[cond], fmt(*p, " dB"), fmt(*pw, " dB"), fmt(*rw, " dB"),
                       str(n), fmt(*hw, " dB"), str(nh)])
    write_csv(OUT / "table2_drr.csv", ["movement", "condition", "n_takes", "paper_mean_db", "paper_sd_db",
                                      "ours_paper_window_mean_db", "ours_paper_window_sd_db",
                                      "ours_repo_window_mean_db", "ours_repo_window_sd_db",
                                      "held_out_n_takes", "held_out_paper_window_mean_db",
                                      "held_out_paper_window_sd_db"], rows)
    return rows, md_table(["Movement", "Method", "Paper (Table 2)", "Ours, paper window", "Ours, repo window", "Takes",
                           "Held-out room, paper window", "Takes"], md)


def per_band(data: dict, prefix: str, fname: str, unit: str, abs_prefix: str | None = None) -> tuple[dict, str]:
    """mean/sd of |error| per band, per movement and condition; returns {(mov, cond): (mean[], sd[])}.

    |error| is taken from abs_prefix columns when given (already absolute per
    take, e.g. averaged over microphones), else as |prefix column|; the signed
    mean always comes from the prefix columns.
    """
    abs_prefix = abs_prefix or prefix
    stats, rows, md = {}, [], []
    for mov in ("headMove", "fullMove", "static"):
        T = takes_for(data, mov)
        if not T:
            continue
        for cond in METHODS:
            ms = [mean_sd(values(data, T, cond, f"{abs_prefix}{b}")) for b in BANDS]
            bias = [float(np.nanmean(values(data, T, cond, f"{prefix}{b}", absolute=False))) for b in BANDS]
            stats[(mov, cond)] = (np.array([m for m, _ in ms]), np.array([s for _, s in ms]))
            for b, (m, s), sg in zip(BANDS, ms, bias):
                rows.append([mov, cond, b, len(T), m, s, sg])
            md.append([MOVEMENTS[mov].split(" (")[0], METHODS[cond]] + [f"{m:.1f}" for m, _ in ms])
    write_csv(OUT / fname, ["movement", "condition", "band_hz", "n_takes", f"abs_mean_{unit}", f"abs_sd_{unit}",
                            f"signed_mean_{unit}"], rows)
    return stats, md_table(["Movement", "Method"] + [f"{b//1000}k" if b >= 1000 else str(b) for b in BANDS], md)


def per_room(data: dict) -> str:
    rows, md = [], []
    for mov in ("headMove", "fullMove", "static"):
        for room in (*ROOMS, *HELD_OUT):
            T = [t for t in takes_for(data, mov) if t.split("_")[0] == room]
            if not T:
                continue
            cells = []
            for cond in PAPER_METHODS:
                t20 = mean_sd(values(data, T, cond, "rtErrMean"))[0]
                drr = mean_sd(values(data, T, cond, "drrErrPaper"))[0]
                rows.append([mov, room, cond, len(T), t20, drr])
                cells.append(f"{t20:.1f} % / {drr:.1f} dB")
            label = f"{room} (held out)" if room in HELD_OUT else room
            md.append([MOVEMENTS[mov].split(" (")[0], label, str(len(T))] + cells)
    write_csv(OUT / "per_room.csv", ["movement", "room", "condition", "n_takes", "t20_abs_mean_pct",
                                     "drr_abs_mean_paper_window_db"], rows)
    return md_table(["Movement", "Room", "Takes"] + [f"{METHODS[c]}: T20 / DRR" for c in PAPER_METHODS], md)


def figure(stats: dict, data: dict, ylabel: str, title: str, path: Path) -> None:
    movs = [m for m in ("headMove", "fullMove") if takes_for(data, m)]
    fig, axes = plt.subplots(len(movs), 1, figsize=(6.6, 2.7*len(movs)), sharex=True, sharey=True,
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(BANDS))
    off = dict(zip(PAPER_METHODS, (-0.18, 0.0, 0.18)))
    for ax, mov in zip(axes, movs):
        for cond in PAPER_METHODS:
            m, s = stats[(mov, cond)]
            ax.errorbar(x + off[cond], m, yerr=s, fmt="o-", ms=5, lw=1.5, elinewidth=1.0, capsize=0,
                        color=COLOURS[cond], label=METHODS[cond])
        ax.set_title(f"{MOVEMENTS[mov].split(' (')[0]} ({len(takes_for(data, mov))} takes)", loc="left",
                     fontsize=10, color=INK)
        ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8.5)
    axes[0].legend(frameon=False, fontsize=8.5, ncol=3, loc="upper left", labelcolor=INK)
    axes[-1].set_xticks(x, [f"{b//1000}k" if b >= 1000 else str(b) for b in BANDS])
    axes[-1].set_xlabel("Octave band (Hz)", fontsize=9, color=MUTED)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def figure_drr(data: dict, path: Path) -> None:
    """Mean |DRR error| (paper window) beside the paper's Table 2, one pair of
    bars per method. Only the comparison the paper reports: the spread is in
    table2_drr.csv (sd) and per_room.csv. Colour marks the source (paper or
    this data); the method is the x-axis, so it needs no colour here.
    """
    movs = [m for m in ("headMove", "fullMove") if takes_for(data, m)]
    fig, axes = plt.subplots(1, len(movs), figsize=(3.3*len(movs) + 0.5, 3.4), sharey=True,
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(PAPER_METHODS))
    w = 0.36
    top = 0.0
    for ax, mov in zip(axes, movs):
        T = takes_for(data, mov)
        paper = [PAPER_TABLE2[(mov, c)][0] for c in PAPER_METHODS]
        ours = [mean_sd(values(data, T, c, "drrErrPaper"))[0] for c in PAPER_METHODS]
        for xs, v, col, lab in ((x - w/2 - 0.015, paper, PAPER_BAR, "Paper (Table 2)"),
                                (x + w/2 + 0.015, ours, OURS_BAR, "This data")):
            ax.bar(xs, v, width=w, color=col, label=lab, zorder=2)
            for xi, vi in zip(xs, v):
                ax.text(xi, vi + 0.1, f"{vi:.1f}", ha="center", va="bottom", fontsize=8.5, color=INK)
        top = max(top, *paper, *ours)
        ax.set_xticks(x, [METHODS[c] for c in PAPER_METHODS])
        ax.set_title(f"{MOVEMENTS[mov].split(' (')[0]} ({len(T)} takes)", loc="left", fontsize=10, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8.5)
    axes[0].set_ylim(0, np.ceil(top + 1))
    axes[0].set_ylabel("Mean |DRR error| (dB)", fontsize=9, color=MUTED)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK)
    fig.suptitle("DRR error, paper's direct-sound window (cf. paper Table 2)", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data, held = load(), load(HELD_OUT)
    _, t2 = table2(data, held)
    t20, t20_md = per_band(data, "t20Err_", "t20_per_band.csv", "pct")
    rms, rms_md = per_band(data, "rmsErr_", "rms_per_band.csv", "db", abs_prefix="rmsAbsErr_")
    room_md = per_room({**data, **held})
    figure(t20, data, "|T20 error| (%)", "Relative octave-band T20 error, mean ± sd (cf. paper Fig. 7)",
           OUT / "fig7_t20.png")
    figure(rms, data, "|tonal-balance error| (dB)",
           "Tonal-balance error at the microphones, mean ± sd (cf. paper Fig. 8)", OUT / "fig8_rms.png")
    figure_drr(data, OUT / "fig_drr.png")
    used = "\n".join(f"- {MOVEMENTS[m].split(' (')[0]}: " + ", ".join(takes_for(data, m))
                     for m in MOVEMENTS if takes_for(data, m))
    used += "\n\nHeld out:\n\n" + "\n".join(f"- {MOVEMENTS[m].split(' (')[0]}: " + ", ".join(takes_for(held, m))
                                         for m in MOVEMENTS if takes_for(held, m))
    md = f"""# RQ1: reproduction of Deppisch et al. (2026) on Aria Gen 1

Generated by `analysis/reproduction.py` from `results/pipeline/summary.csv`.
Rooms: {", ".join(ROOMS)}. Mean ± sd over takes; sd is the sample sd. The held-out room
({", ".join(HELD_OUT)}) is reported separately and is not in the means or the figures.

## Table 2: |DRR error|

The paper's window (direct sound = peak -0.5 to +1 ms) is the like-for-like comparison.

{t2}

## Fig. 7: mean |relative T20 error| per octave band (%)

{t20_md}

## Fig. 8: mean |tonal-balance error| at the microphones (dB)

The octave-band RMS error, scored at the microphones rather than on the omni channel as in the paper:
each estimate is decoded to the 7 microphones and the spectral shape of its reverberant part (band
levels, mean over bands removed, since blind estimates have an arbitrary gain) is compared with the
measured microphone's; |error| is averaged over the microphones.

{rms_md}

## Per room: mean |T20 error| / mean |DRR error| (paper window)

{room_md}

## Takes used

{used}
"""
    (OUT / "summary.md").write_text(md)
    print(md)
    print(f"wrote {OUT.relative_to(ROOT)}/: summary.md, table2_drr.csv, t20_per_band.csv, rms_per_band.csv, "
          f"per_room.csv, fig7_t20.png, fig8_rms.png, fig_drr.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
