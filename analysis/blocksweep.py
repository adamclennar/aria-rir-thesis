#!/usr/bin/env python3
"""RQ2, block length: T20 error against the RLS block length, informed and blind.

    python analysis/blocksweep.py [--paper PATH]

Reads results/blocksweep/blocksweep.csv (pipeline/summariseBlockSweep.m: the
walking and static takes at 0.5x, 1x, 1.5x and 2x the rule block, the smallest 500 ms
multiple >= 2 x the longest octave-band T20 of the omni reference). With
--paper, also the same sweep on the reference study's combined-movement takes
(blocksweep_paper.csv, run outside this repository with the study's own
references and ATFs), drawn as a second panel.

Writes results/blocksweep/: summary.md (per condition and block, the mean of
each band set's T20 error over takes and the paired change from 1x) and
fig_blocksweep.png (the primary metric, T_mid).
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import stats as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARIA = ROOT / "results" / "blocksweep" / "blocksweep.csv"
OUT = ROOT / "results" / "blocksweep"
FACTORS = (0.5, 1.0, 1.5, 2.0)
CONDITIONS = {"informed_motionAware": "Informed (MA)", "blind_motionAware": "Blind (MA)",
              "informed_omni": "Informed (Omni)", "blind_omni": "Blind (Omni)"}
STYLE = {"informed_motionAware": ("#2a78d6", "-"), "blind_motionAware": ("#eb6834", "-"),
         "informed_omni": ("#2a78d6", ":"), "blind_omni": ("#eb6834", ":")}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(path: Path, factor_col: str) -> dict:
    """(take, factor, condition) -> row, with the signed per-band errors in err_<Hz>."""
    out = {}
    for r in csv.DictReader(open(path)):
        out[(r["take"], round(S.num(r[factor_col]), 2), r["condition"])] = r
    return out


def err(D: dict, take: str, f: float, cond: str, bands) -> float:
    r = D.get((take, f, cond))
    return S.t20_err(r, bands, "err_{}") if r else math.nan


def complete(D: dict, cond: str) -> list[str]:
    """Takes that have every block length run for this condition, so each curve
    compares the same takes (a sweep still running has some takes part-done)."""
    fs = {k[1] for k in D if k[2] == cond}
    return sorted(t for t in {k[0] for k in D} if all((t, f, cond) in D for f in fs))


def table(D: dict, name: str) -> tuple[str, dict]:
    """Markdown for one dataset, and {cond: {factor: mean primary error}} for the figure."""
    curves, lines = {}, [f"## {name}: {len({k[0] for k in D})} takes\n"]
    for cond, cname in CONDITIONS.items():
        takes = complete(D, cond)
        if not takes:
            continue
        lines.append(f"**{cname}**, {len(takes)} takes with every block length (mean over takes; change from "
                     f"1x: median, pairs down / n, Wilcoxon p)\n")
        lines.append("| Band set | " + " | ".join(f"{f}x" for f in FACTORS) + " |\n|---|" + "---|" * len(FACTORS))
        for sname, bands in S.BAND_SETS.items():
            cells = []
            for f in FACTORS:
                v = np.array([err(D, t, f, cond, bands) for t in takes])
                if not np.isfinite(v).any():
                    cells.append("--")
                    continue
                cell = f"{np.nanmean(v):.1f} %"
                if f != 1.0:
                    r = S.paired([err(D, t, f, cond, bands) - err(D, t, 1.0, cond, bands) for t in takes])
                    if r:
                        cell += f" ({r['median']:+.1f}, {r['down']}/{r['n']}, p={r['p_wilcoxon']:.2f})"
                cells.append(cell)
                if sname == S.PRIMARY:
                    curves.setdefault(cond, {})[f] = float(np.nanmean(v))
            lines.append(f"| {sname} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines), curves


def figure(panels: list[tuple[str, dict]], path: Path) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6*len(panels) + 0.6, 3.3), sharey=True,
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (title, curves) in zip(axes, panels):
        for cond, pts in curves.items():
            fs = sorted(pts)
            col, ls = STYLE[cond]
            ax.plot(fs, [pts[f] for f in fs], ls, marker="o", ms=4.5, lw=1.6, color=col, label=CONDITIONS[cond])
        ax.set_xticks(FACTORS, [f"{f:g}x" for f in FACTORS])
        ax.set_xlabel("RLS block length / rule block", fontsize=9, color=MUTED)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8.5)
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel("Mean |T20 error|, 500 Hz + 1 kHz (%)", fontsize=9, color=MUTED)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", type=Path, help="the reference study's blocksweep_paper.csv")
    a = ap.parse_args()
    D = load(ARIA, "blockFactor")
    md = ["# RQ2: block length\n", "Generated by `analysis/blocksweep.py`. Rule block = the smallest 500 ms multiple "
          ">= 2 x the longest octave-band T20 of the reference.\n"]
    panels = []
    for key, name in (("_fullMove", "walking"), ("_static", "static")):
        Dm = {k: v for k, v in D.items() if key in k[0]}
        if not Dm:
            continue
        md_m, cur_m = table(Dm, f"Aria Gen 1, {name} takes")
        md.append(md_m)
        panels.append((f"Aria Gen 1, {name} ({len(complete(Dm, 'informed_motionAware'))} takes)", cur_m))
    if a.paper:
        Dp = load(a.paper, "factor")
        md_p, cur_p = table(Dp, "Reference study, combined movement")
        md.append(md_p)
        panels.append((f"Reference study ({len(complete(Dp, 'informed_motionAware'))} takes)", cur_p))
    figure(panels, OUT / "fig_blocksweep.png")
    txt = "\n".join(md)
    (OUT / "summary.md").write_text(txt)
    print(txt)
    print(f"wrote {OUT.relative_to(ROOT)}/: summary.md, fig_blocksweep.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
