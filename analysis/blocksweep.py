#!/usr/bin/env python3
"""RQ1, block length: T20 error against the RLS block length, informed and blind.

    python analysis/blocksweep.py [--paper PATH]

Reads results/blocksweep/blocksweep.csv (pipeline/summariseBlockSweep.m: the
static, head-only and walking takes at 0.5x, 1x, 1.5x and 2x the rule block, the
smallest 500 ms multiple >= 2 x the longest octave-band T20 of the omni reference). With
--paper, also the same sweep run by this project on the reference study's published
combined-movement recordings (blocksweep_paper.csv, produced outside this repository with
that study's own references and ATFs; the study itself published no block-length sweep),
drawn as an extra panel.

Writes results/blocksweep/: summary.md (per condition and block, the mean
T20 error over takes for each band group and each octave band, and the paired
change from 1x) and fig_blocksweep.png (the primary metric, T_mid). The block
length's effect differs by band, so the per-band rows matter: a conclusion on
T_mid holds for T_mid only.

Block length from the glasses alone: the rule needs the omni reference's T20, which the glasses
cannot measure. Sabine's T = 0.161 V / (S alpha) from the SLAM room box (geometry/scene.py), for a
fixed mean absorption coefficient alpha, gives a block length with no acoustic measurement. The
summary lists, per room, the alpha that keep that block inside the range the sweep tolerates, beside
the published values for dwellings (Hoshi et al. 2021, Appl. Sci. 11, 2709: living-dining and child rooms
below 0.17, bedrooms 0.18-0.23, 500 Hz-4 kHz). No alpha is chosen: the point is the range.
With the runs of `run.py sabine` in the CSV, it also scores the pipeline at those blocks: for each alpha interval
inside the range, every take at its room's Sabine block against the same take at the rule block (sabine_runs).
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
import scipy.io as sio  # noqa: E402

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
GEO_ROOMS = ("office", "corridor", "reverb", "meeting", "smallmeeting")
TOLERANCE = {"0.5-2x the rule block (T_mid flat)": 2.0, "0.5-1x (no significant loss in any band)": 1.0}
BEDROOMS, LIVING_MAX = (0.18, 0.23), 0.17       # Hoshi et al. (2021)


def load(path: Path, factor_col: str) -> dict:
    """(take, factor, condition) -> row, with the signed per-band errors in err_<Hz>."""
    out = {}
    for r in csv.DictReader(open(path)):
        out[(r["take"], round(S.num(r[factor_col]), 2), r["condition"])] = r
    return out


def err(D: dict, take: str, f: float, cond: str, bands) -> float:
    r = D.get((take, f, cond))
    return S.t20_err(r, bands, "err_{}") if r else math.nan


def drr_err(D: dict, take: str, f: float, cond: str) -> float:
    """|DRR error| (dB) under the paper's direct-sound window, the one the thesis reports."""
    r = D.get((take, f, cond))
    return abs(S.num(r["drrErrPaper"])) if r else math.nan


def tonal_err(D: dict, take: str, f: float, cond: str) -> float:
    """Tonal-balance error (dB): |octave-band RMS level error|, mean over microphones and bands."""
    r = D.get((take, f, cond))
    return S.num(r["rmsErrMeanDb"]) if r else math.nan


def sweep_table(D: dict, title: str, fn, note: str) -> str:
    """One scalar error across the sweep, per movement and condition: the mean over the takes with every block
    length, and the paired change from 1x (median [95% CI], takes on which the error fell / n, Wilcoxon p)."""
    lines = [f"## {title} across the sweep\n",
             f"{note}: mean over takes; change from 1x (dB, positive = worse): median [95% CI] (takes on which the "
             "error fell / n, Wilcoxon p).\n"]
    for cond, cname in CONDITIONS.items():
        lines.append(f"**{cname}**\n")
        lines.append("| Movement | " + " | ".join(f"{f}x" for f in FACTORS) + " |\n|---|" + "---|" * len(FACTORS))
        for key, name in (("_static", "Static"), ("_headMove", "Head-only"), ("_fullMove", "Walking")):
            Dm = {k: v for k, v in D.items() if key in k[0]}
            takes = complete(Dm, cond)
            cells = []
            for f in FACTORS:
                cell = f"{np.nanmean([fn(Dm, t, f, cond) for t in takes]):.2f}"
                if f != 1.0:
                    r = S.paired([fn(Dm, t, f, cond) - fn(Dm, t, 1.0, cond) for t in takes])
                    cell += (f" ({r['median']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}], {r['down']}/{r['n']}, "
                             f"p={r['p_wilcoxon']:.2f})")
                cells.append(cell)
            lines.append(f"| {name} ({len(takes)} takes) | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def tonal_bias_table(D: dict, cond: str = "blind_motionAware") -> str:
    """Whether block length tilts the spectrum: the paired change from 1x in the SIGNED octave-band RMS level
    error (estimate minus reference, mean over microphones), per band and movement. A |error| that stays flat can
    hide a tilt that moves bands in opposite directions, so the sign is tested here."""
    lines = [f"**{CONDITIONS[cond]}**, signed tonal error per band: change from 1x (dB, positive = the estimate gains "
             "level in that band relative to the reference): median [95% CI], * Wilcoxon p < 0.05\n",
             "| Movement | Factor | " + " | ".join(f"{b} Hz" for b in S.BANDS) + " |\n|---|---|" + "---|" * len(S.BANDS)]
    for key, name in (("_static", "Static"), ("_headMove", "Head-only"), ("_fullMove", "Walking")):
        Dm = {k: v for k, v in D.items() if key in k[0]}
        takes = complete(Dm, cond)
        for f in (0.5, 1.5, 2.0):
            cells = []
            for b in S.BANDS:
                r = S.paired([S.num(Dm[(t, f, cond)][f"rmsErr_{b}"]) - S.num(Dm[(t, 1.0, cond)][f"rmsErr_{b}"])
                              for t in takes])
                cells.append(f"{r['median']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]"
                             + ("*" if r["p_wilcoxon"] < 0.05 else ""))
            lines.append(f"| {name} ({len(takes)}) | {f}x | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def complete(D: dict, cond: str) -> list[str]:
    """Takes that have every block length of the sweep (FACTORS) run for this
    condition, so each curve compares the same takes (a sweep still running has
    some takes part-done). Other block lengths in the CSV are ignored."""
    fs = {k[1] for k in D if k[2] == cond} & set(FACTORS)
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
        lines.append("| Bands | " + " | ".join(f"{f}x" for f in FACTORS) + " |\n|---|" + "---|" * len(FACTORS))
        for sname, bands in [*S.BAND_SETS.items(), *((f"{b} Hz", (b,)) for b in S.BANDS)]:
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


def appendix_tex(D: dict, cond: str = "blind_motionAware") -> str:
    """The appendix table for one condition (blind MA: both tolerances are defined on it): the paired change
    in |T20 error| from 1x at 0.5, 1.5 and 2x, per octave band and movement, with the takes on which the error
    fell and * for Wilcoxon p < 0.05. Returns the LaTeX; also written to appendix_per_band_<cond>.tex."""
    rows = []
    for key, name in (("_static", "Static"), ("_headMove", "Head-only"), ("_fullMove", "Walking")):
        Dm = {k: v for k, v in D.items() if key in k[0]}
        takes = complete(Dm, cond)
        rows.append(rf"\multicolumn{{4}}{{@{{}}l}}{{\emph{{{name}, {len(takes)} takes}}}} \\")
        for b in S.BANDS:
            cells = []
            for f in (0.5, 1.5, 2.0):
                r = S.paired([err(Dm, t, f, cond, (b,)) - err(Dm, t, 1.0, cond, (b,)) for t in takes])
                cells.append(f"${r['median']:+.1f}$ ({r['down']}/{r['n']})" + ("$^{*}$" if r["p_wilcoxon"] < 0.05 else ""))
            rows.append(f"{b if b < 1000 else f'{b // 1000}k'} & " + " & ".join(cells) + r" \\")
        if name != "Walking":
            rows.append(r"\midrule")
    tex = "\n".join([
        r"\begin{table}[H]", r"\centering", r"\small",
        rf"\caption{{{CONDITIONS[cond].replace(' (MA)', ' MA')}: change in $|T_{{20}}$ error$|$ from the rule block "
        r"(1$\times$), per octave band and movement. Median paired change (pp; positive = worse), with the number of "
        r"takes on which the error fell in brackets. $^{*}$ Wilcoxon $p < 0.05$.}",
        rf"\label{{tab:blocksweep-bands}}",
        r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
        r"Band (Hz) & 0.5$\times$ & 1.5$\times$ & 2$\times$ \\", r"\midrule", *rows,
        r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (OUT / f"appendix_per_band_{cond}.tex").write_text(tex)
    return tex


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


def rule_ms(room: str) -> float:
    return float(sio.loadmat(ROOT / "data" / "rooms" / f"{room}_reference.mat",
                             variable_names=["blockLenMsSuggested"], squeeze_me=True)["blockLenMsSuggested"])


def sabine_blocks() -> tuple[np.ndarray, dict]:
    """alpha, and per room the block length / rule block the glasses would choose: Sabine's T from
    the SLAM room box (median V and S over the room's takes), then the rule (the smallest 500 ms
    multiple >= 2 T)."""
    scene = list(csv.DictReader(open(ROOT / "results" / "geometry" / "scene.csv")))
    alpha = np.round(np.arange(0.04, 0.4001, 0.001), 3)
    out = {}
    for room in GEO_ROOMS:
        rs = [r for r in scene if r["take"].startswith(room + "_")]
        vs = np.median([float(r["V_m3"]) for r in rs]) / np.median([float(r["S_m2"]) for r in rs])
        out[room] = np.array([500 * math.ceil(round(2 * 0.161 * vs / a * 1000 / 500, 9)) for a in alpha]) / rule_ms(room)
    return alpha, out


def sabine_runs(D: dict, alpha: np.ndarray, ratios: dict) -> str:
    """The pipeline run at the glasses' own block (run.py sabine): for each alpha interval inside the range
    that keeps all five rooms within 0.5-2x the rule block, every take at its room's Sabine block against
    the same take at the rule block. The paired change is over the takes whose block differs (the rest are
    unchanged by construction); the mean error is over all takes, so it is the cost of the whole choice."""
    ok = np.all([(r >= 0.5 - 1e-9) & (r <= max(TOLERANCE.values()) + 1e-9) for r in ratios.values()], axis=0)
    rules = {room: rule_ms(room) for room in GEO_ROOMS}
    groups = []                                    # (alpha lo, alpha hi, {room: factor}, {room: block ms})
    for a, i in zip(alpha[ok], np.flatnonzero(ok)):
        f = {room: round(float(ratios[room][i]), 2) for room in GEO_ROOMS}   # the CSV's key (load)
        if groups and groups[-1][2] == f and round(a - groups[-1][1], 3) <= 0.001:
            groups[-1][1] = a
        else:
            groups.append([a, a, f, {room: round(float(ratios[room][i]) * rules[room]) for room in GEO_ROOMS}])
    takes = sorted(t for t in {k[0] for k in D} if t.split("_")[0] in GEO_ROOMS)

    def published(lo, hi):
        marks = (["living-dining max"] if lo - 1e-9 <= LIVING_MAX <= hi + 1e-9 else []) + \
                (["bedrooms"] if lo <= BEDROOMS[1] + 1e-9 and hi >= BEDROOMS[0] - 1e-9 else [])
        return f" ({', '.join(marks)})" if marks else ""

    def cell(r, ci=False, dp=1):
        if not r:
            return "--"
        return (f"{r['median']:+.{dp}f}" + (f" [{r['lo']:+.{dp}f}, {r['hi']:+.{dp}f}]" if ci else "")
                + f" ({r['down']}/{r['n']}, p={r['p_wilcoxon']:.2f})")

    lines = ["## The pipeline at the glasses' own block\n",
             f"`run.py sabine`: every take at its room's Sabine block (above) for each alpha that keeps all five rooms "
             f"within 0.5-2x the rule block ({alpha[ok].min():.3f}-{alpha[ok].max():.3f}), against the same take at "
             f"the rule block. Blocks in s (office / corridor / reverb / meeting / smallmeeting; rule "
             + " / ".join(f"{rules[r] / 1000:g}" for r in GEO_ROOMS) + "). Mean |T20 error| over all "
             f"{len(takes)} takes at the Sabine and the rule blocks; change (pp, positive = worse) on the takes whose "
             "block differs: median ([95% CI, all movements only] takes on which the error fell / takes changed, Wilcoxon p). |DRR error| under the paper's direct-sound window, in dB.\n"]
    metrics = (("T_mid", lambda t, f, c: err(D, t, f, c, S.BAND_SETS["T_mid"]), "{:.1f} %", 1),
               ("|DRR error| (dB, paper's window)", lambda t, f, c: drr_err(D, t, f, c), "{:.2f} dB", 2),
               ("tonal-balance error (dB)", lambda t, f, c: tonal_err(D, t, f, c), "{:.2f} dB", 2))
    for mname, fn, fmt, dp in metrics:
        for cond in ("blind_motionAware", "blind_omni"):
            missing = [(t, f) for t in takes for g in groups for f in [g[2][t.split("_")[0]]] if (t, f, cond) not in D]
            lines.append(f"**{CONDITIONS[cond]}**, {mname}"
                         + (f" ({len(missing)} take-blocks missing)" if missing else "") + "\n")
            lines.append("| alpha | Blocks (s) | Mean at Sabine / rule | Change | Static | Head-only | Walking |\n"
                         "|---|---|---|---|---|---|---|")
            for lo, hi, f, ms in groups:
                fac = {t: f[t.split("_")[0]] for t in takes}
                sab = np.nanmean([fn(t, fac[t], cond) for t in takes])
                rule = np.nanmean([fn(t, 1.0, cond) for t in takes])
                row = [f"{lo:.3f}-{hi:.3f}{published(lo, hi)}",
                       " / ".join(f"{ms[r] / 1000:g}" for r in GEO_ROOMS),
                       fmt.format(sab).split()[0] + " / " + fmt.format(rule)]
                for key in ("", "_static", "_headMove", "_fullMove"):
                    ch = [t for t in takes if key in t and fac[t] != 1.0]
                    row.append(cell(S.paired([fn(t, fac[t], cond) - fn(t, 1.0, cond) for t in ch]), ci=not key, dp=dp))
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
    cond = "blind_motionAware"
    lines.append(f"**{CONDITIONS[cond]}**, per octave band: change on the takes whose block differs, all movements "
                 "(median pp, takes on which the error fell / takes changed; * Wilcoxon p < 0.05)\n")
    lines.append("| alpha | " + " | ".join(f"{b} Hz" for b in S.BANDS) + " |\n|---|" + "---|" * len(S.BANDS))
    for lo, hi, f, _ in groups:
        ch = [t for t in takes if f[t.split("_")[0]] != 1.0]
        cells = []
        for b in S.BANDS:
            r = S.paired([err(D, t, f[t.split("_")[0]], cond, (b,)) - err(D, t, 1.0, cond, (b,)) for t in ch])
            cells.append(f"{r['median']:+.1f} ({r['down']}/{r['n']}){'*' if r['p_wilcoxon'] < 0.05 else ''}"
                         if r else "--")
        lines.append(f"| {lo:.3f}-{hi:.3f}{published(lo, hi)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def admissible(alpha: np.ndarray, r: np.ndarray, hi: float) -> str:
    ok = (r >= 0.5 - 1e-9) & (r <= hi + 1e-9)
    if not ok.any():
        return "none"
    top = f"{alpha[-1]:.2f}+" if ok[-1] else f"{alpha[ok].max():.3f}"      # + : still inside at the grid's end
    return f"{alpha[ok].min():.3f}-{top}"


def md_geometry(alpha: np.ndarray, ratios: dict) -> str:
    lines = ["## Block length from the glasses' geometry\n",
             "Sabine's T = 0.161 V / (S alpha) from the SLAM room box, then the rule: no acoustic measurement. "
             "Cells: the range of mean absorption coefficient alpha that keeps the block inside each tolerance of "
             "the sweep above (blind motion-aware). '+': still inside at alpha = 0.40, the end of the range "
             f"searched. Published values for dwellings (Hoshi et al. 2021): living-dining and child rooms below "
             f"{LIVING_MAX}, bedrooms {BEDROOMS[0]}-{BEDROOMS[1]}.\n",
             "| Room | " + " | ".join(TOLERANCE) + " |", "|---|" + "---|" * len(TOLERANCE)]
    for room in GEO_ROOMS:
        lines.append(f"| {room}{' (held out)' if room == 'smallmeeting' else ''} | "
                     + " | ".join(admissible(alpha, ratios[room], hi) for hi in TOLERANCE.values()) + " |")
    cells = []
    for hi in TOLERANCE.values():
        both = np.all([(r >= 0.5 - 1e-9) & (r <= hi + 1e-9) for r in ratios.values()], axis=0)
        cells.append(f"**{alpha[both].min():.3f}-{alpha[both].max():.3f}**" if both.any() else "none")
    lines.append("| **all five rooms** | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", type=Path, help="the reference study's blocksweep_paper.csv")
    a = ap.parse_args()
    D = load(ARIA, "blockFactor")
    md = ["# RQ1: block length\n", "Generated by `analysis/blocksweep.py`. Rule block = the smallest 500 ms multiple "
          ">= 2 x the longest octave-band T20 of the reference.\n"]
    panels = []
    for key, name in (("_fullMove", "walking"), ("_headMove", "head-only"), ("_static", "static")):
        Dm = {k: v for k, v in D.items() if key in k[0]}
        if not Dm:
            continue
        md_m, cur_m = table(Dm, f"Aria Gen 1, {name} takes")
        md.append(md_m)
        panels.append((f"Aria Gen 1, {name} ({len(complete(Dm, 'informed_motionAware'))} takes)", cur_m))
    if a.paper:
        Dp = load(a.paper, "factor")
        md_p, cur_p = table(Dp, "Reference study's recordings, combined movement")
        md.append(md_p)
        panels.append((f"Reference study's recordings ({len(complete(Dp, 'informed_motionAware'))} takes)", cur_p))
    figure(panels, OUT / "fig_blocksweep.png")
    appendix_tex(D)
    md.append(sweep_table(D, "DRR", drr_err, "|DRR error| (dB) under the paper's direct-sound window"))
    md.append(sweep_table(D, "Tonal balance", tonal_err, "Tonal-balance error (dB): |octave-band RMS level error|, "
                          "mean over microphones and bands"))
    md.append(tonal_bias_table(D))
    alpha, ratios = sabine_blocks()
    md.append(md_geometry(alpha, ratios))
    md.append(sabine_runs(D, alpha, ratios))
    txt = "\n".join(md)
    (OUT / "summary.md").write_text(txt)
    print(txt)
    print(f"wrote {OUT.relative_to(ROOT)}/: summary.md, fig_blocksweep.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
