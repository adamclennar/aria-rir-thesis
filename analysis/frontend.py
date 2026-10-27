#!/usr/bin/env python3
"""RQ2, front end: how far the blind front end is from a perfect one, and which stage the gap is in.

    python analysis/frontend.py

Reads results/frontend/<take>_anchors.csv (pipeline/frontEndAnchors.m, via
run.py anchors) and writes results/frontend/:

  tests.csv        every comparison below, every metric, both RLS models
  fig_ladder.png   primary T20 error per condition and movement (motion-aware)
  summary.md       the primary metric's comparisons, then the sensitivity checks

Comparisons, each paired by take (stats.paired), A minus B, so negative = A better:
  headroom         wpe_oracle_steering - perfect_frontend
                   WPE with the steering held at the protocol bearing vs a
                   perfect front end: what better dereverberation could gain
  published        published - perfect_frontend
                   the published front end (MUSIC steering) vs a perfect one
  dereverberation  wpe_oracle_steering - nowpe_oracle_steering
                   what WPE does, steering identical
  steering         published - wpe_oracle_steering
                   MUSIC vs steering at the protocol bearing
  movement         perfect_frontend - informed_lvl_matched
                   what the array's motion costs a perfect front end
  level            informed - informed_lvl_matched
                   the RLS regularisation effect of the reference level

The static and head-only takes are also pooled for the headroom comparison.

Deviation from informed. The T20 error against the omni reference mixes two
things: what the pseudo-reference does to the estimate, and the estimator's own
floor (the informed and perfect-front-end estimates of a take share a bias of
several percent against the omni). Where the pseudo-reference's shift has the
opposite sign to that floor, the two cancel and a worse reference scores a
smaller |error| (the reverb room). So the front end is also measured as each
estimate's deviation from the informed estimate of the same take at the same
level (informed_lvl_matched), in the T_mid bands: the mean |difference| of the two
relative errors, and its signed mean (positive = the estimate's T20 is longer).
Rooms: all five composite-protocol rooms pooled (the small meeting room is held
out only where something is fitted or chosen), with a per-room table.
"""
from __future__ import annotations

import csv
import glob
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import stats as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "results" / "frontend"
ROOM_SETS = {"all": ("office", "corridor", "reverb", "meeting", "smallmeeting")}
MOVEMENTS = {"static": "Static", "headMove": "Head-only", "fullMove": "Walking"}
COMPARISONS = [
    ("headroom", "wpe_oracle_steering", "perfect_frontend"),
    ("published", "published", "perfect_frontend"),
    ("dereverberation", "wpe_oracle_steering", "nowpe_oracle_steering"),
    ("steering", "published", "wpe_oracle_steering"),
    ("movement", "perfect_frontend", "informed_lvl_matched"),
    ("level", "informed", "informed_lvl_matched"),
]
LADDER = {"nowpe_oracle_steering": "No WPE\n(oracle\nsteering)", "published": "Published\n(WPE, MUSIC\nsteering)",
          "wpe_oracle_steering": "Fixed\nsteering\n(WPE, oracle)", "perfect_frontend": "Perfect\nfront end",
          "informed_lvl_matched": "Informed,\nmatched\nlevel"}
COLOURS = {"static": "#2a78d6", "headMove": "#eb6834", "fullMove": "#1baf7a"}
# the figure's two groups: static and head-only follow the same trends; colours carried over from the
# three-movement version (static's blue, walking's aqua)
GROUPS = {"Not walking": ("static", "headMove"), "Walking": ("fullMove",)}
GROUP_COLOURS = {"Not walking": "#2a78d6", "Walking": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
METRICS = {f"t20_{k}": ((lambda r, b=b: S.t20_err(r, b, "t20Err_{}")), "pp") for k, b in S.BAND_SETS.items()}
METRICS["drr_repo"] = (lambda r: abs(S.num(r["drrErr"])), "dB")
METRICS["drr_paper"] = (lambda r: abs(S.num(r["drrErrPaper"])), "dB")
METRICS["tonal"] = (lambda r: S.num(r["rmsErrMeanDb"]), "dB")
PRIMARY = f"t20_{S.PRIMARY}"


def movement(take: str) -> str:
    return next(m for m in MOVEMENTS if f"_{m}" in take)


def load() -> dict:
    """room set -> take -> (label, model) -> row."""
    data = {k: defaultdict(dict) for k in ROOM_SETS}
    for f in sorted(glob.glob(str(DIR / "*_anchors.csv"))):
        for r in csv.DictReader(open(f)):
            room = r["take"].split("_")[0]
            for k, rooms in ROOM_SETS.items():
                if room in rooms:
                    data[k][r["take"]][(r["label"], r["model"])] = r
    return data


def value(data: dict, take: str, label: str, model: str, met: str) -> float:
    r = data[take].get((label, model))
    return METRICS[met][0](r) if r else math.nan


def tests(data: dict, set_name: str) -> list[dict]:
    out = []
    for kind, a, b in COMPARISONS:
        for model in ("motionAware", "omni"):
            for mov in MOVEMENTS:
                T = sorted(t for t in data if movement(t) == mov)
                for met in METRICS:
                    r = S.paired([value(data, t, a, model, met) - value(data, t, b, model, met) for t in T])
                    if r:
                        out.append(dict(rooms=set_name, kind=kind, a=a, b=b, model=model, movement=mov,
                                        metric=met, **r))
    return out


def pooled_headroom(data: dict) -> str:
    """wpe_oracle_steering vs perfect_frontend on the static and head-only takes pooled (T_mid, motion-aware)."""
    T = sorted(t for t in data if movement(t) in ("static", "headMove"))
    r = S.paired([value(data, t, "perfect_frontend", "motionAware", PRIMARY)
                  - value(data, t, "wpe_oracle_steering", "motionAware", PRIMARY) for t in T])
    return "--" if r is None else f"perfect_frontend - wpe_oracle_steering: {S.fmt_paired(r, 'pp')}"


def deviation(data: dict, take: str, label: str, model: str, signed: bool = False, bands=None) -> float:
    """Mean over the T_mid bands (or `bands`) of (relative error of label) - (relative error of
    informed_lvl_matched), same take and model: absolute by default, signed if asked."""
    r, ri = data[take].get((label, model)), data[take].get(("informed_lvl_matched", model))
    if not r or not ri:
        return math.nan
    d = [S.num(r[f"t20Err_{b}"]) - S.num(ri[f"t20Err_{b}"]) for b in (bands or S.BAND_SETS[S.PRIMARY])]
    return float(np.mean(d)) if signed else float(np.mean(np.abs(d)))


def md_deviation(data: dict) -> str:
    """Mean |deviation from informed| (signed mean in brackets) per condition and movement, each model;
    then the headroom on this measure, paired, for takes where the wearer does not walk and for walking."""
    labs = [l for l in LADDER if l != "informed_lvl_matched"]
    out = []
    for model, mname in (("motionAware", "Motion-aware"), ("omni", "Omni")):
        out += [f"\n*{mname}*\n", "| Movement | Takes | " + " | ".join(labs) + " |", "|---|---|" + "---|" * len(labs)]
        for mov, name in MOVEMENTS.items():
            T = [t for t in data if movement(t) == mov]
            cells = [f"{np.nanmean([deviation(data, t, l, model) for t in T]):.1f} "
                     f"({np.nanmean([deviation(data, t, l, model, True) for t in T]):+.1f})" for l in labs]
            out.append(f"| {name} | {len(T)} | " + " | ".join(cells) + " |")
        for what, movs in (("not walking", ("static", "headMove")), ("walking", ("fullMove",))):
            T = [t for t in data if movement(t) in movs]
            r = S.paired([deviation(data, t, "perfect_frontend", model) - deviation(data, t, "wpe_oracle_steering", model) for t in T])
            out.append(f"\n- Headroom on this measure, {what}: perfect_frontend - wpe_oracle_steering {S.fmt_paired(r, 'pp')}")
    return "\n".join(out)


def md_per_room(data: dict) -> str:
    """Per room, mean primary error (motion-aware) of the published front end, WPE with oracle
    steering, and the perfect front end, for the takes where the wearer does not walk and for walking."""
    lines = ["| Room | Not walking: takes | published | WPE, oracle steering | perfect | Walking: takes | published | "
             "WPE, oracle steering | perfect |", "|---|---|---|---|---|---|---|---|---|"]
    for room in ROOM_SETS["all"]:
        cells = []
        for movs in (("static", "headMove"), ("fullMove",)):
            T = [t for t in data if t.split("_")[0] == room and movement(t) in movs]
            m = lambda lab: np.mean([value(data, t, lab, "motionAware", PRIMARY) for t in T]) if T else np.nan
            cells += [str(len(T))] + [("--" if not T else f"{m(l):.1f}") for l in ("published", "wpe_oracle_steering", "perfect_frontend")]
        lines.append(f"| {room} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def md_primary(Tt: list[dict], set_name: str, model: str = "motionAware") -> str:
    lines = []
    for kind, a, b in COMPARISONS:
        rs = [r for r in Tt if r["rooms"] == set_name and r["model"] == model and r["metric"] == PRIMARY
              and r["a"] == a and r["b"] == b]
        if not rs:
            continue
        lines.append(f"\n*{kind}: {a} - {b}*\n")
        lines += [f"- {MOVEMENTS[r['movement']]}: {S.fmt_paired(r, 'pp')}" for r in rs]
    return "\n".join(lines)


def md_sensitivity(Tt: list[dict], set_name: str, model: str = "motionAware") -> str:
    mets = list(METRICS)
    idx = {(r["a"], r["b"], r["movement"], r["metric"]): r for r in Tt
           if r["rooms"] == set_name and r["model"] == model}
    lines = ["| Comparison | Movement | " + " | ".join(m.replace("drr_repo", "drr_wide") for m in mets) + " |", "|---|---|" + "---|" * len(mets)]
    for kind, a, b in COMPARISONS:
        for mov, name in MOVEMENTS.items():
            if not any((a, b, mov, m) in idx for m in mets):
                continue
            cells = []
            for m in mets:
                r = idx.get((a, b, mov, m))
                cells.append("--" if r is None else f"{r['median']:+.1f}{'*' if r['p_wilcoxon'] < 0.05 else ''} "
                                                    f"({r['down']}/{r['n']})")
            lines.append(f"| {a} - {b} | {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def signed_drr(data: dict, take: str, label: str, model: str) -> float:
    r = data[take].get((label, model))
    return S.num(r["drrErrPaper"]) if r else math.nan          # the paper's window


def md_ladder(data: dict) -> str:
    """Median [interquartile range] over takes of |T20 error| (T_mid) and |DRR error| (paper's window) per condition,
    not walking vs walking, both models: the figure. And the median SIGNED DRR error: |error| hides that
    fixed steering moves the walking estimate from too low to too high."""
    labs = list(LADDER)
    lines = ["| Model | Group | Metric | Takes | " + " | ".join(labs) + " |", "|---|---|---|---|" + "---|" * len(labs)]
    for model, mname in (("motionAware", "Motion-aware"), ("omni", "Omni")):
        for g, movs in GROUPS.items():
            T = [t for t in data if movement(t) in movs]
            for metric, f in (("abs. T20 error, T_mid (%)", lambda t, lab: value(data, t, lab, model, PRIMARY)),
                              ("abs. DRR error (dB)", lambda t, lab: value(data, t, lab, model, "drr_paper"))):
                cells = []
                for lab in labs:
                    q1, med, q3 = np.nanpercentile([f(t, lab) for t in T], [25, 50, 75])
                    cells.append(f"{med:.1f} [{q1:.1f}-{q3:.1f}]")
                lines.append(f"| {mname} | {g} | {metric} | {len(T)} | " + " | ".join(cells) + " |")
            cells = [f"{np.nanmedian([signed_drr(data, t, lab, model) for t in T]):+.1f}" for lab in labs]
            lines.append(f"| {mname} | {g} | DRR, signed (dB) | {len(T)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def md_means(data: dict) -> str:
    """Mean primary error per condition and movement (motion-aware)."""
    labs = list(LADDER) + ["informed"]
    lines = ["| Movement | Takes | " + " | ".join(labs) + " |", "|---|---|" + "---|" * len(labs)]
    for mov, name in MOVEMENTS.items():
        T = [t for t in data if movement(t) == mov]
        if not T:
            continue
        cells = []
        for lab in labs:
            v = np.array([value(data, t, lab, "motionAware", PRIMARY) for t in T])
            v = v[np.isfinite(v)]
            cells.append(f"{v.mean():.1f} ({v.size})" if v.size else "--")
        lines.append(f"| {name} | {len(T)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def figure(data: dict, path: Path) -> None:
    """The ladder: |T20 error| (T_mid) and |DRR error| (paper's window) per condition, not walking (static and
    head-only, which follow the same trends) vs walking, for each model; median and interquartile range over
    takes, as the paired tests use. Medians beyond the axis are drawn as a triangle at its top, labelled with median
    [interquartile range]."""
    models = [("motionAware", "Motion-aware"), ("omni", "Omnidirectional")]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.8), sharex=True, sharey="col", constrained_layout=True)
    x = np.arange(len(LADDER))
    for row, (model, mname) in enumerate(models):
        for col, (title, met, hi) in enumerate((("|T20 error|, T_mid (%)", PRIMARY, 40),
                                                 ("|DRR error| (dB)", "drr_paper", 10))):
            ax = axes[row, col]
            for k, (g, movs) in enumerate(GROUPS.items()):
                T = [t for t in data if movement(t) in movs]
                q = np.array([np.nanpercentile([value(data, t, lab, model, met) for t in T], [25, 50, 75])
                              for lab in LADDER])
                q1, med, q3 = q[:, 0], q[:, 1], q[:, 2]
                xs = x + (k - 0.5) * 0.12
                inside = med <= hi                          # no bar around a median drawn at the axis edge
                ax.errorbar(xs[inside], med[inside], yerr=[med[inside] - q1[inside], q3[inside] - med[inside]],
                            fmt="none", ecolor=GROUP_COLOURS[g], elinewidth=1.2, capsize=2.5, alpha=0.75, zorder=2)
                # markers only: the conditions are categories, so no line joins them
                ax.plot(xs[inside], med[inside], "o", ms=6, color=GROUP_COLOURS[g], zorder=3,
                        markeredgecolor="white", markeredgewidth=1,
                        label=f"{g} ({len(T)} takes)" if row == 0 and col == 0 else None)
                # a median beyond the axis: a triangle just inside its top, labelled beside it (clear of the title)
                ytop = 0.95 * hi
                ax.plot(xs[~inside], np.full((~inside).sum(), ytop), "^", ms=7, color=GROUP_COLOURS[g], zorder=3,
                        markeredgecolor="white", markeredgewidth=1)
                for xi, a, m, b in zip(xs[~inside], q1[~inside], med[~inside], q3[~inside]):
                    ax.annotate(f"{m:.0f} [{a:.0f}-{b:.0f}]", (xi, ytop), xytext=(7, 0), textcoords="offset points",
                                ha="left", va="center", fontsize=8, color=GROUP_COLOURS[g])
            ax.set_ylim(0, hi)
            ax.set_title(f"{mname}: {title}", loc="left", fontsize=10, color=INK, pad=12)
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(MUTED)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.tick_params(axis="x", colors=INK)
    for ax in axes[-1]:
        ax.set_xticks(x, list(LADDER.values()))
    axes[0, 0].legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper right")
    fig.suptitle("Front end: from no dereverberation to a perfect pseudo-reference (median and interquartile range "
                 "over takes)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> int:
    data = load()
    if not any(data.values()):
        raise SystemExit(f"no {DIR.relative_to(ROOT)}/*_anchors.csv yet: run python run.py anchors first")
    Tt = [r for k in ROOM_SETS for r in tests(data[k], k)]
    if Tt:
        with open(DIR / "tests.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(Tt[0]))
            w.writeheader()
            w.writerows({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in r.items()} for r in Tt)
    figure(data["all"], DIR / "fig_ladder.png")
    md = f"""# RQ2: the front end

Generated by `analysis/frontend.py` from `results/frontend/*_anchors.csv`. Motion-aware RLS unless
stated. Primary metric: mean |T20 error| at 500 Hz and 1 kHz (T_mid). Tests: median paired
difference A - B [bootstrap 95% CI], pairs where A was lower, exact sign and Wilcoxon p;
* = Wilcoxon p < 0.05. Negative = A better.

## Headroom when not walking (static + head-only pooled, T_mid, motion-aware)

{pooled_headroom(data["all"])}

## Deviation from the informed estimate of the same take (T_mid; pp, signed mean in brackets)
{md_deviation(data["all"])}

## The ladder (fig_ladder.png): median [interquartile range] over takes; signed DRR: positive = too high

{md_ladder(data["all"])}

## Mean primary error per condition (takes in brackets)

{md_means(data["all"])}

## Per room (mean primary error, motion-aware)

{md_per_room(data["all"])}

## Primary metric
{md_primary(Tt, "all")}

## Sensitivity: every metric (median paired difference, pairs down / n; T20 in pp, DRR and tonal in dB)

{md_sensitivity(Tt, "all")}
"""
    (DIR / "summary.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
