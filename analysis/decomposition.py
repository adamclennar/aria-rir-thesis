#!/usr/bin/env python3
"""RQ2, first part: what movement costs, and what blind estimation costs on top.

    python analysis/decomposition.py

Reads results/pipeline/summary.csv (pipeline/rescore.m) and writes
results/decomposition/:

  grid.csv        per room set x movement x condition: n, mean and median over
                  takes of every metric
  tests.csv       every paired comparison below, every metric
  fig_grid.png    primary T20 error by movement, informed vs blind
                  (motion-aware), one dot per take
  summary.md      the primary metric's grid and tests, then the sensitivity
                  checks

Metrics: mean |T20 error| under the primary band set (T_mid: 500 Hz and 1 kHz)
and the other band groups (stats.BAND_SETS); |DRR error| under both
direct-sound windows; the tonal-balance error at the microphones.

Comparisons, each paired (stats.paired):
  blind cost     blind minus informed, same take, per movement and model
  movement cost  head-only minus static, walking minus static, walking minus
                 head-only, per condition, paired by room and voice (the two
                 walking takes of a room and voice averaged first)
  model          omni minus motion-aware, same take, per movement and estimation

Rooms: all five composite-protocol rooms pooled (the small meeting room is held
out only where something is fitted or chosen); the L-shape takes not at all
(acoustically the corridor; they exist for the pose comparison).
"""
from __future__ import annotations

import csv
import itertools
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import stats as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "pipeline" / "summary.csv"
OUT = ROOT / "results" / "decomposition"

ROOM_SETS = {"all": ("office", "corridor", "reverb", "meeting", "smallmeeting")}
MOVEMENTS = {"static": "Static", "headMove": "Head-only", "fullMove": "Walking"}
CONDITIONS = {"informed_motionAware": "Informed (MA)", "blind_motionAware": "Blind (MA)",
              "informed_omni": "Informed (Omni)", "blind_omni": "Blind (Omni)"}
COLOURS = {"informed_motionAware": "#2a78d6", "blind_motionAware": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def metrics() -> dict:
    """name -> (function of a summary.csv row, unit)."""
    m = {f"t20_{k}": ((lambda r, b=b: S.t20_err(r, b)), "%") for k, b in S.BAND_SETS.items()}
    m["drr_repo"] = (lambda r: abs(S.num(r["drrErr"])), "dB")
    m["drr_paper"] = (lambda r: abs(S.num(r["drrErrPaper"])), "dB")
    m["tonal"] = (lambda r: S.num(r["rmsErrMeanDb"]), "dB")
    return m


METRICS = metrics()
PRIMARY = f"t20_{S.PRIMARY}"


def parse(take: str) -> tuple[str, str, str] | None:
    """(room, voice, movement) from e.g. office_maleSpeech_fullMove2."""
    room, voice, mov = take.split("_")[:3]
    mov = next((m for m in MOVEMENTS if mov.startswith(m)), None)
    return (room, voice.replace("Speech", ""), mov) if mov else None


def load() -> dict:
    """room set -> take -> condition -> row."""
    data = {k: defaultdict(dict) for k in ROOM_SETS}
    for r in csv.DictReader(open(SUMMARY)):
        p = parse(r["take"])
        if not p or r["condition"] not in CONDITIONS:
            continue
        for k, rooms in ROOM_SETS.items():
            if p[0] in rooms:
                data[k][r["take"]][r["condition"]] = r
    return data


def value(data: dict, take: str, cond: str, metric: str) -> float:
    r = data.get(take, {}).get(cond)
    return METRICS[metric][0](r) if r else math.nan


def takes_of(data: dict, mov: str) -> list[str]:
    return sorted(t for t in data if parse(t)[2] == mov)


def grid(data: dict, set_name: str) -> list[dict]:
    rows = []
    for mov in MOVEMENTS:
        T = takes_of(data, mov)
        for cond in CONDITIONS:
            for met in METRICS:
                v = np.array([value(data, t, cond, met) for t in T])
                v = v[np.isfinite(v)]
                rows.append(dict(rooms=set_name, movement=mov, condition=cond, metric=met, n=v.size,
                                 mean=float(v.mean()) if v.size else math.nan,
                                 median=float(np.median(v)) if v.size else math.nan))
    return rows


def by_room_voice(data: dict, mov: str, cond: str, met: str) -> dict:
    """(room, voice) -> mean over that movement's takes."""
    acc = defaultdict(list)
    for t in takes_of(data, mov):
        room, voice, _ = parse(t)
        v = value(data, t, cond, met)
        if math.isfinite(v):
            acc[(room, voice)].append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def tests(data: dict, set_name: str) -> list[dict]:
    out = []

    def add(kind, what, a, b, met, d):
        r = S.paired(d)
        if r:
            out.append(dict(rooms=set_name, kind=kind, comparison=what, a=a, b=b, metric=met, **r))

    for met in METRICS:
        for mov in MOVEMENTS:
            T = takes_of(data, mov)
            for model in ("motionAware", "omni"):
                a, b = f"blind_{model}", f"informed_{model}"
                add("blind cost", f"{MOVEMENTS[mov]}: blind - informed ({model})", a, b, met,
                    [value(data, t, a, met) - value(data, t, b, met) for t in T])
            for est in ("informed", "blind"):
                a, b = f"{est}_omni", f"{est}_motionAware"
                add("model", f"{MOVEMENTS[mov]}: omni - motion-aware ({est})", a, b, met,
                    [value(data, t, a, met) - value(data, t, b, met) for t in T])
        for cond in CONDITIONS:
            for m1, m0 in (("headMove", "static"), ("fullMove", "static"), ("fullMove", "headMove")):
                x1, x0 = by_room_voice(data, m1, cond, met), by_room_voice(data, m0, cond, met)
                add("movement cost", f"{CONDITIONS[cond]}: {MOVEMENTS[m1]} - {MOVEMENTS[m0]}", m1, m0, met,
                    [x1[k] - x0[k] for k in sorted(x1.keys() & x0.keys())])
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in r.items()})


def md_grid(G: list[dict], set_name: str, met: str) -> str:
    unit = METRICS[met][1]
    cell = {(r["movement"], r["condition"]): r for r in G if r["rooms"] == set_name and r["metric"] == met}
    head = "| Movement | " + " | ".join(CONDITIONS.values()) + " | Takes |\n|---|" + "---|" * (len(CONDITIONS) + 1)
    lines = [head]
    for mov, name in MOVEMENTS.items():
        cs = [cell[(mov, c)] for c in CONDITIONS]
        lines.append(f"| {name} | " + " | ".join(f"{c['mean']:.1f} {unit} ({c['median']:.1f})" for c in cs)
                     + f" | {cs[0]['n']} |")
    return "\n".join(lines)


def md_tests(Tt: list[dict], set_name: str, met: str, kinds=("blind cost", "movement cost", "model")) -> str:
    unit = {"%": "pp"}.get(METRICS[met][1], METRICS[met][1])
    lines = []
    for kind in kinds:
        rs = [r for r in Tt if r["rooms"] == set_name and r["metric"] == met and r["kind"] == kind]
        if not rs:
            continue
        lines.append(f"\n*{kind}*\n")
        lines += [f"- {r['comparison']}: {S.fmt_paired(r, unit)}" for r in rs]
    return "\n".join(lines)


def sensitivity(Tt: list[dict], set_name: str) -> str:
    """The blind-cost and movement-cost medians (motion-aware) under every metric, one row per comparison."""
    keep = [("blind cost", "motionAware"), ("movement cost", "Informed (MA)"), ("movement cost", "Blind (MA)")]
    comps = []
    for r in Tt:
        if r["rooms"] == set_name and any(r["kind"] == k and s in r["comparison"] for k, s in keep):
            if r["comparison"] not in comps:
                comps.append(r["comparison"])
    mets = list(METRICS)
    idx = {(r["comparison"], r["metric"]): r for r in Tt if r["rooms"] == set_name}
    head = "| Comparison | " + " | ".join(m.replace("drr_repo", "drr_wide") for m in mets) + " |\n|---|" + "---|" * len(mets)
    lines = [head]
    for c in comps:
        cells = []
        for m in mets:
            r = idx.get((c, m))
            cells.append("--" if r is None else f"{r['median']:+.1f}{'*' if r['p_wilcoxon'] < 0.05 else ''} "
                                                f"({r['down']}/{r['n']})")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def figure(data: dict, path: Path) -> None:
    """The cost of blind estimation for each model: per take, blind minus informed (the paired difference the
    tests use), by movement; median and interquartile range over takes. Above zero, going blind cost
    accuracy. Rows share a y-axis per metric, so the two models compare directly."""
    panels = [("t20_T_mid", "|T20 error|, T_mid", "Blind minus informed (pp)"),
              ("drr_paper", "|DRR error|", "Blind minus informed (dB)")]
    models = (("motionAware", "Motion-aware"), ("omni", "Omnidirectional"))
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.12), sharey="col", constrained_layout=True)
    for (row, (model, mname)), (col, (met, title, ylab)) in itertools.product(enumerate(models), enumerate(panels)):
        ax = axes[row, col]
        ax.axhline(0, color=MUTED, lw=1, zorder=1)
        for i, mov in enumerate(MOVEMENTS):
            d = np.array([value(data, t, f"blind_{model}", met) - value(data, t, f"informed_{model}", met)
                          for t in takes_of(data, mov)])
            q1, med, q3 = np.nanpercentile(d, [25, 50, 75])
            ax.plot([i, i], [q1, q3], color=INK, lw=1.6, solid_capstyle="butt", zorder=2)
            ax.plot([i - 0.07, i + 0.07], [q1, q1], color=INK, lw=1.2, zorder=2)
            ax.plot([i - 0.07, i + 0.07], [q3, q3], color=INK, lw=1.2, zorder=2)
            ax.scatter([i], [med], s=60, color=INK, edgecolor="white", linewidth=1.2, zorder=3)
        ax.set_xticks(range(len(MOVEMENTS)), list(MOVEMENTS.values()))
        ax.set_xlim(-0.5, len(MOVEMENTS) - 0.5)
        ax.set_title(f"{mname}: {title}", loc="left", fontsize=10, color=INK)
        ax.set_ylabel(ylab, fontsize=9, color=MUTED)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        ax.tick_params(axis="x", colors=INK)
    fig.text(0.01, -0.01, "Per take, blind minus informed; dot: median over takes, bar: interquartile range. Above "
             "zero, blind estimation is worse.", fontsize=7.5, color=MUTED, va="top")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load()
    G = [r for k in ROOM_SETS for r in grid(data[k], k)]
    Tt = [r for k in ROOM_SETS for r in tests(data[k], k)]
    write_csv(OUT / "grid.csv", G)
    write_csv(OUT / "tests.csv", Tt)
    figure(data["all"], OUT / "fig_grid.png")
    md = f"""# RQ2: movement and blind estimation

Generated by `analysis/decomposition.py` from `results/pipeline/summary.csv`.
Rooms: {", ".join(ROOM_SETS["all"])}, pooled.

Primary metric: mean |T20 error| at 500 Hz and 1 kHz (T_mid, ISO 3382-1). Cells: mean over takes
(median in brackets). Tests: median paired difference [bootstrap 95% CI], pairs that went down, exact
sign and Wilcoxon p; * = Wilcoxon p < 0.05. Movement cost pairs room and voice.

## Primary metric

{md_grid(G, "all", PRIMARY)}
{md_tests(Tt, "all", PRIMARY)}

## Sensitivity: the same comparisons under every metric (median paired difference, pairs down / n)

T20 in percentage points, DRR and tonal balance in dB.

{sensitivity(Tt, "all")}
"""
    (OUT / "summary.md").write_text(md)
    print(md)
    print(f"wrote {OUT.relative_to(ROOT)}/: summary.md, grid.csv, tests.csv, fig_grid.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
