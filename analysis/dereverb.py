#!/usr/bin/env python3
"""RQ2: what the front-end residual contains -- oracle candidates, read against the front-end anchors.

    python analysis/dereverb.py

Reads results/dereverb/eval/<take>_<candidate>.csv (pipeline/evalDereverb.m, the candidates from
dereverb/oracle.py) and the front-end anchors (results/frontend/<take>_anchors.csv), for the takes
where the wearer does not walk (static and head-only). Walking is left out: it has no T_mid headroom
to decompose, and two essentially perfect references (perfect_frontend, informed_lvl_matched) already
give estimates further apart than any candidate could move them.

The signals: the current reference y (WPE, beamformer steered at the oracle bearing: the anchors'
wpe_oracle_steering), the received source x (perfect_frontend), and h, the reference's fixed
response, the long-window cross-spectral estimate of the filter from x to y (100 Hz - 8 kHz). Each
candidate removes one component from the real reference, using x and h:

  H1  the dominant echo       oracle_floor          y filtered by G_direct / G_(direct + echo window)
  H2  every early reflection  oracle_early          the same filter with the 2.5-20 ms window, which
                                                    contains H1's
                              oracle_early_sub      h's 2.5-20 ms applied to x and subtracted
  H3  fixed against non-fixed oracle_fixed          h applied to x: the fixed part only
                              oracle_perfect_resid  x plus (y minus its fixed part): the non-fixed part
                                                    only. With the current reference (both parts) and
                                                    the perfect front end (neither), a 2x2.

oracle_comb (the echo removed by a single-echo model, d(t) = y(t) - a d(t - tau)) is still built and
scored by run.py oracle but not reported: it is a realisable canceller rather than a component
removed, and it is capped by oracle_floor.

Measures, per take and model:
  T_mid  deviation from the informed estimate of the same take (frontend.deviation): primary
  low    the same over 125 and 250 Hz
  DRR    |DRR error|, the paper's window

Reported per group (static, head-only, not walking pooled), each model:
  each candidate minus the current reference, paired by take: median [bootstrap 95% CI], takes
  improved, exact Wilcoxon p (stats.paired). This is the primary result;
  the share of the headroom recovered, median(current - candidate) / median(current - perfect),
  with a bootstrap 95% CI over takes, given only where the perfect front end's own gain is
  significant;
  the H3 2x2: the median at each corner, and per take the interaction
  (current - fixed only) - (non-fixed only - perfect), negative where the two parts partly cancel.
Also: the passthrough check (the unmodified reference through the harness, which must reproduce
wpe_oracle_steering), the references' early residual and coherence, a mean-against-median sign
check, the earlier mean-based shares and per-room values. Writes results/dereverb/summary.md.
"""
from __future__ import annotations

import csv
import glob
import math
from pathlib import Path

import numpy as np

import frontend as F
import stats as S

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "results" / "dereverb" / "eval"
OUT = ROOT / "results" / "dereverb"
CURRENT, PERFECT = "wpe_oracle_steering", "perfect_frontend"
HYPOTHESES = {
    "H1": ("the dominant echo", {"oracle_floor": "Echo removed (filtered)"}),
    "H2": ("every early reflection", {"oracle_early": "Early reflections removed (filtered)",
                                      "oracle_early_sub": "Early reflections removed (subtracted)"}),
    "H3": ("fixed and non-fixed parts", {"oracle_fixed": "Fixed part only",
                                         "oracle_perfect_resid": "Perfect front end + non-fixed part"}),
}
CANDIDATES = {lab: name for _, cands in HYPOTHESES.values() for lab, name in cands.items()}
HYP_OF = {lab: h for h, (_, cands) in HYPOTHESES.items() for lab in cands}
NAMES = {CURRENT: "Current reference (WPE, oracle steering)", **CANDIDATES, PERFECT: "Perfect front end"}
GROUPS = {"Static": ("static",), "Head-only": ("headMove",), "Not walking": ("static", "headMove")}
MODELS = {"motionAware": "motion-aware", "omni": "omni"}
MEASURES = {
    "T_mid": ("deviation from informed, 500 Hz + 1 kHz", "pp",
              lambda d, t, l, m: F.deviation(d, t, l, m)),
    "low": ("deviation from informed, 125 + 250 Hz", "pp",
            lambda d, t, l, m: F.deviation(d, t, l, m, bands=S.BAND_SETS["low"])),
    "DRR": ("|DRR error|, paper's window", "dB",
            lambda d, t, l, m: abs(S.num(d[t][(l, m)]["drrErrPaper"])) if (l, m) in d[t] else math.nan),
}
N_BOOT = 10_000


def load() -> dict:
    data = F.load()["all"]
    for f in glob.glob(str(EVAL / "*.csv")):
        for r in csv.DictReader(open(f)):
            if r["take"] in data:
                data[r["take"]][(r["label"], r["model"])] = r
    return data


def group_takes(data: dict, group: str) -> list[str]:
    """Takes of the group with the current reference and at least one candidate scored."""
    return sorted(t for t in data if F.movement(t) in GROUPS[group] and (CURRENT, "motionAware") in data[t]
                  and any((c, "motionAware") in data[t] for c in CANDIDATES))


def values(data: dict, T: list[str], meas: str, model: str, labels: list[str]) -> np.ndarray:
    """takes x labels, only the takes where every label is finite."""
    f = MEASURES[meas][2]
    rows = [[f(data, t, l, model) for l in labels] for t in T]
    return np.array([r for r in rows if all(math.isfinite(x) for x in r)], float).reshape(-1, len(labels))


def against_current(data, T, meas, model, lab) -> dict | None:
    v = values(data, T, meas, model, [lab, CURRENT])
    return S.paired(v[:, 0] - v[:, 1]) if len(v) else None


def ceiling_significant(data, T, meas, model) -> bool:
    r = against_current(data, T, meas, model, PERFECT)
    return bool(r and r["p_wilcoxon"] < 0.05 and r["median"] < 0)


def share(data, T, meas, model, lab) -> tuple[float, float, float]:
    """median(current - candidate) / median(current - perfect), bootstrap 95% CI over takes."""
    v = values(data, T, meas, model, [CURRENT, lab, PERFECT])
    if len(v) < 2:
        return (math.nan,) * 3
    num, den = v[:, 0] - v[:, 1], v[:, 0] - v[:, 2]
    rng = np.random.default_rng(S.SEED)
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    bd = np.median(den[idx], axis=1)
    boot = np.median(num[idx], axis=1) / np.where(bd == 0, np.nan, bd)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return float(np.median(num) / np.median(den)), float(lo), float(hi)


def fmt_r(r: dict | None) -> str:
    if r is None:
        return "--"
    star = "*" if r["p_wilcoxon"] < 0.05 else ""
    return (f"{r['median']:+.1f} [{r['lo']:+.1f}, {r['hi']:+.1f}], {r['down']}/{r['n']}, "
            f"p={r['p_wilcoxon']:.3f}{star}")


def fmt_share(s: tuple[float, float, float]) -> str:
    p, lo, hi = s
    return "--" if not math.isfinite(p) else f"{100 * p:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}]"


def md_passthrough(data: dict) -> list[str]:
    pt = sorted(t for t in data if ("passthrough", "motionAware") in data[t])
    if not pt:
        return []
    out = ["## Check: the unmodified reference through the harness\n",
           "| Take | Model | T_mid error, passthrough / anchors | DRR error, passthrough / anchors |", "|---|---|---|---|"]
    dmax = [0.0, 0.0]
    for t in pt:
        for m in MODELS:
            a, b = data[t][("passthrough", m)], data[t][(CURRENT, m)]
            ta, tb = F.METRICS[F.PRIMARY][0](a), F.METRICS[F.PRIMARY][0](b)
            da, db = S.num(a["drrErrPaper"]), S.num(b["drrErrPaper"])
            dmax = [max(dmax[0], abs(ta - tb)), max(dmax[1], abs(da - db))]
            out.append(f"| {t} | {m} | {ta:.2f} / {tb:.2f} | {da:+.2f} / {db:+.2f} |")
    out += ["", f"{len(pt)} take(s), both models: largest difference {dmax[0]:.3f} pp (T_mid), "
                f"{dmax[1]:.3f} dB (DRR).\n"]
    return out


def md_signals(data: dict, T: list[str]) -> list[str]:
    out = ["## The reference signals (median over the takes where the wearer does not walk)\n",
           "Early residual: 2.5-20 ms after the direct path, re the direct path. Coherence: with the received "
           "source, compensated for the changing propagation delay (refCohEff).\n",
           "| Condition | Takes | Early residual (dB) | Coherence |", "|---|---|---|---|"]
    for lab in (CURRENT, *CANDIDATES, PERFECT):
        Tl = [t for t in T if (lab, "motionAware") in data[t]]
        e = [S.num(data[t][(lab, "motionAware")]["refEarlyDb"]) for t in Tl]
        c = [S.num(data[t][(lab, "motionAware")]["refCohEff"]) for t in Tl]
        out.append(f"| {NAMES[lab]} | {len(Tl)} | {np.nanmedian(e):+.1f} | {np.nanmedian(c):.2f} |")
    return out + [""]


def md_measure(data: dict, groups: dict, meas: str, model: str) -> list[str]:
    what, unit, _ = MEASURES[meas]
    head = "| Hyp. | Candidate | " + " | ".join(f"{g} ({len(T)})" for g, T in groups.items()) + " |"
    rule = "|---|---|" + "---|" * len(groups)
    out = [f"## {meas}, {MODELS[model]}: {what}\n",
           f"Candidate minus the current reference, paired by take ({unit}): median [95% CI], takes improved / n, "
           "Wilcoxon p (* p < 0.05). Negative = closer to informed (lower error for DRR).\n", head, rule,
           f"| -- | {NAMES[PERFECT]} (ceiling) | " + " | ".join(
               fmt_r(against_current(data, T, meas, model, PERFECT)) for T in groups.values()) + " |"]
    for lab, name in CANDIDATES.items():
        out.append(f"| {HYP_OF[lab]} | {name} | " + " | ".join(
            fmt_r(against_current(data, T, meas, model, lab)) for T in groups.values()) + " |")
    if meas == "DRR" and model == "omni":
        return out + ["", "No shares: the omnidirectional DRR is biased high with every non-informed reference, so "
                          "its headroom does not measure the front end.\n"]
    out += ["", "Share of the headroom recovered: median(current - candidate) / median(current - perfect) "
                "[bootstrap 95% CI]; -- where the perfect front end's own gain is not significant.\n", head, rule]
    ok = {g: ceiling_significant(data, T, meas, model) for g, T in groups.items()}
    for lab, name in CANDIDATES.items():
        out.append(f"| {HYP_OF[lab]} | {name} | " + " | ".join(
            fmt_share(share(data, T, meas, model, lab)) if ok[g] else "--" for g, T in groups.items()) + " |")
    return out + [""]


def md_two_by_two(data: dict, groups: dict) -> list[str]:
    labs = [PERFECT, "oracle_fixed", "oracle_perfect_resid", CURRENT]
    out = ["## H3 as a 2x2 (median at each corner)\n",
           "Corners: neither part (perfect front end), fixed part only, non-fixed part only (perfect front end + "
           "non-fixed part), both (current reference). Interaction, per take: (current - fixed only) - "
           "(non-fixed only - perfect), median [95% CI], Wilcoxon p; 0 if the two parts' costs add, negative "
           "where they partly cancel.\n",
           "| Measure | Model | Group | Takes | Neither | Fixed only | Non-fixed only | Both | Interaction |",
           "|---|---|---|---|---|---|---|---|---|"]
    for meas in MEASURES:
        for model in MODELS:
            for g, T in groups.items():
                v = values(data, T, meas, model, labs)
                if not len(v):
                    continue
                med = np.median(v, axis=0)
                r = S.paired((v[:, 3] - v[:, 1]) - (v[:, 2] - v[:, 0]))
                out.append(f"| {meas} | {MODELS[model]} | {g} | {len(v)} | " + " | ".join(f"{x:.1f}" for x in med)
                           + f" | {fmt_r(r)} |")
    return out + [""]


def md_sign_check(data: dict, groups: dict) -> list[str]:
    bad = []
    for meas in MEASURES:
        for model in MODELS:
            for g, T in groups.items():
                for lab in CANDIDATES:
                    v = values(data, T, meas, model, [lab, CURRENT])
                    if len(v) < 2:
                        continue
                    d = v[:, 0] - v[:, 1]
                    if np.sign(np.mean(d)) != np.sign(np.median(d)):
                        bad.append(f"- {meas}, {MODELS[model]}, {g}, {CANDIDATES[lab]}: mean {np.mean(d):+.1f}, "
                                   f"median {np.median(d):+.1f}")
    return ["## Mean against median\n", "Candidate minus current reference where the mean and the median disagree "
            "in sign (the median is the reported value):\n", *(bad or ["- none"]), ""]


def md_appendix(data: dict, T: list[str]) -> list[str]:
    out = ["## Appendix: shares from the means (the earlier summary's definition)\n",
           "(mean(current) - mean(candidate)) / (mean(current) - mean(perfect)), T_mid deviation, not walking. "
           "A few takes can dominate a mean over 20; the medians above are the reported values.\n",
           "| Candidate | Motion-aware | Omni |", "|---|---|---|"]
    for lab, name in CANDIDATES.items():
        cells = []
        for model in MODELS:
            v = values(data, T, "T_mid", model, [CURRENT, lab, PERFECT])
            m = v.mean(axis=0)
            cells.append(f"{100 * (m[0] - m[1]) / (m[0] - m[2]):.0f}%")
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    labs = [CURRENT, *CANDIDATES, PERFECT]
    out += ["", "## Appendix: per room (T_mid deviation, motion-aware, mean over the room's takes)\n",
            "| Room | Takes | " + " | ".join(NAMES[l] for l in labs) + " |", "|---|---|" + "---|" * len(labs)]
    for room in F.ROOM_SETS["all"]:
        Tr = [t for t in T if t.split("_")[0] == room]
        if Tr:
            out.append(f"| {room} | {len(Tr)} | " + " | ".join(
                f"{np.nanmean([F.deviation(data, t, l, 'motionAware') for t in Tr]):.1f}" for l in labs) + " |")
    return out + [""]


def main() -> int:
    data = load()
    groups = {g: group_takes(data, g) for g in GROUPS}
    T = groups["Not walking"]
    if not T:
        raise SystemExit("no evaluated candidates yet")
    lines = ["# RQ2: what the front-end residual contains\n",
             f"Generated by `analysis/dereverb.py`. Takes where the wearer does not walk: {len(T)} "
             f"({len(groups['Static'])} static, {len(groups['Head-only'])} head-only). Each candidate removes one "
             "component from the current reference (WPE, beamformer steered at the oracle bearing), using the "
             "received source; see the docstring for each. T_mid is the primary measure.\n"]
    lines += md_passthrough(data)
    lines += md_signals(data, T)
    for meas in MEASURES:
        for model in MODELS:
            lines += md_measure(data, groups, meas, model)
    lines += md_two_by_two(data, groups)
    lines += md_sign_check(data, groups)
    lines += md_appendix(data, T)
    md = "\n".join(lines) + "\n"
    (OUT / "summary.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
