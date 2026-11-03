#!/usr/bin/env python3
"""RQ2: does the array's motion count twice when the reference already carries it?

    python analysis/movingref.py

Reads results/frontend/*_anchors.csv (the anchors, translation on: perfect_frontend,
informed_lvl_matched, informed, published) and results/movingref/*_movingref.csv
(pipeline/movingReference.m). The primary measure is the T_mid deviation from the dry source at
matched level (informed_lvl_matched with the same model family, translation on), as in frontend.py:
the estimator's own error is taken out, so what is left is how far the reference moves the estimate.
Raw |T_mid error| and |DRR error| (paper's window) beside it.

Test 1, which part of the motion matters and whether the glasses can remove it: each reference's
deviation from the dry one, and the paired change against the perfect front end (published_demoved:
against published). Both models.
Test 2, whether the model counts the motion a second time: the 2 x 2 reference carries the motion
(perfect_frontend, informed_lvl_matched) x model translates (motionAware, motionAwareNoTrans), each
cell's deviation from dry-and-translating. Double counting predicts that the moving reference without
translation lands near the dry reference with it. The published reference with and without
translation beside it, as the practical reading.
Then the readings the thesis draws from these: the front-end headroom when walking with the motion left
in and removed from both references (the published reference stands in for fixed steering, which
changes T_mid by at most 0.6 pp); the omnidirectional model's paired changes with the motion removed;
whether the deviation grows with the take's range change, in pp and in ms (the deviation times the
reference T20: a room with a short T20 shows the same shift in time as a larger percentage); and the
corrected omnidirectional DRR against the geometric DRR of RQ3 (geometric_drr.py's glasses variant, the
paper's window, the same array reference), raw and calibrated as geometric_drr.py calibrates the acoustic
estimate (one constant, leave-one-room-out).

Writes results/movingref/summary.md.
"""
from __future__ import annotations

import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import geometric_drr as G
import stats as S

ROOT = Path(__file__).resolve().parents[1]
FRONT, OUT = ROOT / "results" / "frontend", ROOT / "results" / "movingref"
GEO = ROOT / "results" / "geometry"
ROOMS = ("office", "corridor", "reverb", "meeting", "smallmeeting")
HELD = "smallmeeting"
MOVEMENTS = {"static": "Static", "headMove": "Head-only", "fullMove": "Walking"}
DRY = "informed_lvl_matched"
FAMILY = {"motionAware": "motionAware", "motionAwareNoTrans": "motionAware", "omni": "omni"}
TEST1 = [  # (label, name, compare against)
    ("informed", "Dry source, own level (the level check)", None),
    ("received_gain_only", "Received: 1/r gain only", "perfect_frontend"),
    ("received_delay_only", "Received: range delay only", "perfect_frontend"),
    ("perfect_frontend", "Received: delay and gain (perfect front end)", None),
    ("perfect_frontend_demoved", "Perfect front end, motion removed by the glasses", "perfect_frontend"),
    ("published", "Published pseudo-reference", None),
    ("published_demoved", "Published, motion removed by the glasses", "published"),
]
TEST2 = [  # (label, name)
    (DRY, "Dry source"),
    ("perfect_frontend", "Received (perfect front end)"),
    ("published", "Published pseudo-reference"),
]


def movement(take: str) -> str:
    return next(m for m in MOVEMENTS if f"_{m}" in take)


def load() -> dict:
    """take -> (label, model) -> row, for the takes movingReference has run on."""
    data = defaultdict(dict)
    takes = [Path(f).name[:-len("_movingref.csv")] for f in glob.glob(str(OUT / "*_movingref.csv"))]
    for t in takes:
        for f in (FRONT / f"{t}_anchors.csv", OUT / f"{t}_movingref.csv"):
            if f.exists():
                for r in csv.DictReader(open(f)):
                    data[t][(r["label"], r["model"])] = r
    return data


def errs(r: dict | None, bands=S.BAND_SETS[S.PRIMARY]) -> np.ndarray:
    return np.array([S.num(r[f"t20Err_{b}"]) if r else math.nan for b in bands])


def deviation(data: dict, t: str, label: str, model: str) -> float:
    """Mean |e - e_dry| over the T_mid bands; e_dry with the same model family, translation on."""
    d = errs(data[t].get((label, model))) - errs(data[t].get((DRY, FAMILY[model])))
    return float(np.mean(np.abs(d)))


def t_mid_ref_s(t: str) -> np.ndarray:
    """The take's reference T20 (s) in the T_mid bands, as the pipeline scored against it."""
    b = json.loads((GEO / "bearing" / f"{t}.json").read_text())
    return np.array([b["rt_ref_s"][b["band_hz"].index(f)] for f in S.BAND_SETS[S.PRIMARY]], float)


def deviation_ms(data: dict, t: str, label: str, model: str) -> float:
    """deviation() in time: mean |e - e_dry| x the reference T20 over the T_mid bands (ms)."""
    d = errs(data[t].get((label, model))) - errs(data[t].get((DRY, FAMILY[model])))
    return float(np.mean(np.abs(d) / 100 * t_mid_ref_s(t) * 1e3))


def take_value(data: dict, t: str, key: str) -> float:
    """A per-take field movingReference.m writes into every row (the motion-removal checks)."""
    r = next((r for r in data[t].values() if key in r), None)
    return S.num(r[key]) if r else math.nan


def diff(x: list, y: list) -> list:
    return [a - b for a, b in zip(x, y)]


def room(t: str) -> str:
    return t.split("_")[0]


def raw(data: dict, t: str, label: str, model: str) -> float:
    return float(np.mean(np.abs(errs(data[t].get((label, model))))))


def drr(data: dict, t: str, label: str, model: str) -> float:
    r = data[t].get((label, model))
    return abs(S.num(r["drrErrPaper"])) if r else math.nan


def med_iqr(v) -> str:
    v = np.array([x for x in v if math.isfinite(x)])
    if not v.size:
        return "--"
    return f"{np.median(v):.1f} [{np.percentile(v, 25):.1f}-{np.percentile(v, 75):.1f}]"


def med(v) -> str:
    v = [x for x in v if math.isfinite(x)]
    return f"{np.median(v):.1f}" if v else "--"


def short(r: dict | None) -> str:
    """median change, pairs down / n, Wilcoxon p."""
    if r is None:
        return "--"
    return f"{r['median']:+.1f}, {r['down']}/{r['n']}, p={r['p_wilcoxon']:.3f}{'*' if r['p_wilcoxon'] < 0.05 else ''}"


def n_takes(data: dict, T: list, label: str, model: str) -> int:
    return sum(1 for t in T if math.isfinite(deviation(data, t, label, model)))


def test1(data: dict, T: list, model: str) -> list[str]:
    lines = ["| Reference | Takes | Deviation from dry, median [IQR] (pp) | Paired change in deviation "
             "(pp, takes closer / n, Wilcoxon p) | Raw T_mid error (%) | DRR error (dB) |", "|---|---|---|---|---|---|"]
    for label, name, ref in TEST1:
        if not n_takes(data, T, label, model):
            continue
        dv = [deviation(data, t, label, model) for t in T]
        pr = "--"
        if ref:
            pr = f"vs {ref}: " + short(S.paired([deviation(data, t, label, model) - deviation(data, t, ref, model)
                                                 for t in T]))
        lines.append(f"| {name} | {n_takes(data, T, label, model)} | {med_iqr(dv)} | {pr} | "
                     f"{med(raw(data, t, label, model) for t in T)} | {med(drr(data, t, label, model) for t in T)} |")
    return lines


def test2(data: dict, T: list) -> list[str]:
    lines = ["| Reference | Model translates | Takes | Deviation from dry + translating, median [IQR] (pp) | "
             "Raw T_mid error (%) | DRR error (dB) |", "|---|---|---|---|---|---|"]
    for label, name in TEST2:
        for model, tr in (("motionAware", "yes"), ("motionAwareNoTrans", "no")):
            if not n_takes(data, T, label, model):
                continue
            dv = [deviation(data, t, label, model) for t in T]
            lines.append(f"| {name} | {tr} | {n_takes(data, T, label, model)} | {med_iqr(dv)} | "
                         f"{med(raw(data, t, label, model) for t in T)} | {med(drr(data, t, label, model) for t in T)} |")
    return lines


def paired_lines(data: dict, T: list) -> list[str]:
    """The comparisons the 2 x 2 is read by: median paired difference [95% CI], pairs down, sign and Wilcoxon."""
    dv = lambda lab, mod: [deviation(data, t, lab, mod) for t in T]
    rw = lambda lab, mod: [raw(data, t, lab, mod) for t in T]
    dr = lambda lab, mod: [drr(data, t, lab, mod) for t in T]
    rows = [
        ("Received, translation off minus on (deviation)", diff(dv("perfect_frontend", "motionAwareNoTrans"),
                                                               dv("perfect_frontend", "motionAware")), "pp"),
        ("Received, translation off, minus the level check (deviation)",
         diff(dv("perfect_frontend", "motionAwareNoTrans"), dv("informed", "motionAware")), "pp"),
        ("Dry, translation off minus received, translation on (deviation)",
         diff(dv(DRY, "motionAwareNoTrans"), dv("perfect_frontend", "motionAware")), "pp"),
        ("Published, translation off minus on (raw T_mid error)",
         diff(rw("published", "motionAwareNoTrans"), rw("published", "motionAware")), "pp"),
        ("Published, translation off minus on (DRR error)",
         diff(dr("published", "motionAwareNoTrans"), dr("published", "motionAware")), "dB"),
        ("Perfect front end, motion removed minus as is (raw T_mid error)",
         diff(rw("perfect_frontend_demoved", "motionAware"), rw("perfect_frontend", "motionAware")), "pp"),
        ("Published, motion removed minus as is (raw T_mid error)",
         diff(rw("published_demoved", "motionAware"), rw("published", "motionAware")), "pp"),
        ("Published, motion removed minus as is (DRR error)",
         diff(dr("published_demoved", "motionAware"), dr("published", "motionAware")), "dB"),
    ]
    return [f"- {name}: {S.fmt_paired(S.paired(d), unit)}" for name, d, unit in rows]


def removal(data: dict, T: list) -> str:
    """The takes' motion-removal checks (movingReference.m writes them into every row), medians over takes."""
    m = {k: med(take_value(data, t, k) for t in T) for k in
         ("rangeChangeMaxMs", "glassesDelayErrRmsMs", "demovedResidOracleDb", "demovedResidGlassesDb")}
    return (f"Motion removal (medians over takes): the range change reaches {m['rangeChangeMaxMs']} ms; the "
            f"glasses' delay estimate differs from the oracle's by {m['glassesDelayErrRmsMs']} ms rms; the perfect "
            f"front end with the motion removed is {m['demovedResidOracleDb']} dB from the dry source with the "
            f"oracle's geometry and {m['demovedResidGlassesDb']} dB with the glasses'.")


def headroom_lines(data: dict, T: list) -> list[str]:
    """The perfect front end minus the published reference, with the motion left in and removed from both.
    First the stand-in itself: the published reference against fixed steering (wpe_oracle_steering), the
    reference the thesis defines the headroom against, which was not run with the motion removed."""
    rows, apart = [], []
    for model, mlab in (("motionAware", "Motion-aware"), ("omni", "Omni")):
        dv = lambda lab: [deviation(data, t, lab, model) for t in T]
        rw = lambda lab: [raw(data, t, lab, model) for t in T]
        gap = [float(np.mean(np.abs(errs(data[t].get(("published", model))) -
                                    errs(data[t].get(("wpe_oracle_steering", model)))))) for t in T]
        apart.append(f"{mlab}: {med(gap)} pp, deviation median published {med(dv('published'))} "
                     f"[IQR {med_iqr(dv('published')).split('[')[1]}, fixed steering {med(dv('wpe_oracle_steering'))}")
        rows += [
            (f"{mlab}, stand-in: published minus fixed steering, motion left in (deviation)",
             diff(dv("published"), dv("wpe_oracle_steering")), "pp"),
            (f"{mlab}, stand-in: published minus fixed steering, motion left in (raw T_mid error)",
             diff(rw("published"), rw("wpe_oracle_steering")), "pp"),
            (f"{mlab}, motion left in (deviation)", diff(dv("perfect_frontend"), dv("published")), "pp"),
            (f"{mlab}, motion removed from both (deviation)",
             diff(dv("perfect_frontend_demoved"), dv("published_demoved")), "pp"),
            (f"{mlab}, motion left in (raw T_mid error)", diff(rw("perfect_frontend"), rw("published")), "pp"),
            (f"{mlab}, motion removed from both (raw T_mid error)",
             diff(rw("perfect_frontend_demoved"), rw("published_demoved")), "pp"),
        ]
    return (["How far apart the published and fixed-steering estimates are on a take (median of their mean |T20 error "
             "difference| over the T_mid bands), beside the two references' deviations over takes:", ""]
            + [f"- {a}" for a in apart] + [""] + [f"- {name}: {S.fmt_paired(S.paired(d), unit)}" for name, d, unit in rows])


def omni_lines(data: dict, T: list) -> list[str]:
    """The omnidirectional model with the motion removed: paired changes, and against the motion-aware model."""
    rw = lambda lab, mod: [raw(data, t, lab, mod) for t in T]
    dr = lambda lab, mod: [drr(data, t, lab, mod) for t in T]
    rows = [
        ("Published, motion removed minus as is (raw T_mid error)",
         diff(rw("published_demoved", "omni"), rw("published", "omni")), "pp"),
        ("Published, motion removed minus as is (DRR error)",
         diff(dr("published_demoved", "omni"), dr("published", "omni")), "dB"),
        ("Perfect front end, motion removed minus as is (raw T_mid error)",
         diff(rw("perfect_frontend_demoved", "omni"), rw("perfect_frontend", "omni")), "pp"),
        ("Perfect front end, motion removed minus as is (DRR error)",
         diff(dr("perfect_frontend_demoved", "omni"), dr("perfect_frontend", "omni")), "dB"),
        ("Published with the motion removed, omni, minus published as is, motion-aware (raw T_mid error)",
         diff(rw("published_demoved", "omni"), rw("published", "motionAware")), "pp"),
        ("Published with the motion removed, omni, minus published as is, motion-aware (DRR error)",
         diff(dr("published_demoved", "omni"), dr("published", "motionAware")), "dB"),
    ]
    return [f"- {name}: {S.fmt_paired(S.paired(d), unit)}" for name, d, unit in rows]


def dose(data: dict, T: list) -> list[str]:
    """Does the deviation grow with the take's range change? Spearman over takes, in pp and in ms, then per room."""
    rc = [take_value(data, t, "rangeChangeMaxMs") for t in T]
    lines = []
    for label, model in (("perfect_frontend", "motionAware"), ("received_delay_only", "motionAware"),
                         ("perfect_frontend", "omni")):
        for unit, f in (("pp", deviation), ("ms", deviation_ms)):
            r = spearmanr(rc, [f(data, t, label, model) for t in T])
            lines.append(f"- {label} ({model}), deviation in {unit}: Spearman rho {r.statistic:+.2f}, p={r.pvalue:.3f}")
    lines += ["", "| Room | Takes | Range change (ms) | Reference T_mid (s) | Deviation, perfect front end, MA (pp) | "
                  "(ms) | Delay only, MA (pp) | (ms) |", "|---|---|---|---|---|---|---|---|"]
    for rm in ROOMS:
        R = [t for t in T if room(t) == rm]
        if not R:
            continue
        lines.append(f"| {rm} | {len(R)} | {med(take_value(data, t, 'rangeChangeMaxMs') for t in R)} | "
                     f"{np.median([t_mid_ref_s(t).mean() for t in R]):.2f} | "
                     + " | ".join(med(f(data, t, lab, "motionAware") for t in R)
                                  for lab in ("perfect_frontend", "received_delay_only")
                                  for f in (deviation, deviation_ms)) + " |")
    return lines


def geometry_lines(data: dict, T: list) -> list[str]:
    """The corrected omnidirectional DRR against the geometric DRR (RQ3's glasses variant: calibrated
    leave-one-room-out on the four main rooms, frozen for the held-out room), |error|, the paper's window.
    The acoustic estimates raw and calibrated by geometric_drr.calibrated, the same procedure: one constant
    per fold, here fitted on the training rooms' walking takes, the only ones movingReference ran on."""
    # recomputed rather than read from drr_per_take.csv, which rounds to 0.001 dB: enough to swap near-tied ranks
    g_rows = G.load()
    ref = {r["take"]: r["ref"]["paper"] for r in g_rows}
    g_pred, _ = G.calibrated(g_rows, G.VARIANTS["glasses"], "paper")
    geo = {t: abs(g_pred[t] - ref[t]) for t in g_pred}
    cal_rows = [dict(take=t, room=room(t), ref={"paper": ref[t]}) for t in T]
    est = {}
    for key, lab, mod in (("omni_rm", "published_demoved", "omni"), ("ma", "published", "motionAware")):
        e = {t: S.num(data[t][(lab, mod)]["drrDbPaper"]) for t in T}
        pred, cs = G.calibrated(cal_rows, lambda r: e[r["take"]], "paper")
        est[key] = {t: abs(e[t] - ref[t]) for t in T}
        est[key + "_cal"] = {t: abs(pred[t] - ref[t]) for t in pred}
        est[key + "_c"] = cs
    col = lambda R, k: [(geo if k == "geo" else est[k]).get(t, math.nan) for t in R]
    cols = (("geo", "Geometric"), ("omni_rm", "Omni, motion removed"), ("omni_rm_cal", "calibrated"),
            ("ma", "Motion-aware, published"), ("ma_cal", "calibrated"))
    groups = (("all walking", T), ("four rooms", [t for t in T if room(t) != HELD]),
              (f"{HELD} (held out)", [t for t in T if room(t) == HELD]))
    lines = ["| Takes | n | " + " | ".join(n for _, n in cols) + " |", "|---" * (len(cols) + 2) + "|"]
    for name, R in groups:
        lines.append(f"| {name} | {len(R)} | " + " | ".join(med(col(R, k)) for k, _ in cols) + " |")
    lines.append("")
    for name, R in groups:
        lines += [f"- {name}, {a_name} minus {b_name}: {S.fmt_paired(S.paired(diff(col(R, a), col(R, b))), 'dB')}"
                  for a, a_name, b, b_name in (
                      ("omni_rm", "omni with the motion removed", "geo", "geometric"),
                      ("omni_rm_cal", "the same calibrated", "geo", "geometric"),
                      ("omni_rm_cal", "omni with the motion removed, calibrated", "omni_rm", "uncalibrated"),
                      ("ma", "motion-aware published", "geo", "geometric"),
                      ("ma_cal", "the same calibrated", "geo", "geometric"))]
    lines += ["", "The constant per fold (dB; the fold's room left out, the held-out room uses all four):", "",
              "| Estimate | " + " | ".join(ROOMS) + " |", "|---" * (len(ROOMS) + 1) + "|"]
    for k, n in (("omni_rm_c", "Omni, motion removed"), ("ma_c", "Motion-aware, published")):
        lines.append(f"| {n} | " + " | ".join(f"{est[k][r]:+.2f}" if est[k].get(r) is not None else "--"
                                              for r in ROOMS) + " |")
    return lines


def main() -> None:
    data = load()
    if not data:
        raise SystemExit("no results/movingref/*_movingref.csv yet (python run.py movingref)")
    lines = [
        "# RQ2: does the array's motion count twice?", "",
        "Generated by `analysis/movingref.py` from `results/frontend/*_anchors.csv` and "
        "`results/movingref/*_movingref.csv` (`pipeline/movingReference.m`). Deviation: the mean "
        "|T20 error - T20 error with the dry source at matched level| over 500 Hz and 1 kHz, the dry estimate "
        "with the same model family and the translation on (0 = the reference gives the dry reference's "
        "estimate). Raw errors against the omni reference (T20) and the array reference (DRR, paper's window). "
        "Medians over takes.", ""]
    for mov, mname in MOVEMENTS.items():
        T = sorted(t for t in data if movement(t) == mov)
        if not T:
            continue
        lines += [f"## {mname}: {len(T)} takes", "",
                  "### Test 1: which part of the motion, and can the glasses remove it", "", removal(data, T), ""]
        for model, mlab in (("motionAware", "Motion-aware"), ("omni", "Omni")):
            lines += [f"*{mlab}*", ""] + test1(data, T, model) + [""]
        lines += ["### Test 2: does the model's translation count it a second time (motion-aware)", "",
                  *test2(data, T), "", *paired_lines(data, T), ""]
        if mov != "fullMove":
            continue
        lines += ["### Front-end headroom: perfect front end minus published, motion left in and removed from both", "",
                  *headroom_lines(data, T), "",
                  "### The omnidirectional model with the motion removed", "", *omni_lines(data, T), "",
                  "### Does the deviation grow with the range change?", "",
                  "Range change: the take's largest change in the direct path's delay. Deviation in ms: the "
                  "deviation times the reference T20, averaged over the T_mid bands.", "", *dose(data, T), "",
                  "### Corrected omni DRR against the geometric DRR (|DRR error|, dB, the paper's window, median)", "",
                  *geometry_lines(data, T), ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {(OUT / 'summary.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
