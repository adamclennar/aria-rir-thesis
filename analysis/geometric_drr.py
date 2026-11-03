#!/usr/bin/env python3
"""RQ3: DRR from the glasses' scene geometry, against the pipeline's acoustic DRR.

    python analysis/geometric_drr.py [--tape]

The diffuse-field model predicts the broadband (100 Hz - 8 kHz) DRR at the reference pose:

    DRR_geo = 10 log10( A_eff / (16 pi r^2) ) + c

  A_eff  the room's absorption area, A_j = 0.161 V / T60_j per octave, combined as the
         reverberant energy the broadband DRR integrates: 1/A_eff = sum_j w_j / A_j / sum_j w_j,
         w_j the octave's bandwidth inside 100-8000 Hz (getDrr's band). Absorption is MEASURED,
         through each room's own decay: T60 is the blind pipeline's T20, nothing is assumed
         about materials
  r      source distance at the reference pose
  c      one constant: the source's directivity toward the wearer, the head-worn array's own
         directivity, and the model's departure from a diffuse field. The first two belong to the
         source and the device; the last depends partly on the room, so c varies somewhat between
         rooms

Inputs, per variant. Every variant uses the SLAM room box volume: the rooms' tape dimensions
(config dims_m) approximate them as boxes, which most are not, so they are not a volume reference.
The variants swap in the glasses' T20 and distance one at a time:
  reference inputs  SLAM V, the reference T20, the measured r (the omni sweep's delay,
                    reference.mat)
  blind T20         as reference inputs, with the blind pipeline's T20
  glasses           SLAM V, the blind T20, the vision distance (geometry/scene.py): the glasses alone
  acoustic          the blind pipeline's own DRR (published front end, motion-aware), raw and with a
                    constant offset fitted the same way as c -- the fair comparison: a calibrated
                    geometric estimate against an equally calibrated acoustic one
With --tape, also the tape-volume variants (tape V with the reference or the blind T20, the
measured r) and the SLAM / tape volume ratio: for when the rooms' volumes are measured properly.

Calibration: on the four rooms recorded first, c is fitted leave-one-room-out (each room's
prediction uses c from the other three). The held-out small meeting room is predicted with c
fitted once on all four, frozen. DRR is scored against the array reference under both
direct-sound windows. Writes results/geometry/drr_summary.md, drr_per_take.csv and fig_drr.png
(|DRR error| of the acoustic pipeline's own estimate against the glasses' geometry, the paper's window: the
thesis reports the paper's window only; the summary keeps both).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

import stats as S

ROOT = Path(__file__).resolve().parents[1]
GEO = ROOT / "results" / "geometry"
ROOMS = json.loads((ROOT / "config" / "rooms.json").read_text())["rooms"]
MAIN = ("office", "corridor", "reverb", "meeting")
HELD = "smallmeeting"
MOVEMENTS = {"static": "Static", "headMove": "Head-only", "fullMove": "Walking"}
FC = np.array([125, 250, 500, 1000, 2000, 4000, 8000.0])
W = np.minimum(FC * np.sqrt(2), 8000) - np.maximum(FC / np.sqrt(2), 100)
WINDOWS = {"repo": ("drr_ref_db", "drrDb"), "paper": ("drr_ref_paper_db", "drrDbPaper")}
WLABEL = {"repo": "wide", "paper": "paper's"}            # how the windows are named for readers
ACOUSTIC, GEOMETRY = "#eb6834", "#4a3aa7"          # the blind acoustic estimate keeps the blind colour
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def drr_geo(V: float, T60, r: float) -> float:
    A = 0.161 * V / np.asarray(T60, float)
    return 10 * math.log10((np.sum(W) / np.sum(W / A)) / (16 * math.pi * r ** 2))


def movement(take: str) -> str:
    return next(m for m in MOVEMENTS if f"_{m}" in take)


def load() -> list[dict]:
    summ = {(r["take"], r["condition"]): r for r in csv.DictReader(open(ROOT / "results" / "pipeline" / "summary.csv"))}
    scene = {r["take"]: r for r in csv.DictReader(open(GEO / "scene.csv"))}
    rows = []
    for f in sorted((GEO / "bearing").glob("*.json")):
        b = json.loads(f.read_text())
        t, room = b["take"], b["room"]
        bl = summ.get((t, "blind_motionAware"))
        if bl is None or t not in scene or room not in MAIN + (HELD,):
            continue
        T_ref = np.asarray(b["rt_ref_s"], float)
        dims = ROOMS[room].get("dims_m")
        rows.append(dict(
            take=t, room=room, movement=movement(t), T_ref=T_ref,
            T_blind=T_ref * (1 + np.array([S.num(bl[f"t20Err_{int(fc)}"]) for fc in FC]) / 100),
            V_tape=float(np.prod(dims)) if dims else None, V_slam=float(scene[t]["V_m3"]),
            r_meas=float(b["distance_m"]), r_vis=float(scene[t]["r_vision_m"]),
            ref={w: float(b[k]) for w, (k, _) in WINDOWS.items()},
            acoustic={w: S.num(bl[c]) for w, (_, c) in WINDOWS.items()}))
    return rows


VARIANTS = {
    "reference inputs": lambda r: drr_geo(r["V_slam"], r["T_ref"], r["r_meas"]),
    "blind T20": lambda r: drr_geo(r["V_slam"], r["T_blind"], r["r_meas"]),
    "glasses": lambda r: drr_geo(r["V_slam"], r["T_blind"], r["r_vis"]) if np.isfinite(r["r_vis"]) else None,
}
TAPE_VARIANTS = {                     # --tape only: the tape volume is a box approximation, not a reference
    "tape V, reference": lambda r: drr_geo(r["V_tape"], r["T_ref"], r["r_meas"]) if r["V_tape"] else None,
    "tape V, blind T20": lambda r: drr_geo(r["V_tape"], r["T_blind"], r["r_meas"]) if r["V_tape"] else None,
}
# the cost of each of the glasses' inputs: candidate minus the variant it replaces, paired by take
INPUT_COSTS = (("blind T20 for the reference T20", "blind T20", "reference inputs"),
               ("vision distance for the measured distance", "glasses", "blind T20"))


def calibrated(rows: list[dict], pred, w: str) -> tuple[dict, dict]:
    """{take: calibrated prediction}, {fold: c}. Main rooms leave-one-room-out; the held-out room with
    c from all four main rooms."""
    out, cs = {}, {}
    def c_from(train_rooms):
        v = [r["ref"][w] - pred(r) for r in rows if r["room"] in train_rooms and pred(r) is not None]
        return float(np.mean(v)) if v else None
    for fold in MAIN + (HELD,):
        c = c_from([m for m in MAIN if m != fold] if fold in MAIN else MAIN)
        cs[fold] = c
        for r in rows:
            if r["room"] == fold and pred(r) is not None and c is not None:
                out[r["take"]] = pred(r) + c
    return out, cs


def figure(rows: list[dict], P: dict, w: str, path: Path) -> None:
    """|DRR error|, the acoustic pipeline's own estimate (uncalibrated) against the glasses' geometry: median and interquartile
    range over takes. The four main rooms by movement (constant fitted leave-one-room-out); the held-out room
    not walking vs walking (constant frozen)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    main = [(name, [r for r in rows if r["room"] in MAIN and r["movement"] == m]) for m, name in MOVEMENTS.items()]
    held = [("Not walking", [r for r in rows if r["room"] == HELD and r["movement"] != "fullMove"]),
            ("Walking", [r for r in rows if r["room"] == HELD and r["movement"] == "fullMove"])]
    keep = lambda rs: [r for r in rs if r["take"] in P["glasses"] and r["take"] in P["acoustic"]]
    err = lambda rs, k: np.array([abs(P[k][r["take"]] - r["ref"][w]) for r in keep(rs)])
    top = max(np.percentile(err(rs, k), 75) for _, rs in main + held for k in ("glasses", "acoustic"))
    fig = plt.figure(figsize=(10.4, 3.9), layout="constrained")
    sub = fig.subfigures(1, 2, width_ratios=[3, 2.05], wspace=0.04)
    for sf, groups, head in ((sub[0], main, "Four rooms (constant fitted leave-one-room-out)"),
                             (sub[1], held, "Small meeting room, held out (constant frozen)")):
        axes = sf.subplots(1, len(groups), sharey=True)
        sf.suptitle(head, x=0.02, ha="left", fontsize=10, color=INK)
        for ax, (title, rs) in zip(axes, groups):
            for x, k, col in ((0, "acoustic", ACOUSTIC), (1, "glasses", GEOMETRY)):
                q1, med, q3 = np.percentile(err(rs, k), [25, 50, 75])
                ax.plot([x, x], [q1, q3], color=col, lw=1.6, zorder=2)
                for q in (q1, q3):
                    ax.plot([x - 0.08, x + 0.08], [q, q], color=col, lw=1.2, zorder=2)
                ax.scatter([x], [med], s=64, color=col, edgecolor="white", linewidth=1.2, zorder=3)
                ax.text(x + 0.14, med, f"{med:.1f}", va="center", fontsize=8, color=INK)
            ax.set_title(title, loc="left", fontsize=9.5, color=INK)
            ax.set_xticks([0, 1], ["acoustic\npipeline", "glasses'\ngeometry"])
            ax.set_xlim(-0.5, 1.6)
            ax.set_ylim(0, top * 1.1)
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(MUTED)
            ax.tick_params(colors=MUTED, labelsize=8.5)
            ax.tick_params(axis="x", colors=INK)
        if groups is main:
            axes[0].set_ylabel("|DRR error| (dB)", fontsize=9, color=MUTED)
    fig.suptitle("DRR from the glasses' scene geometry against the acoustic pipeline (median and interquartile "
                 "range over takes)", x=0.005, ha="left", fontsize=11, color=INK)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tape", action="store_true", help="also the tape-volume variants and the SLAM / tape ratio")
    a = ap.parse_args()
    variants = {**VARIANTS, **(TAPE_VARIANTS if a.tape else {})}
    rows = load()
    preds = {}
    lines = ["# RQ3: DRR from the glasses' scene geometry\n",
             f"Generated by `analysis/geometric_drr.py{' --tape' if a.tape else ''}`. {len(rows)} takes. Errors against "
             f"the array reference DRR (dB). Calibration: leave-one-room-out on {', '.join(MAIN)}; {HELD} (held out) "
             f"with the constant fitted on all four, frozen. Every variant but the tape ones uses the SLAM room box "
             f"volume.\n"]
    for w in WINDOWS:
        P, C = {}, {}
        for name, f in variants.items():
            P[name], C[name] = calibrated(rows, f, w)
        P["acoustic, calibrated"], C["acoustic, calibrated"] = calibrated(rows, lambda r, w=w: r["acoustic"][w], w)
        P["acoustic"] = {r["take"]: r["acoustic"][w] for r in rows}
        preds[w] = P
        err = lambda name, rs: [P[name][r["take"]] - r["ref"][w] for r in rs if r["take"] in P[name]]
        lines.append(f"## {WLABEL[w]} window: |DRR error| (dB), median (mean) [takes]\n")
        order = ["acoustic", "acoustic, calibrated", *variants]
        lines.append("| Rooms | Movement | " + " | ".join(order) + " |")
        lines.append("|---|---|" + "---|" * len(order))
        for label, rooms in (("four rooms (LORO)", MAIN), (f"{HELD} (held out)", (HELD,))):
            for mov in [None, *MOVEMENTS]:
                rs = [r for r in rows if r["room"] in rooms and (mov is None or r["movement"] == mov)]
                cells = []
                for name in order:
                    e = np.abs(err(name, rs))
                    cells.append(f"{np.median(e):.2f} ({np.mean(e):.2f}) [{len(e)}]" if len(e) else "--")
                lines.append(f"| {label} | {'all' if mov is None else MOVEMENTS[mov]} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("Paired: |error| of the glasses' estimate minus |error| of the calibrated acoustic one "
                     "(negative = geometry better):\n")
        for label, rooms in (("four rooms (LORO)", MAIN), (f"{HELD} (held out)", (HELD,))):
            rs = [r for r in rows if r["room"] in rooms and r["take"] in P["glasses"]]
            d = [abs(P["glasses"][r["take"]] - r["ref"][w]) - abs(P["acoustic, calibrated"][r["take"]] - r["ref"][w])
                 for r in rs]
            lines.append(f"- {label}: {S.fmt_paired(S.paired(d), 'dB')}")
        lines.append("\nThe same against the acoustic pipeline's own estimate, uncalibrated (fig_drr.png):\n")
        groups = [(f"four rooms, {name}", [r for r in rows if r["room"] in MAIN and r["movement"] == m])
                  for m, name in MOVEMENTS.items()]
        groups = [("four rooms (LORO), all", [r for r in rows if r["room"] in MAIN])] + groups + [
            (f"{HELD} (held out), all", [r for r in rows if r["room"] == HELD]),
            (f"{HELD} (held out), not walking", [r for r in rows if r["room"] == HELD and r["movement"] != "fullMove"]),
            (f"{HELD} (held out), walking", [r for r in rows if r["room"] == HELD and r["movement"] == "fullMove"])]
        for label, rs in groups:
            rs = [r for r in rs if r["take"] in P["glasses"]]
            d = [abs(P["glasses"][r["take"]] - r["ref"][w]) - abs(P["acoustic"][r["take"]] - r["ref"][w]) for r in rs]
            lines.append(f"- {label}: {S.fmt_paired(S.paired(d), 'dB')}")
        lines.append("\nThe cost of each of the glasses' inputs: |error| with it minus |error| without it, paired by take "
                     "(positive = the glasses' input is worse; the medians in the table above need not pair):\n")
        for label, a_name, b_name in INPUT_COSTS:
            for rlabel, rooms in (("four rooms (LORO)", MAIN), (f"{HELD} (held out)", (HELD,))):
                rs = [r for r in rows if r["room"] in rooms and r["take"] in P[a_name] and r["take"] in P[b_name]]
                d = [abs(P[a_name][r["take"]] - r["ref"][w]) - abs(P[b_name][r["take"]] - r["ref"][w]) for r in rs]
                lines.append(f"- {label}, {rlabel}: {S.fmt_paired(S.paired(d), 'dB')}")
        lines.append("\nThe constant c per fold (dB; the fold's room is left out, the held-out room uses all four):\n")
        lines.append("| Variant | " + " | ".join(MAIN + (HELD,)) + " |")
        lines.append("|---|" + "---|" * (len(MAIN) + 1))
        for name in (*variants, "acoustic, calibrated"):
            lines.append(f"| {name} | " + " | ".join("--" if C[name][k] is None else f"{C[name][k]:+.2f}"
                                                      for k in MAIN + (HELD,)) + " |")
        lines.append("")

    # the geometry's own accuracy, as DRR: the distance always; the volume only against a volume reference (--tape)
    lines.append("## The vision distance against the measured one, as DRR (dB)\n")
    lines.append("Distance: 20 log10(r_measured / r_vision); the measured distance is the omni sweep's direct-path "
                 "delay." + (" Volume (--tape): 10 log10(V_SLAM / V_tape), against the tape box approximation; a common "
                             "factor is absorbed by c, only its spread between rooms costs accuracy." if a.tape else
                             " The volume is not assessed: the tape dimensions are a box approximation.") + "\n")
    head = ["Room", "Takes", "vision - measured r (m)", "vision / measured r, max |dev|", "distance as DRR, median (max)"]
    if a.tape:
        head += ["V_SLAM / V_tape", "volume as DRR"]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "---|" * len(head))
    for room in MAIN + (HELD,):
        rs = [r for r in rows if r["room"] == room]
        ok = [r for r in rs if np.isfinite(r["r_vis"])]
        dr = [r["r_vis"] - r["r_meas"] for r in ok]
        dev = [abs(r["r_vis"] / r["r_meas"] - 1) for r in ok]
        dd = [20 * math.log10(r["r_meas"] / r["r_vis"]) for r in ok]
        cells = [room, str(len(rs)), f"{np.median(dr):+.2f}", f"{100 * np.max(dev):.1f}%",
                 f"{np.median(np.abs(dd)):.2f} ({np.max(np.abs(dd)):.2f})"]
        if a.tape:
            vr = [r["V_slam"] / r["V_tape"] for r in rs if r["V_tape"]]
            cells += [f"{np.median(vr):.2f}" if vr else "no tape", f"{10 * math.log10(np.median(vr)):+.2f}" if vr else "--"]
        lines.append("| " + " | ".join(cells) + " |")
    md = "\n".join(lines) + "\n"
    (GEO / "drr_summary.md").write_text(md)
    figure(rows, preds["paper"], "paper", GEO / "fig_drr.png")      # the paper's window, primary

    with open(GEO / "drr_per_take.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        names = ["acoustic", "acoustic, calibrated", *variants]
        wr.writerow(["take", "room", "movement", "V_slam_m3", "V_tape_m3", "r_vision_m", "r_measured_m"]
                    + [f"ref_{w}" for w in WINDOWS] + [f"{n}_{w}" for w in WINDOWS for n in names])
        for r in rows:
            wr.writerow([r["take"], r["room"], r["movement"], f"{r['V_slam']:.1f}",
                         "" if r["V_tape"] is None else f"{r['V_tape']:.1f}", f"{r['r_vis']:.3f}", f"{r['r_meas']:.3f}"]
                        + [f"{r['ref'][w]:.3f}" for w in WINDOWS]
                        + [("" if r["take"] not in preds[w][n] else f"{preds[w][n][r['take']]:.3f}") for w in WINDOWS for n in names])
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
