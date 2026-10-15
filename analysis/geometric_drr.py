#!/usr/bin/env python3
"""RQ3: DRR from the glasses' scene geometry, against the pipeline's acoustic DRR.

    python analysis/geometric_drr.py

The diffuse-field model predicts the broadband (100 Hz - 8 kHz) DRR at the reference pose:

    DRR_geo = 10 log10( A_eff / (16 pi r^2) ) + c

  A_eff  the room's absorption area, A_j = 0.161 V / T60_j per octave, combined as the
         reverberant energy the broadband DRR integrates: 1/A_eff = sum_j w_j / A_j / sum_j w_j,
         w_j the octave's bandwidth inside 100-8000 Hz (getDrr's band). Absorption is MEASURED,
         through each room's own decay: T60 is the blind pipeline's T20, nothing is assumed
         about materials
  r      source distance at the reference pose
  c      one constant: the source's directivity toward the wearer, the head-worn array's own
         directivity, and the model's departure from a diffuse field. It is a property of the
         source and the device, not the room

Inputs, per variant:
  measured     tape-measured V (config dims_m), the reference T20, the measured r (the omni
               sweep's delay, reference.mat) -- the model's feasibility with good inputs
  blind T20    as measured, with the blind pipeline's T20
  glasses      SLAM room box V, the blind T20, the vision distance (geometry/scene.py): the
               glasses alone
  acoustic     the blind pipeline's own DRR (published front end, motion-aware), raw and with a
               constant offset fitted the same way as c -- the fair comparison: a calibrated
               geometric estimate against an equally calibrated acoustic one

Calibration: on the four rooms recorded first, c is fitted leave-one-room-out (each room's
prediction uses c from the other three). The held-out small meeting room is predicted with c
fitted once on all four, frozen. DRR is scored against the array reference under both
direct-sound windows. Writes results/geometry/drr_summary.md and drr_per_take.csv.
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
    "measured": lambda r: drr_geo(r["V_tape"], r["T_ref"], r["r_meas"]) if r["V_tape"] else None,
    "blind T20": lambda r: drr_geo(r["V_tape"], r["T_blind"], r["r_meas"]) if r["V_tape"] else None,
    "glasses": lambda r: drr_geo(r["V_slam"], r["T_blind"], r["r_vis"]) if np.isfinite(r["r_vis"]) else None,
}


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


def main() -> int:
    rows = load()
    preds = {}
    lines = ["# RQ3: DRR from the glasses' scene geometry\n",
             f"Generated by `analysis/geometric_drr.py`. {len(rows)} takes. Errors against the array reference DRR "
             f"(dB). Calibration: leave-one-room-out on {', '.join(MAIN)}; {HELD} (held out) with the constant "
             f"fitted on all four, frozen.\n"]
    for w in WINDOWS:
        P, C = {}, {}
        for name, f in VARIANTS.items():
            P[name], C[name] = calibrated(rows, f, w)
        P["acoustic, calibrated"], C["acoustic, calibrated"] = calibrated(rows, lambda r, w=w: r["acoustic"][w], w)
        P["acoustic"] = {r["take"]: r["acoustic"][w] for r in rows}
        preds[w] = P
        err = lambda name, rs: [P[name][r["take"]] - r["ref"][w] for r in rs if r["take"] in P[name]]
        lines.append(f"## {w} window: |DRR error| (dB), median (mean) [takes]\n")
        order = ["acoustic", "acoustic, calibrated", "measured", "blind T20", "glasses"]
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
        lines.append("\nThe constant c per fold (dB; the fold's room is left out, the held-out room uses all four):\n")
        lines.append("| Variant | " + " | ".join(MAIN + (HELD,)) + " |")
        lines.append("|---|" + "---|" * (len(MAIN) + 1))
        for name in ("measured", "blind T20", "glasses", "acoustic, calibrated"):
            lines.append(f"| {name} | " + " | ".join("--" if C[name][k] is None else f"{C[name][k]:+.2f}"
                                                      for k in MAIN + (HELD,)) + " |")
        lines.append("")

    # the geometry's own accuracy, as DRR
    lines.append("## The glasses' geometry against measurement, as DRR (dB)\n")
    lines.append("Volume: 10 log10(V_SLAM / V_tape) -- a common factor is absorbed by c, only its spread "
                 "between rooms costs accuracy. Distance: 20 log10(r_measured / r_vision).\n")
    lines.append("| Room | Takes | V_SLAM / V_tape | volume as DRR | vision - measured r (m) | distance as DRR, median (max) |")
    lines.append("|---|---|---|---|---|---|")
    for room in MAIN + (HELD,):
        rs = [r for r in rows if r["room"] == room]
        vr = [r["V_slam"] / r["V_tape"] for r in rs if r["V_tape"]]
        dr = [r["r_vis"] - r["r_meas"] for r in rs if np.isfinite(r["r_vis"])]
        dd = [20 * math.log10(r["r_meas"] / r["r_vis"]) for r in rs if np.isfinite(r["r_vis"])]
        lines.append(f"| {room} | {len(rs)} | " + (f"{np.median(vr):.2f}" if vr else "no tape") + " | "
                     + (f"{10 * math.log10(np.median(vr)):+.2f}" if vr else "--") + " | "
                     + f"{np.median(dr):+.2f} | {np.median(np.abs(dd)):.2f} ({np.max(np.abs(dd)):.2f}) |")
    md = "\n".join(lines) + "\n"
    (GEO / "drr_summary.md").write_text(md)

    with open(GEO / "drr_per_take.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        names = ["acoustic", "acoustic, calibrated", "measured", "blind T20", "glasses"]
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
