#!/usr/bin/env python3
"""Scene geometry from the glasses alone: room volume and source distance from each take's
MPS semidense point cloud (RQ3).

    python geometry/scene.py [<take> ...]      # default: every take with an exported bearing

Per take, the point cloud (points with dist_std < 5 cm) is moved into the take's frame W'
(origin = the tracked CPF at the reference pose, x = forward there, z up; build/pose_mps.py),
using the reference pose build_take stored in the take's meta:  p_W' = Rz(-yaw_ref) (p - p_ref).

  room box   rotated about z to the wall alignment that makes the horizontal coordinate
             histograms sharpest (walls are planes), then the box spanned by the 1st-99th
             percentile of each axis. No tuned threshold. Stray points (through doors and
             glass, above the ceiling) make it larger than the room by a similar factor in
             every room; a common scale error is absorbed by the geometric DRR's constant,
             only its room-to-room variation costs accuracy. An 'outermost strong wall'
             estimator was rejected: in large rooms the far walls carry fewer points than
             nearby furniture.
  distance   along the glasses' acoustic source bearing (pipeline/sourceBearing.m, measured
             from the head's acoustic centre at the reference pose, c0 = (-0.08, 0)): the points
             within +-9 deg of it, in a height band -0.8..+0.3 m around eye level, 0.6-20 m away;
             the nearest 30 cm window at least half as dense as the densest one (along its
             bearing the loudspeaker is one of the strongest SLAM clusters, and what lies behind
             it is partly occluded: the plain nearest surface fires on foreground clutter, the
             densest one on the wall behind the source); r = the median horizontal distance of
             the points in that window. The 0.5 density ratio was chosen on eight walking takes
             of the four rooms recorded first; it is frozen for everything else. The cone is sized
             to the acoustic bearing's own error, which on those four rooms is 3.0 deg (median),
             6.5 deg (90th percentile) and 9.5 deg (max) against the protocol bearing: at 2 m the
             loudspeaker spans only a few degrees, so a cone narrower than the bearing error
             grazes it and lets a wide nearby surface or the wall behind win. +-9 deg is the
             narrowest cone with no failure on those rooms (+-7 deg failed two of the 27 takes).

Blind: no source signal, no oracle -- the acoustic bearing and the visual geometry only. Writes
results/geometry/scene.csv (take, room box, distance) and a top / side view per take in
results/geometry/vision/.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
ROOMS = json.loads((ROOT / "config" / "rooms.json").read_text())["rooms"]
RAW = Path(json.loads((ROOT / "config" / "paths.json").read_text())["raw_root"])
OUT = ROOT / "results" / "geometry"
C0 = np.array([-0.08, 0.0])          # the head's acoustic centre at the reference pose (acousticCentre.m)
STD_MAX = 0.05                       # m, MPS point depth uncertainty
CONE, ZBAND, RMIN, RMAX = 9.0, (-0.8, 0.3), 0.6, 20.0     # CONE: deg, sized to the bearing error (above)
WIN, STEP, REL = 0.30, 0.05, 0.5


def bearing(take: str) -> dict:
    return json.loads((OUT / "bearing" / f"{take}.json").read_text())


def load_points(take: str) -> np.ndarray:
    """The take's semidense points with dist_std < STD_MAX, in the take frame W'."""
    room = take.split("_")[0]
    rc, tc = ROOMS[room], ROOMS[room]["takes"][take]
    mps = RAW / rc["aria_dir"] / (tc.get("mps") or f"mps_{Path(tc['vrs']).stem}_vrs")
    P = np.loadtxt(mps / "slam" / "semidense_points.csv.gz", delimiter=",", skiprows=1, usecols=(2, 3, 4, 6))
    meta = sio.loadmat(ROOT / "data" / "rooms" / f"{take}.mat", variable_names=["meta"],
                       squeeze_me=True, struct_as_record=False)["meta"]
    yaw, p_ref = float(meta.yaw_ref_world_rad), np.asarray(meta.p_ref_world_m, float)
    c, s = np.cos(-yaw), np.sin(-yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    xyz = (P[:, :3] - p_ref) @ Rz.T
    return xyz[P[:, 3] < STD_MAX]


def room_box(xyz: np.ndarray, q=(1, 99)) -> dict:
    bins = np.arange(-30, 30, 0.05)
    rot = lambda ang: xyz[:, :2] @ np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]]).T
    score = lambda u: sum(np.sum(np.histogram(u[:, k], bins=bins)[0].astype(float) ** 2) for k in (0, 1))
    best = max(np.arange(0, 90, 1.0), key=lambda a: score(rot(np.radians(a))))
    u = rot(np.radians(best))
    span = lambda v: float(np.percentile(v, q[1]) - np.percentile(v, q[0]))
    L, W, H = span(u[:, 0]), span(u[:, 1]), span(xyz[:, 2])
    return dict(V_m3=L * W * H, L_m=max(L, W), W_m=min(L, W), H_m=H, S_m2=2 * (L * W + L * H + W * H), align_deg=best)


def strong_surface(d: np.ndarray) -> tuple[float, int]:
    """The nearest WIN-wide window at least REL times as dense as the densest one; its median distance."""
    d = np.sort(d)
    grid = np.arange(RMIN, 8.0, STEP)
    cnt = np.searchsorted(d, grid + WIN) - np.searchsorted(d, grid)
    if cnt.max() == 0:
        return float("nan"), 0
    i = int(np.argmax(cnt >= REL * cnt.max()))
    sel = d[(d >= grid[i]) & (d < grid[i] + WIN)]
    return float(np.median(sel)), int(cnt[i])


def vision_distance(xyz: np.ndarray, bearing_deg: float) -> dict:
    v = xyz[:, :2] - C0
    dist = np.hypot(v[:, 0], v[:, 1])
    daz = (np.degrees(np.arctan2(v[:, 1], v[:, 0])) - bearing_deg + 180) % 360 - 180
    inband = (xyz[:, 2] >= ZBAND[0]) & (xyz[:, 2] <= ZBAND[1])
    ray = inband & (np.abs(daz) <= CONE) & (dist >= RMIN) & (dist <= RMAX)
    r, n = strong_surface(dist[ray])
    return dict(r_m=r, n_surface=n, n_ray=int(ray.sum()), dist=dist, daz=daz, inband=inband, ray=ray)


def figure(take: str, xyz: np.ndarray, b: dict, e: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ray, inb, r_meas = e["ray"], e["inband"], b["distance_m"]
    near = inb & (np.abs(xyz[:, 0]) < 12) & (np.abs(xyz[:, 1]) < 12)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    a = ax[0]
    sub = np.flatnonzero(near)[:: max(1, near.sum() // 40000)]
    a.scatter(xyz[sub, 0], xyz[sub, 1], s=1, c="0.75", label="SLAM points in the height band")
    a.scatter(xyz[ray, 0], xyz[ray, 1], s=3, c="tab:blue", label=f"within +-{CONE:g} deg of the bearing")
    br = np.radians(b["bearing_deg"])
    L = 1.3 * max(e["r_m"] if np.isfinite(e["r_m"]) else 3, r_meas)
    a.plot([C0[0], C0[0] + L * np.cos(br)], [C0[1], C0[1] + L * np.sin(br)], c="tab:orange", lw=1.5, label="acoustic bearing")
    if np.isfinite(e["r_m"]):
        a.plot(C0[0] + e["r_m"] * np.cos(br), C0[1] + e["r_m"] * np.sin(br), "r*", ms=16,
               label=f"vision distance {e['r_m']:.2f} m")
    th = np.linspace(-np.pi, np.pi, 200)
    a.plot(C0[0] + r_meas * np.cos(th), C0[1] + r_meas * np.sin(th), "g--", lw=1, label=f"measured distance {r_meas:.2f} m")
    a.plot(*C0, "k^", ms=9, label="wearer at the reference pose")
    a.set_aspect("equal"); a.grid(alpha=0.3); a.legend(fontsize=7, loc="best")
    a.set_xlabel("x forward (m)"); a.set_ylabel("y left (m)")
    a.set_title("top view, height band %.1f..%.1f m re eye level" % ZBAND, fontsize=9)
    s = ax[1]
    cz = (np.abs(e["daz"]) <= CONE) & (e["dist"] < 15)
    s.scatter(e["dist"][cz], xyz[cz, 2], s=2, c="0.6", label="points along the bearing (all heights)")
    s.scatter(e["dist"][ray], xyz[ray, 2], s=3, c="tab:blue", label="in the height band")
    s.axhspan(*ZBAND, color="tab:blue", alpha=0.07)
    s.axvline(r_meas, c="g", ls="--", label="measured")
    if np.isfinite(e["r_m"]):
        s.axvline(e["r_m"], c="r", label="vision")
    s.set_xlabel("horizontal distance from the wearer (m)"); s.set_ylabel("height re eye level (m)")
    s.grid(alpha=0.3); s.legend(fontsize=7, loc="best")
    s.set_title("side view along the bearing", fontsize=9)
    fig.suptitle(f"{take}: bearing {b['bearing_deg']:+.1f} deg, vision {e['r_m']:.2f} m, measured {r_meas:.2f} m",
                 fontweight="bold")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(takes: list[str]) -> int:
    (OUT / "vision").mkdir(parents=True, exist_ok=True)
    rows = []
    for t in takes:
        b = bearing(t)
        xyz = load_points(t)
        box = room_box(xyz)
        e = vision_distance(xyz, b["bearing_deg"])
        figure(t, xyz, b, e, OUT / "vision" / f"{t}.png")
        rows.append({"take": t, **{k: round(v, 3) if isinstance(v, float) else v for k, v in box.items()},
                     "bearing_deg": round(b["bearing_deg"], 2), "r_vision_m": round(e["r_m"], 3),
                     "r_measured_m": round(b["distance_m"], 3), "n_surface": e["n_surface"], "n_ray": e["n_ray"],
                     "n_points": len(xyz)})
        print(f"{t:36s} box {box['L_m']:5.1f} x {box['W_m']:4.1f} x {box['H_m']:3.1f} m = {box['V_m3']:5.0f} m3 | "
              f"bearing {b['bearing_deg']:+5.1f} deg, vision {e['r_m']:.2f} m, measured {b['distance_m']:.2f} m", flush=True)
    with open(OUT / "scene.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return 0


if __name__ == "__main__":
    args = sys.argv[1:] or sorted(p.stem for p in (OUT / "bearing").glob("*.json"))
    raise SystemExit(main(args))
