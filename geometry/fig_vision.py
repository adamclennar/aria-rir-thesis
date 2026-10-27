#!/usr/bin/env python3
"""The vision distance for one take, as a thesis figure (RQ3).

    python geometry/fig_vision.py [<take>]      # default: meeting_maleSpeech_fullMove

Three panels, all from geometry/scene.py's own computation (nothing is refitted):
  top view    the SLAM points in the height band, the +-CONE deg cone around the acoustic bearing, and
              the window the rule selects: the loudspeaker;
  side view   the points inside the cone at every height, against horizontal distance from the wearer:
              the loudspeaker and its stand show as a column rising from the floor;
  density     the number of points in the height band in each 30 cm window along the bearing, with the
              rule drawn on it: the nearest window at least REL times as dense as the densest.
The default take is the meeting room's clearest: its loudspeaker window is about 55 times a typical
window, and nothing nearer reaches 5 % of it. Writes results/geometry/fig_vision_<take>.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Wedge  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene as SC  # noqa: E402

GREY, BLUE, ORANGE, GREEN = "#b9b7b0", "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
XMAX = 6.0          # m along the bearing shown in the side and density panels


def style(ax) -> None:
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def main(take: str) -> int:
    b = SC.bearing(take)
    xyz = SC.load_points(take)
    e = SC.vision_distance(xyz, b["bearing_deg"])
    dist, daz, inband, ray = e["dist"], e["daz"], e["inband"], e["ray"]
    br, r_vis, r_meas = np.radians(b["bearing_deg"]), e["r_m"], b["distance_m"]

    # the rule, as scene.strong_surface applies it
    d = np.sort(dist[ray])
    grid = np.arange(SC.RMIN, 8.0, SC.STEP)
    cnt = np.searchsorted(d, grid + SC.WIN) - np.searchsorted(d, grid)
    i = int(np.argmax(cnt >= SC.REL * cnt.max()))
    w0, w1 = grid[i], grid[i] + SC.WIN
    sel = ray & (dist >= w0) & (dist < w1)

    fig = plt.figure(figsize=(10.4, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1])
    ax_top, ax_side, ax_den = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])

    # (a) top view: the room in the height band, the cone, the selected window
    near = inband & (np.abs(xyz[:, 0]) < 12) & (np.abs(xyz[:, 1]) < 12)
    bg = np.flatnonzero(near & ~ray)[:: max(1, (near & ~ray).sum() // 40000)]
    ax_top.scatter(xyz[bg, 0], xyz[bg, 1], s=0.5, c=GREY, lw=0)
    ax_top.add_patch(Wedge(SC.C0, XMAX, np.degrees(br) - SC.CONE, np.degrees(br) + SC.CONE, color=BLUE, alpha=0.10, lw=0))
    ax_top.scatter(xyz[ray & ~sel, 0], xyz[ray & ~sel, 1], s=0.8, c=BLUE, lw=0)
    ax_top.scatter(xyz[sel, 0], xyz[sel, 1], s=1.5, c=ORANGE, lw=0)
    ax_top.plot(*SC.C0, marker="^", ms=9, color=INK, ls="none")
    ax_top.plot(SC.C0[0] + r_meas * np.cos(br), SC.C0[1] + r_meas * np.sin(br), "o", ms=11, mfc="none", mec=GREEN,
                mew=1.8, ls="none")
    lo, hi = np.percentile(xyz[near, :2], [1, 99], axis=0)
    ax_top.set_xlim(lo[0] - 0.5, hi[0] + 0.5)
    ax_top.set_ylim(lo[1] - 0.5, hi[1] + 0.5)
    ax_top.set_aspect("equal")
    ax_top.set_xlabel("forward (m)", fontsize=9, color=MUTED)
    ax_top.set_ylabel("left (m)", fontsize=9, color=MUTED)
    ax_top.set_title("(a) Top view, height band", loc="left", fontsize=10, color=INK)
    dot = dict(marker="o", ls="none", ms=4, mec="none")      # legend handles sized by hand, not scaled
    handles = [Line2D([], [], mfc=GREY, label="SLAM points, height band", **dot),
               Line2D([], [], mfc=BLUE, label=f"within ±{SC.CONE:g}° of the acoustic bearing", **dot),
               Line2D([], [], mfc=ORANGE, label="selected window (loudspeaker)", **dot),
               Line2D([], [], marker="^", ls="none", ms=7, color=INK, label="wearer at the reference pose"),
               Line2D([], [], marker="o", ls="none", ms=8, mfc="none", mec=GREEN, mew=1.6,
                      label=f"measured source position ({r_meas:.2f} m)")]
    leg = ax_top.legend(handles=handles, fontsize=7.5, loc="upper left", frameon=True, labelcolor=INK)
    leg.get_frame().set_edgecolor(GRID)
    style(ax_top)

    # (b) side view: every height within the cone
    cone = (np.abs(daz) <= SC.CONE) & (dist < XMAX)
    ax_side.axhspan(*SC.ZBAND, color=BLUE, alpha=0.08, lw=0)
    ax_side.scatter(dist[cone & ~inband], xyz[cone & ~inband, 2], s=0.8, c=GREY, lw=0)
    ax_side.scatter(dist[ray & ~sel & (dist < XMAX)], xyz[ray & ~sel & (dist < XMAX), 2], s=0.8, c=BLUE, lw=0)
    ax_side.scatter(dist[sel], xyz[sel, 2], s=1.2, c=ORANGE, lw=0)
    ax_side.axvline(r_meas, color=GREEN, ls="--", lw=1.3)
    ax_side.set_ylim(-2.2, 1.2)
    ax_side.set_ylabel("height re eye level (m)", fontsize=9, color=MUTED)
    ax_side.annotate("loudspeaker\non its stand", (2.8, -1.1), xytext=(1.3, -1.6), fontsize=8, color=INK,
                     arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9))
    ax_side.set_title(f"(b) Side view, within ±{SC.CONE:g}° of the bearing (shaded: height band)",
                      loc="left", fontsize=10, color=INK)
    style(ax_side)

    # (c) density along the bearing, with the rule
    centres = grid + SC.WIN / 2
    keep = centres <= XMAX
    ax_den.axvspan(w0, w1, color=ORANGE, alpha=0.25, lw=0, label=f"selected window (vision: {r_vis:.2f} m)")
    ax_den.plot(centres[keep], cnt[keep], color=INK, lw=1.3)
    ax_den.axhline(SC.REL * cnt.max(), color=MUTED, ls=":", lw=1.2, label="half the densest")
    ax_den.axvline(r_meas, color=GREEN, ls="--", lw=1.3, label=f"measured: {r_meas:.2f} m")
    ax_den.set_xlabel("horizontal distance from the wearer (m)", fontsize=9, color=MUTED)
    ax_den.set_ylabel("points per 30 cm window", fontsize=9, color=MUTED)
    ax_den.set_title("(c) Points in the height band along the bearing", loc="left", fontsize=10, color=INK)
    ax_den.legend(fontsize=7.5, frameon=False, loc="upper right", labelcolor=INK)
    style(ax_den)
    for ax in (ax_side, ax_den):
        ax.set_xlim(0, XMAX)

    out = SC.OUT / f"fig_vision_{take}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"{take}: vision {r_vis:.2f} m, measured {r_meas:.2f} m, selected window {w0:.2f}-{w1:.2f} m "
          f"({cnt[i]} points, densest {cnt.max()}) -> {out.relative_to(SC.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "meeting_maleSpeech_fullMove"))
