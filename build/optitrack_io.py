"""OptiTrack pose loading, frame alignment, and clock synchronisation.

Covers Week 2 Steps 8-10. Step 9 is the one your progress report does not
address and it silently biases every result if skipped.

THE FRAME ALIGNMENT PROBLEM
---------------------------
Equation 19 of the report assumes you can express the source direction in
"the glasses frame". OptiTrack does not give you that. It gives the pose
of a MARKER RIGID BODY whose local axes were defined arbitrarily when the
body was created in Motive. The rotation R_cpf_rb between that rigid body
and the Aria CPF is UNKNOWN.

Any error in R_cpf_rb appears as a CONSTANT BIAS in every DoA result you
report, and it is invisible unless you look for it -- a 6 deg systematic
offset looks exactly like "the array has a 6 deg bias", which is the
wrong conclusion.

Solve it via hand-eye calibration (AX = XB) using rotation sequences from
both systems, and REPORT THE RESIDUAL. A stated alignment uncertainty of
1.5 deg is a strength in the evaluation chapter; an unstated one is a hole.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PoseTrack:
    t_s: np.ndarray            # (N,) seconds
    position: np.ndarray       # (N, 3) metres
    rotation: np.ndarray       # (N, 3, 3) world <- body

    def interp_to(self, t_query: np.ndarray) -> "PoseTrack":
        """Resample onto new timestamps (SLERP for rotation)."""
        from scipy.spatial.transform import Rotation, Slerp

        t_query = np.clip(np.asarray(t_query), self.t_s[0], self.t_s[-1])
        pos = np.stack([np.interp(t_query, self.t_s, self.position[:, k])
                        for k in range(3)], axis=1)
        slerp = Slerp(self.t_s, Rotation.from_matrix(self.rotation))
        return PoseTrack(t_query, pos, slerp(t_query).as_matrix())

    def angular_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """Body-frame angular velocity by finite difference.

        Returns (t_mid, omega (N-1, 3)) in rad/s. Used for the clock
        offset estimate against the Aria gyroscope.
        """
        from scipy.spatial.transform import Rotation

        dt = np.diff(self.t_s)
        dR = np.einsum("nij,njk->nik",
                       self.rotation[:-1].transpose(0, 2, 1), self.rotation[1:])
        rotvec = Rotation.from_matrix(dR).as_rotvec()
        return self.t_s[:-1] + dt / 2, rotvec / dt[:, None]


def read_motive_metadata(path: str) -> dict:
    """Read the key/value metadata from row 0 of a Motive CSV export.

    Contains Capture Frame Rate, Rotation Type, Length Units, Coordinate
    Space, Capture Start Time, and Total Frames. Worth logging per
    session -- if Length Units is ever Millimeters or Rotation Type is
    ever Euler, every downstream number silently changes scale or meaning.
    """
    import csv as _csv

    with open(path, newline="") as fh:
        row = next(_csv.reader(fh))
    meta = {}
    for k in range(0, len(row) - 1, 2):
        key = row[k].strip()
        if key:
            meta[key] = row[k + 1].strip()
    return meta


def _find_header_rows(rows: list[list[str]]) -> dict:
    """Locate the Motive header block.

    Layout (Format Version 1.23):
        row 0    metadata key/value pairs
        row 1    blank
        row 2    Type   ("Rigid Body" / "Rigid Body Marker" / "Marker")
        row 3    Name   ("MyRigidBody", "MyRigidBody:Marker1", ...)
        row 4    ID
        row 5    Field  ("Rotation" / "Position")
        row 6    Axis   ("Frame", "Time (Seconds)", "X", "Y", "Z", "W")
        row 7+   data

    Anchored on the row whose first cell is "Frame" rather than on fixed
    indices, so extra metadata rows in other Motive versions do not
    silently shift the mapping.
    """
    axis_idx = None
    for i, row in enumerate(rows[:30]):
        if row and row[0].strip().lower() == "frame":
            axis_idx = i
            break
    if axis_idx is None or axis_idx < 4:
        raise ValueError(
            "could not locate the Motive header block (no row starting "
            "with 'Frame'). Export with headers enabled, or parse the "
            "file yourself and construct a PoseTrack directly."
        )
    return {
        "type": axis_idx - 4,
        "name": axis_idx - 3,
        "id": axis_idx - 2,
        "field": axis_idx - 1,
        "axis": axis_idx,
        "data": axis_idx + 1,
    }


def list_rigid_bodies(path: str) -> dict[str, dict]:
    """List the rigid bodies in a Motive CSV, with their column indices.

    Run this FIRST on any new export. Motive files routinely contain
    several rigid bodies plus per-marker columns, and picking the wrong
    one produces plausible trajectories of the wrong object.

        python -c "import sys; sys.path.insert(0,'src'); \
            from aria_audio.io.optitrack import list_rigid_bodies; \
            print(list_rigid_bodies('take.csv'))"
    """
    import csv as _csv

    with open(path, newline="") as fh:
        rows = list(_csv.reader(fh))
    h = _find_header_rows(rows)
    type_row, name_row = rows[h["type"]], rows[h["name"]]
    field_row, axis_row = rows[h["field"]], rows[h["axis"]]

    bodies: dict[str, dict] = {}
    for c in range(len(type_row)):
        # EXACT type match. "Rigid Body Marker" is a DIFFERENT type and
        # must not be collected here -- markers also carry Position
        # columns, and substring matching on either Type or Name silently
        # returns marker data instead of body data.
        if type_row[c].strip() != "Rigid Body":
            continue
        nm = name_row[c].strip()
        key = f"{field_row[c].strip().lower()}_{axis_row[c].strip().lower()}"
        bodies.setdefault(nm, {})[key] = c
    return bodies


def load_motive_csv(
    path: str,
    body_name: str | None = None,
    up_axis: str = "y",
    strict: bool = True,
) -> PoseTrack:
    """Load one rigid body from a Motive CSV export.

    body_name : EXACT rigid-body name, matched case-insensitively but not
        as a substring. Call list_rigid_bodies() to see the options. If
        omitted and the file has exactly one body, that one is used;
        otherwise this raises rather than guessing.

    up_axis : Motive defaults to Y-up. Set 'y' to convert to Z-up
        (recommended, and what the CPF convention in conventions.py
        assumes), or 'z' to leave the data untouched.

    strict : verify the file's metadata declares Quaternion rotation and
        Meters. Turn off only if you know what you are doing.

    Quaternion component order is read from the file's own axis row
    (X, Y, Z, W), so it does not need to be guessed.

    ALWAYS plot the trajectory afterwards and confirm it matches what
    physically happened in the lab.
    """
    import csv as _csv

    meta = read_motive_metadata(path)
    if strict:
        rot_type = meta.get("Rotation Type", "").lower()
        units = meta.get("Length Units", "").lower()
        if rot_type and rot_type != "quaternion":
            raise ValueError(
                f"Rotation Type is '{meta['Rotation Type']}', expected "
                "Quaternion. Re-export from Motive with quaternions, or "
                "handle Euler angles explicitly (order matters)."
            )
        if units and units not in ("meters", "metres"):
            raise ValueError(
                f"Length Units is '{meta['Length Units']}', expected "
                "Meters. Convert, or every distance is off by 1000x."
            )

    with open(path, newline="") as fh:
        rows = list(_csv.reader(fh))
    h = _find_header_rows(rows)

    bodies = list_rigid_bodies(path)
    if not bodies:
        raise ValueError(f"no rigid bodies found in {path}")

    if body_name is None:
        if len(bodies) > 1:
            raise ValueError(
                f"{len(bodies)} rigid bodies in this file: "
                f"{sorted(bodies)}. Pass body_name explicitly -- guessing "
                "would silently track the wrong object."
            )
        body_name = next(iter(bodies))
    else:
        matches = [b for b in bodies if b.lower() == body_name.lower()]
        if not matches:
            raise KeyError(
                f"rigid body '{body_name}' not found. Available: "
                f"{sorted(bodies)}"
            )
        body_name = matches[0]

    cols = bodies[body_name]
    need = ["rotation_x", "rotation_y", "rotation_z", "rotation_w",
            "position_x", "position_y", "position_z"]
    missing = [k for k in need if k not in cols]
    if missing:
        raise KeyError(f"body '{body_name}' is missing columns: {missing}")

    # Time column: located by name, not assumed to be index 1.
    axis_row = rows[h["axis"]]
    t_col = next(
        (c for c, v in enumerate(axis_row) if "time" in v.strip().lower()), 1
    )

    data_rows = [r for r in rows[h["data"]:] if r and r[0].strip()]
    n = len(data_rows)
    if n == 0:
        raise ValueError("no data rows found")

    t = np.full(n, np.nan)
    pos = np.full((n, 3), np.nan)
    quat = np.full((n, 4), np.nan)

    def val(r, c):
        try:
            s = r[c].strip()
            return float(s) if s else np.nan
        except (ValueError, IndexError):
            return np.nan

    for i, r in enumerate(data_rows):
        t[i] = val(r, t_col)
        pos[i] = [val(r, cols[f"position_{a}"]) for a in "xyz"]
        quat[i] = [val(r, cols[f"rotation_{a}"]) for a in "xyzw"]

    good = (~np.isnan(pos).any(axis=1) & ~np.isnan(quat).any(axis=1)
            & ~np.isnan(t))
    dropped = int((~good).sum())
    if dropped:
        pct = 100.0 * dropped / n
        print(f"WARNING: dropped {dropped}/{n} frames ({pct:.1f}%) with "
              f"missing data for '{body_name}' -- marker occlusion. "
              "Blank cells in a Motive export mean the body was not "
              "solved on that frame.")
        if pct > 5:
            print("  >5% dropout: improve camera coverage before "
                  "trusting head-rotation results.")
    t, pos, quat = t[good], pos[good], quat[good]
    if t.size == 0:
        raise ValueError(f"no valid frames for rigid body '{body_name}'")

    norms = np.linalg.norm(quat, axis=1)
    if strict and not np.allclose(norms, 1.0, atol=1e-3):
        print(f"WARNING: quaternion norms deviate from 1 "
              f"(min {norms.min():.4f}, max {norms.max():.4f}); "
              "check the export.")

    from scipy.spatial.transform import Rotation
    R = Rotation.from_quat(quat).as_matrix()  # scipy expects (x, y, z, w)

    if up_axis.lower() == "y":
        # Motive Y-up -> Z-up:  (x, y, z) -> (x, -z, y)
        C = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
        pos = pos @ C.T
        R = C @ R @ C.T

    return PoseTrack(t_s=t, position=pos, rotation=R)


def drop_frozen_frames(
    track: PoseTrack,
    min_run: int = 3,
    verbose: bool = True,
) -> PoseTrack:
    """Drop repeats within any run of >=min_run byte-identical poses.

    On a marker-solve failure Motive may HOLD the last solved pose
    rather than leaving the cell blank, so the dropout check in
    load_motive_csv() -- which only looks for blank/NaN cells -- passes
    these through silently. The frozen samples then look like perfectly
    good data that merely happens to be stationary.

    Two things go wrong if they are left in. Any fit over the take is
    biased toward the held pose, and, worse, the jump OUT of a freeze
    spans the whole occlusion gap in a single frame interval: divided by
    one 1/120 s step that reads as an enormous angular velocity
    (thousands of deg/s -- physically impossible for a head) which then
    dominates a least-squares fit.

    Dropping the repeats leaves a real sample at each end of the gap but
    no samples inside it. That is deliberate: PoseTrack.interp_to() has
    no concept of a gap and will happily interpolate straight across
    one, so callers sampling near a removed run must ALSO check the gap
    width before trusting the result (see scripts/11's max_gap check).

    First hit in techlab_slam_t4_002.csv: 107 frames frozen at the tail,
    and a separate 625-frame freeze at the start of the same take. Both
    calibration takes of session 2 have them too. See notes/methods.md
    2026-08-17.
    """
    if track.t_s.size < 2:
        return track

    same = np.all(np.diff(track.position, axis=0) == 0, axis=1) & \
        np.all(np.diff(track.rotation, axis=0) == 0, axis=(1, 2))
    keep = np.ones(track.t_s.size, dtype=bool)

    i = 0
    while i < same.size:
        if not same[i]:
            i += 1
            continue
        j = i
        while j < same.size and same[j]:
            j += 1
        if j + 1 - i >= min_run:
            if verbose:
                print(f"  WARNING: dropping {j + 1 - i} frames of frozen "
                      f"(byte-identical) pose, t="
                      f"{track.t_s[i] - track.t_s[0]:.3f}-"
                      f"{track.t_s[j] - track.t_s[0]:.3f}s -- Motive held the "
                      "last solved pose rather than leaving it blank, most "
                      "likely marker occlusion.")
            keep[i + 1:j + 1] = False
        i = j

    return PoseTrack(t_s=track.t_s[keep], position=track.position[keep],
                     rotation=track.rotation[keep])


def drop_implausible_rotation(
    track: PoseTrack,
    max_rate_deg_s: float = 1000.0,
    max_dt_s: float = 0.05,
    verbose: bool = True,
) -> PoseTrack:
    """Drop poses whose rotation from the previous frame is not physical.

    A SEPARATE failure from drop_frozen_frames(). Where a freeze means
    Motive held the last good pose, this catches the opposite: Motive
    resolving the marker constellation into a briefly WRONG orientation
    (a mirrored or permuted rigid-body solution), then recovering. The
    pose is fully populated, non-repeating, and passes every NaN and
    dropout check, so nothing else in this module notices it.

    The gate is biomechanical, not statistical. Peak human head yaw is
    roughly 500-600 deg/s, and session 1's deliberately fast turn peaked
    at 446 deg/s, so a frame-to-frame rotation implying more than
    ~1000 deg/s is a solver artefact rather than motion. Session 2's
    walking takes contain jumps implying up to 22000 deg/s -- about 62
    revolutions per second.

    Intervals longer than `max_dt_s` are skipped rather than judged: a
    large rotation across a real gap is expected and says nothing about
    whether the pose is valid.

    Applies to GROUND TRUTH only. It is a statement about what a head can
    do, never about what the estimator should have said, so it cannot
    flatter a DoA result.
    """
    if track.t_s.size < 2:
        return track

    keep = np.ones(track.t_s.size, dtype=bool)
    for i in range(1, track.t_s.size):
        dt = track.t_s[i] - track.t_s[i - 1]
        if dt <= 0 or dt > max_dt_s:
            continue
        dR = track.rotation[i - 1].T @ track.rotation[i]
        ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
        if ang / dt > max_rate_deg_s:
            keep[i] = False

    n = int((~keep).sum())
    if n and verbose:
        print(f"  dropped {n} poses implying >{max_rate_deg_s:.0f} deg/s "
              "-- Motive mis-solve, not head motion")
    return PoseTrack(t_s=track.t_s[keep], position=track.position[keep],
                     rotation=track.rotation[keep])


def clean_angular_velocity(
    track: PoseTrack,
    max_dt_s: float = 0.009,
) -> tuple[np.ndarray, np.ndarray]:
    """angular_velocity(), keeping only pairs that are genuinely adjacent.

    PoseTrack.angular_velocity() differences EVERY consecutive pair. Once
    frozen or implausible poses have been removed the survivors are no
    longer uniformly spaced, so some pairs straddle a hole.

    Note what this is NOT: angular_velocity() divides by the true dt, so an
    across-hole pair does not produce a huge spike. (An earlier version of
    this docstring claimed it did, by analogy with interp_to(); that was
    wrong. The genuine multi-thousand-deg/s samples in session 2's walking
    takes occur at NORMAL 8 ms spacing and are Motive mis-solves --
    drop_implausible_rotation() is what removes those.)

    What an across-hole pair actually gives is the rotation AVERAGED over
    the hole. That is a perfectly good average and a poor instantaneous
    sample, and joint_align() matches instantaneous gyro readings, so
    including it quietly degrades the fit. On session 2's walking takes
    roughly 750-930 of ~2500 velocity samples per take span a hole, which
    is far too many to leave in.
    """
    t_mid, omega = track.angular_velocity()
    ok = np.diff(track.t_s) <= max_dt_s
    return t_mid[ok], omega[ok]


# -- Step 9: hand-eye frame alignment --------------------------------
#
# estimate_frame_alignment() (orientation-sequence Procrustes, needing
# MPS/VIO orientation) was removed 2026-08-18: unused by every script,
# untested, and superseded in practice by estimate_alignment_from_
# angular_velocity() below, which needs only the raw Aria gyro and no
# MPS. Recoverable from git history if an MPS-orientation-based fit is
# ever specifically wanted.

def estimate_alignment_from_angular_velocity(
    omega_rb: np.ndarray,
    omega_dev: np.ndarray,
    min_rate_rad_s: float = 0.15,
) -> tuple[np.ndarray, float, int]:
    """Frame alignment from ANGULAR VELOCITY alone. No MPS required.

    WHY THIS EXISTS
    ---------------
    R_dev_rb -- the rotation between the OptiTrack rigid body's marker
    frame and Aria's device frame -- can't be read off from pose, even
    with MPS. MPS gives Aria's pose in the SLAM world frame, but the
    rigid body's local axes are arbitrary, fixed only by however the
    markers happened to be stuck on. Nothing measures that rotation
    directly; it has to be SOLVED, by correlating motion the two frames
    both observe.

    Angular velocity is what makes that correlation possible without a
    world frame or any integration: the same physical rotation observed
    in two rigidly-connected frames satisfies

        omega_dev = R_dev_rb @ omega_rb

    at every instant -- so raw Aria gyroscope and OptiTrack orientation
    differences are enough on their own, with no gyro drift to worry
    about either.

    Parameters
    ----------
    omega_rb  : (N, 3) body-frame angular velocity from OptiTrack,
                via PoseTrack.angular_velocity(), resampled onto the
                same timestamps as omega_dev.
    omega_dev : (N, 3) Aria gyroscope, rad/s.
    min_rate_rad_s : discard samples where the rotation is too slow to
                carry direction information. Near-stationary samples are
                dominated by gyro bias and bias would otherwise bend the
                solution.

    Returns
    -------
    (R_dev_rb, residual_deg, n_used)

    residual_deg is the median angle between measured and predicted
    angular-velocity directions. REPORT IT. Above ~5 deg means the
    marker body is loose on the frame, the clock offset is still wrong,
    or the motion lacked rotational diversity.

    IMPORTANT: run estimate_clock_offset() FIRST and apply it. A timing
    error makes the two streams describe different instants and this
    silently returns a rotation that is partly a time shift.
    """
    omega_rb = np.asarray(omega_rb, dtype=float)
    omega_dev = np.asarray(omega_dev, dtype=float)
    if omega_rb.shape != omega_dev.shape:
        raise ValueError("angular velocity arrays must have the same shape; "
                         "resample one onto the other's timestamps first")

    speed = np.minimum(np.linalg.norm(omega_rb, axis=1),
                       np.linalg.norm(omega_dev, axis=1))
    keep = speed > min_rate_rad_s
    n_used = int(keep.sum())
    if n_used < 50:
        raise ValueError(
            f"only {n_used} samples exceed {min_rate_rad_s} rad/s. "
            "Record a take with sustained rotation about all three axes "
            "-- yaw alone is not enough to constrain the rotation."
        )
    A, B = omega_rb[keep], omega_dev[keep]

    # Kabsch. Solve for R with  omega_dev = R @ omega_rb,  i.e. row-wise
    # B = A @ R.T, so the SVD of B.T @ A gives R directly -- NO transpose.
    #
    # Verified against 20 random rotations (recovery error < 1e-13 deg).
    # A transposed rotation is still a valid rotation and passes every
    # shape check, so getting this backwards fails silently and would
    # rotate ground-truth directions the wrong way, biasing every DoA
    # comparison in the thesis.
    U, _, Vt = np.linalg.svd(B.T @ A)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt

    # Residual on the actual vector error, NOT on direction cosine alone.
    # A cosine-only metric reported 0.00 deg for a solution that was
    # 156 deg wrong, because it was compared against the already-rotated
    # prediction rather than the measurement.
    pred = A @ R.T
    err_ang = np.degrees(np.arccos(np.clip(
        np.sum(pred * B, axis=1)
        / (np.linalg.norm(pred, axis=1) * np.linalg.norm(B, axis=1) + 1e-15),
        -1.0, 1.0)))
    residual = float(np.median(err_ang))

    rms_vec = float(np.sqrt(np.mean(np.sum((pred - B) ** 2, axis=1))))
    scale = float(np.sqrt(np.mean(np.sum(B ** 2, axis=1))))
    if scale > 0 and rms_vec / scale > 0.5:
        print(f"WARNING: alignment fit is poor (relative RMS "
              f"{rms_vec / scale:.2f}). Check the clock offset first -- a "
              "timing error looks exactly like a rotation error here.")

    # Rotational diversity check: all three axes must be excited, or the
    # solution is unconstrained about the single axis that was used.
    # compute_uv=False is REQUIRED: the default builds an N x N left
    # singular matrix, which for a 60 s take at 1 kHz is 60000 x 60000
    # (27 GB) and raises MemoryError. Only the singular values are needed.
    sv = np.linalg.svd(A - A.mean(axis=0), compute_uv=False)
    if sv[2] / (sv[0] + 1e-15) < 0.05:
        print(f"WARNING: rotation was nearly planar (axis ratio "
              f"{sv[2] / sv[0]:.3f}). The alignment is poorly constrained "
              "about one axis. Rotate about yaw, pitch AND roll.")

    return R, residual, n_used


# -- Step 10: clock offset -------------------------------------------

def estimate_clock_offset(
    t_a: np.ndarray, omega_a: np.ndarray,
    t_b: np.ndarray, omega_b: np.ndarray,
    max_offset_s: float = 2.0,
    step_s: float = 0.001,
) -> tuple[float, float]:
    """Estimate the clock offset between two angular-velocity streams.

    Cross-correlates angular-velocity MAGNITUDE, which is invariant to
    the unknown frame rotation, so this can run BEFORE Step 9.

    Returns (offset_s, peak_correlation) where t_b + offset aligns with
    t_a. A peak correlation below ~0.7 means the estimate is unreliable
    -- usually too little rotational motion in the take.

    At 100 deg/s of head rotation, 10 ms of residual offset is ~1 deg of
    DoA bias. For static trials this barely matters; for Phase 3 head
    rotation it matters a great deal.
    """
    mag_a = np.linalg.norm(omega_a, axis=1)
    mag_b = np.linalg.norm(omega_b, axis=1)

    t0 = max(t_a[0], t_b[0])
    t1 = min(t_a[-1], t_b[-1])
    if t1 - t0 < 1.0:
        raise ValueError("streams overlap by less than 1 s")

    grid = np.arange(t0, t1, step_s)
    ref = np.interp(grid, t_a, mag_a)
    ref = ref - ref.mean()

    offsets = np.arange(-max_offset_s, max_offset_s, step_s)
    best, best_corr = 0.0, -np.inf
    for off in offsets:
        cand = np.interp(grid, t_b + off, mag_b)
        cand = cand - cand.mean()
        denom = np.linalg.norm(ref) * np.linalg.norm(cand)
        if denom < 1e-12:
            continue
        c = float(np.dot(ref, cand) / denom)
        if c > best_corr:
            best_corr, best = c, float(off)

    if best_corr < 0.7:
        print(f"WARNING: weak sync correlation ({best_corr:.2f}). "
              "Record a take with more vigorous rotation about all axes.")
    return best, best_corr


# estimate_clock_offset_and_drift() (multi-window offset+drift fit)
# removed 2026-08-18: unused by every script, untested. joint_align()
# below fits offset (and implicitly absorbs slow drift within a take)
# directly; a separate drift-only estimator was never actually needed
# in practice. Recoverable from git history if a long-take drift number
# is ever specifically wanted for the methods chapter.

def shift_by_angular_speed(
    t_rb: np.ndarray, omega_rb: np.ndarray,
    t_dev: np.ndarray, omega_dev: np.ndarray,
    resample_hz: float = 200.0,
    n_candidates: int = 40,
    min_overlap: int = 200,
) -> tuple[float, float]:
    """Gross clock shift from angular-SPEED cross-correlation. Returns
    (shift_s, peak_corr) where t_dev - shift aligns with t_rb.

    WHY NOT estimate_clock_offset() OR joint_align()
    ------------------------------------------------
    estimate_clock_offset() cross-correlates the same quantity but over a
    bounded grid (`max_offset_s`, default 2 s). Aria device time is a free
    -running counter and Motive take time restarts at zero every take, so
    the offset between them is THOUSANDS of seconds -- ~2460-2950 s across
    session 2's walking takes. A bounded grid cannot reach that, and a
    1 ms grid over 2500 s is 2.5 M evaluations. This does the whole range
    in one FFT instead.

    joint_align() can find a large offset when driven by scripts/09's
    coarse-to-fine scan, but it solves the ROTATION at the same time, so
    it needs rotational diversity. Walking takes are mostly yaw: on
    session 2's turn takes it warned "nearly planar (axis ratio 0.048)"
    and returned a shift 12.5 s from the truth, which silently sampled
    ground truth from the wrong part of the take entirely.

    Angular SPEED is invariant to the unknown frame rotation, so this
    needs no R and does not care about planarity. Use it when R is already
    known (from a dedicated calibration take) and only timing is wanted --
    do NOT use it to avoid a proper calibration, since it yields no
    rotation at all.

    peak_corr near 1 means a confident lock; below ~0.5 means the two
    streams share too little rotational structure to align. Session 2's
    walking takes score 0.83-1.00.
    """
    a = np.linalg.norm(np.asarray(omega_rb, dtype=float), axis=1)
    b = np.linalg.norm(np.asarray(omega_dev, dtype=float), axis=1)
    if a.size < 2 or b.size < 2:
        raise ValueError("need at least two samples in each stream")

    ga = np.arange(t_rb[0], t_rb[-1], 1.0 / resample_hz)
    gb = np.arange(t_dev[0], t_dev[-1], 1.0 / resample_hz)
    A = np.interp(ga, t_rb, a)
    B = np.interp(gb, t_dev, b)
    A = (A - A.mean()) / (A.std() + 1e-12)
    B = (B - B.mean()) / (B.std() + 1e-12)

    n = A.size + B.size
    N = 1 << int(np.ceil(np.log2(n)))
    raw = np.fft.irfft(np.fft.rfft(B, N) * np.conj(np.fft.rfft(A, N)), N)[:n]

    # The raw FFT correlation is NOT a correlation coefficient: with
    # zero-padding the overlap shrinks away from full alignment, so
    # dividing by a fixed length (an earlier version used
    # min(A.size, B.size)) can exceed 1 and does not rank lags fairly.
    # Score the strongest candidates by a real Pearson correlation over
    # the samples that actually overlap.
    # At lag k the FFT correlation aligns B[i] with A[i - k], so the
    # overlapping index range is max(0, k) <= i < min(nb, na + k).
    na, nb = A.size, B.size
    cands = np.argsort(raw)[::-1][:n_candidates]
    best_lag, best_corr = int(cands[0]), -2.0
    for lag in cands.tolist():
        i_lo, i_hi = max(0, lag), min(nb, na + lag)
        if i_hi - i_lo < min_overlap:
            continue
        b = B[i_lo:i_hi]
        a = A[i_lo - lag:i_hi - lag]
        sa, sb = a.std(), b.std()
        if sa < 1e-9 or sb < 1e-9:
            continue
        c = float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))
        if c > best_corr:
            best_corr, best_lag = c, lag
    return (float(gb[0] - ga[0] + best_lag / resample_hz),
            float(np.clip(best_corr, -1.0, 1.0)))


def joint_align(
    t_rb: np.ndarray, omega_rb: np.ndarray,
    t_dev: np.ndarray, omega_dev: np.ndarray,
    tau_init: float | None = None,
    R_init: np.ndarray | None = None,
    robust: bool = True,
    min_rate_rad_s: float = 0.15,
) -> dict:
    """Estimate rotation AND time offset SIMULTANEOUSLY. Preferred method.

    WHY JOINT, NOT SEQUENTIAL
    -------------------------
    The obvious pipeline is: cross-correlate to get the time offset, then
    Procrustes to get the rotation. That is what estimate_clock_offset()
    followed by estimate_alignment_from_angular_velocity() does, and it is
    statistically wrong, because THE TWO PARAMETERS ARE COUPLED.

    During any smooth rotation, omega changes direction continuously. A
    small time shift therefore looks like a small rotation, and vice
    versa. Estimating the offset first FREEZES an error that the rotation
    step is then forced to absorb -- and neither step can report that it
    happened, because each fits its own stage perfectly.

    Solving for both together lets the coupling resolve itself, gives a
    continuous (not grid-quantised) time offset, and produces a real
    covariance so you can quote an uncertainty rather than a residual.

    Model
    -----
        omega_dev(t) = R @ omega_rb(t - tau)

    SIGN CONVENTION: tau is the offset you ADD TO THE MOCAP TIMESTAMPS to
    put them on the Aria device clock -- identical to what
    estimate_clock_offset(t_dev, w_dev, t_rb, w_rb) returns, so the two
    compose without a sign flip. The opposite convention was used on the
    first pass and produced a tau that was exactly negated: numerically
    perfect, silently backwards, and it would have shifted every mocap
    sample the wrong way by twice the true offset.

    4 free parameters: a 3-vector rotation (axis-angle) and tau. Solved by
    Levenberg-Marquardt on the interpolated residual.

    robust : soft-L1 loss. Marker dropout and mocap gap-filling produce
        occasional wild angular-velocity samples; least squares would let
        a handful of them dominate. Turn off only to compare.

    Returns dict: R, tau_s, residual_deg, tau_std_ms, rot_std_deg,
    n_used, converged.

    tau_std_ms and rot_std_deg come from the Jacobian at the solution.
    These are the numbers to put in the thesis -- they are uncertainties,
    not goodness-of-fit.
    """
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    t_rb = np.asarray(t_rb, dtype=float)
    t_dev = np.asarray(t_dev, dtype=float)
    omega_rb = np.asarray(omega_rb, dtype=float)
    omega_dev = np.asarray(omega_dev, dtype=float)

    # Coarse initialisation. The joint solve is local, so it needs a
    # starting point within roughly half a rotation period.
    if tau_init is None:
        tau_init, _ = estimate_clock_offset(
            t_dev, omega_dev, t_rb, omega_rb, max_offset_s=2.0, step_s=0.002)
    if R_init is None:
        w_rb_i = np.stack([np.interp(t_dev - tau_init, t_rb, omega_rb[:, k])
                           for k in range(3)], axis=1)
        try:
            R_init, _, _ = estimate_alignment_from_angular_velocity(
                w_rb_i, omega_dev, min_rate_rad_s=min_rate_rad_s)
        except ValueError:
            R_init = np.eye(3)

    # Evaluate only where BOTH streams are rotating meaningfully and the
    # interpolation is safe (inside the overlap, with margin for tau).
    margin = abs(tau_init) + 0.5
    lo, hi = max(t_dev[0], t_rb[0] + margin), min(t_dev[-1], t_rb[-1] - margin)
    sel = (t_dev >= lo) & (t_dev <= hi)
    sel &= np.linalg.norm(omega_dev, axis=1) > min_rate_rad_s
    n_used = int(sel.sum())
    if n_used < 100:
        raise ValueError(
            f"only {n_used} usable samples. The take needs sustained "
            "rotation and enough overlap between the two streams.")

    t_eval = t_dev[sel]
    w_dev = omega_dev[sel]

    def residual(params):
        R = Rotation.from_rotvec(params[:3]).as_matrix()
        tau = params[3]
        w_rb_i = np.stack(
            [np.interp(t_eval - tau, t_rb, omega_rb[:, k]) for k in range(3)],
            axis=1)
        return (w_dev - w_rb_i @ R.T).ravel()

    # MULTI-START over tau. The cost surface in tau is periodic-ish for
    # oscillatory motion, so a single start from a noisy cross-correlation
    # peak can settle in a local minimum -- and the covariance then reports
    # a tiny sigma for an answer that is milliseconds wrong. Restarting
    # across a spread of tau and keeping the lowest cost avoids that
    # failure, which is otherwise invisible.
    rot0 = Rotation.from_matrix(R_init).as_rotvec()
    starts = [tau_init + d for d in
              (0.0, -0.05, 0.05, -0.02, 0.02, -0.1, 0.1)]
    sol = None
    for tau_s0 in starts:
        cand = least_squares(
            residual, np.concatenate([rot0, [tau_s0]]), method="trf",
            loss="soft_l1" if robust else "linear",
            f_scale=0.05, x_scale=[1, 1, 1, 0.01])
        if sol is None or cand.cost < sol.cost:
            sol = cand

    R = Rotation.from_rotvec(sol.x[:3]).as_matrix()
    tau = float(sol.x[3])

    w_rb_i = np.stack(
        [np.interp(t_eval - tau, t_rb, omega_rb[:, k]) for k in range(3)],
        axis=1)
    pred = w_rb_i @ R.T
    cos = np.sum(pred * w_dev, axis=1) / (
        np.linalg.norm(pred, axis=1) * np.linalg.norm(w_dev, axis=1) + 1e-15)
    residual_deg = float(np.median(np.degrees(np.arccos(np.clip(cos, -1, 1)))))

    # Parameter covariance from the Jacobian: sigma^2 (J^T J)^-1
    tau_std_ms = rot_std_deg = float("nan")
    try:
        J = sol.jac
        dof = max(J.shape[0] - J.shape[1], 1)
        sigma2 = 2 * sol.cost / dof
        cov = sigma2 * np.linalg.inv(J.T @ J)
        d = np.sqrt(np.diag(cov))
        rot_std_deg = float(np.degrees(np.linalg.norm(d[:3])))
        tau_std_ms = float(d[3] * 1e3)
    except np.linalg.LinAlgError:
        pass

    return {
        "R": R,
        "tau_s": tau,
        "residual_deg": residual_deg,
        "tau_std_ms": tau_std_ms,
        "rot_std_deg": rot_std_deg,
        "n_used": n_used,
        "converged": bool(sol.success),
        "tau_init": float(tau_init),
    }


# utc_anchor_offset() (wall-clock-stamp coarse offset) removed
# 2026-08-18: unused by every script, untested, and its one real-world
# use came out ~105s off the true offset on session 1's data (see
# scripts/09_hand_eye_calibration.py's docstring) -- Aria's UTC
# timestamp is presumably stale in this lab. joint_align() does its own
# coarse-to-fine search instead and does not depend on this. Recoverable
# from git history if a wall-clock cross-check is ever wanted again on
# a device with reliable UTC.


# -- ground-truth direction (Eq. 19) ---------------------------------

def ground_truth_direction(
    glasses: PoseTrack,
    source_position_world: np.ndarray,
    R_cpf_rb: np.ndarray | None = None,
) -> np.ndarray:
    """Source direction in the glasses frame. Eq. 19 of the report.

        u_H(t) = R_HW(t) (p_s - p_H) / ||p_s - p_H||

    R_cpf_rb : rotation from the marker rigid body into CPF, from a real
        hand-eye calibration (see scripts/09_hand_eye_calibration.py,
        which drives estimate_alignment_from_angular_velocity() /
        joint_align() below). IF YOU PASS None YOU ARE ASSUMING THE
        MARKER BODY IS PERFECTLY ALIGNED WITH CPF, which it is not, and
        every reported DoA inherits that unquantified bias.
    """
    p_s = np.asarray(source_position_world, dtype=float).reshape(1, 3)
    d_world = p_s - glasses.position
    d_world /= np.linalg.norm(d_world, axis=1, keepdims=True)

    # world -> rigid body
    u_rb = np.einsum("nji,nj->ni", glasses.rotation, d_world)

    if R_cpf_rb is not None:
        u_rb = u_rb @ np.asarray(R_cpf_rb).T
    return u_rb
