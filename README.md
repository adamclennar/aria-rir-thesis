# Room acoustics from moving head-worn arrays: baseline reproduction

Reproduction of the method of Deppisch et al. (2026), *Identification and
Matching of Room Acoustics With Moving Head-Worn Microphone Arrays*
(EURASIP JASMP 2026:30), on recordings made with Aria Gen 1 glasses in six
rooms. The paper's algorithms (WPE dereverberation, MUSIC + MPDR beamformer,
motion-aware and omni RLS) run **unmodified** from the authors' code, pinned as
a submodule; this repository supplies everything around them: turning the
recordings into the paper's take format, the reference measurements, and the
scoring.

> **Status:** the baseline (the paper's four conditions) has been run from
> here on every take: the five composite-protocol rooms and the L-shape
> pose-comparison takes. So have the block-length sweep on the walking takes
> of the four rooms recorded first, and the front-end anchors (RQ2). The rest
> is still being ported from the working repository (see "Not in here (yet)").

## Two stages

```
raw recordings                       take files                       results
(Aria VRS + MPS, OptiTrack,   --->   data/rooms/*.mat          --->   RIR estimates,
 drive + reference WAVs)      build/  (the paper's format)  pipeline/  T20 / DRR errors
                              Python                        MATLAB
```

`pipeline/` runs on the published take files. `build/` documents how those
files were produced from raw recordings that are not distributed: it is
provenance, not something readers can run. The two stages share no code;
`build/` needs Python and Aria's tools, `pipeline/` needs only MATLAB.

## What is in here

**Stage 1 -- `build/` (raw -> `.mat`)**

| file | role |
|---|---|
| `build_take.py` | one take file: audio trim, source-signal placement and its validation, pose in the paper's convention, noise floor, in-take sweep IRs |
| `vrs_audio.py` | 7-channel audio and device calibration from the VRS |
| `pose_mps.py` | MPS closed-loop SLAM -> array pose, reference frame W' |
| `source_align.py` | GCC-PHAT placement of the drive signal |
| `noise.py` | silence segment and noise PSD |
| `build_reference.py` | omni sweep at the listening position -> T20, DRR, distance, block length |
| `build_array_sweeps.py` | standalone array-side sweeps -> 7-channel IRs (L-shape room) |
| `build_atf.py` | CHiME Aria ATFs -> the paper's ATF layout |
| `pose_optitrack.py`, `geometry_placement.py`, `optitrack_io.py` | OptiTrack link and geometry-based placement, needed for the two L-shape walking takes that have no still window (see `config/rooms.json`) |
| `check_takes.py` | one-line-per-take sanity table |
| `paths.py` | merges `config/rooms.json` with the machine-specific `config/paths.json` |

**Stage 2 -- `pipeline/` (`.mat` -> results)**

| file | role |
|---|---|
| `runTake.m`, `runAll.m` | the paper's four conditions (informed/blind x motion-aware/omni) on one take / all takes |
| `prepAriaTake.m` | loads a take and sets everything up as `estimateRirs.m` does |
| `rlsFit.m`, `bfEstimate.m`, `getWpe.m`, `defaultWpeParams.m` | thin wrappers calling the upstream functions with the published parameters |
| `scoreRir.m`, `rescore.m` | T20, DRR and octave-band RMS scoring against the room references; `rescore` re-scores saved RIRs into `results/pipeline/summary.csv` |
| `drrWindows.m`, `revBandLevels.m` | the two DRR direct-sound windows (paper's and this repo's), and the reverberant part's octave-band levels behind the RMS error |
| `chDecodeToMics.m`, `micRevShape.m` | an estimate decoded to the 7 microphones, and each microphone's reverberant spectral shape: the tonal-balance error is scored there |
| `chEncodeArrayIr.m`, `buildArrayReference.m`, `buildSweepReference.m` | array-side references in the CH domain |
| `summariseBlockSweep.m` | block-length sweep (`run.py blocksweep`): each walking and static take at 0.5x, 1.5x and 2x the rule block, beside the baseline's 1x, re-scored into `results/blocksweep/blocksweep.csv` |
| `ensurePaths.m` | puts the pinned dependencies on the path and checks nothing shadows them |

**Stage 2, diagnostics -- `pipeline/` (where the blind pipeline loses accuracy, RQ2)**

| file | role |
|---|---|
| `frontEndAnchors.m` (`run.py anchors`) | the RLS against a ladder of pseudo-references, from the published blind front end to a perfect one: MUSIC steering; steering held at the protocol bearing, with and without WPE; the perfect front end; and the informed source at both levels -> `results/frontend/` |
| `evalReference.m`, `scoreRefSig.m`, `rowsToTable.m` | one reference through both RLS models, scored as a signal (coherence with the source, early and late residual, filter drift) and through its RIRs |
| `receivedDirect.m` | the perfect front end: the source as it reaches the moving array (range delay and 1/r) |
| `addOracleDoa.m`, `acousticCentre.m` | the protocol bearing: the source straight ahead of the reference pose (the wearer faces the loudspeaker during the in-take sweep; for the corridor's static and head-only takes, which have no sweep, `faces_source_at_reference` in `config/rooms.json`) at the omni-measured distance, followed through the tracked pose; bearing and range per sample from the head's acoustic centre |
| `snapToAtf.m`, `doaErrVsOracle.m` | DoA errors against the protocol bearing, before and after the beamformer snaps to the ATF grid |
| `musicWorldFrame.m` | MUSIC's estimates fitted to one source position using the pose: the glasses' own source bearing, kept for the vision distance (RQ3); not used to steer |
| `beamform.m`, `upstream_modified/` | the MUSIC + MPDR front end steered along a given trajectory, via the one modified upstream function (see its README) |
| `evalDereverb.m`, `exportInputs.m` | harness for a candidate dereverberator: its inputs exported as WAVs (optionally with the received source, for oracle candidates), its output scored with the beamformer steered at the protocol bearing |

**Oracle front-end candidates -- `dereverb/` (`run.py oracle`)**

`oracle.py` builds candidate pseudo-references with the source known, to measure what parts of today's reference cost accuracy: the floor echo removed (as measured, and as a simple echo model), all early reflections removed (by subtraction, and by inverse filtering), and the split into the reference's fixed part only and the perfect front end plus its non-fixed residual. They go through `evalDereverb` like any dereverberator; `analysis/dereverb.py` reads them.

**Analysis -- `analysis/` (results -> tables and figures)**

Reads only the result CSVs, so it needs Python with numpy and matplotlib, not MATLAB.

| file | role |
|---|---|
| `reproduction.py` | RQ1: the paper's Table 2 (DRR, both windows), Fig. 7 (T20 per band) and Fig. 8 (RMS per band) per movement type, plus per-room errors -> `results/reproduction/` |
| `stats.py` | shared by the scripts below: the primary T20 measure and its sensitivity band sets, and the paired tests (bootstrap CI of the median, exact sign and Wilcoxon) |
| `decomposition.py` | RQ2: what movement costs and what blind estimation costs on top, per metric, paired -> `results/decomposition/` |
| `blocksweep.py` | RQ2: T20 error against block length, walking and static takes, optionally beside the reference study's own sweep -> `results/blocksweep/` |
| `frontend.py` | RQ2: the front-end anchors, against the omni reference and as deviation from the informed estimate of the same take -> `results/frontend/` |
| `dereverb.py` | RQ2/RQ3: the oracle candidates on the front-end ladder, and the share of the headroom each recovers -> `results/dereverb/summary.md` |

**Configuration**: `config/rooms.json` defines the dataset -- every room, every
take, the files each uses, and every exclusion with its reason.

**Rooms**: the analyses use the five composite-protocol rooms (office,
corridor, reverb, meeting, small meeting room). The L-shape takes were recorded at the
end of the same corridor, opening into a larger room -- acoustically the same
space (T20 within ~5 % of the corridor's in six of seven octave bands) -- with
OptiTrack running. They exist for the pose-source comparison (the `*Opti`
takes: the same audio, placement and reference instant with OptiTrack pose
instead of SLAM), not as an additional room. Both versions of each walking
take are trimmed to the span OptiTrack tracks completely, so the comparison
is pure SLAM against pure OptiTrack.

The small meeting room (`"held_out": true`) was recorded last, after
everything else was settled. It follows the composite protocol but has no SM57
recording. It is pooled with the other four wherever nothing is fitted or
chosen (the reproduction and the RQ2 diagnostics), and held out wherever
something is: the glasses-only block-length rule, the geometry stage's fitted
constants, and the dereverberator's design. Those are fixed on the other four
rooms and then applied to it unchanged.

**External code**

| path | what | licence |
|---|---|---|
| `external/room-acoustic-matching` | Deppisch et al.'s code, submodule pinned at `76c6ea3`, unmodified | Chalmers non-commercial academic licence |
| `pipeline/upstream_modified` | one upstream function, modified to steer along a trajectory; diagnostics only | Chalmers non-commercial academic licence |
| `external/ita-toolbox` | ITA toolbox (T20 via Lundeby), submodule pinned at `fbf29e1` | BSD-style (RWTH Aachen) |
| `thirdparty/stft` | T. Shlomo's STFT, vendored (upstream's nested submodule, `d7993ac`, is not fetched) | MIT |

## Not in here (yet)

Still in the working repository, to be ported as the thesis needs them: the
controlled-contamination (dose-response) curves and blind equalisation of the
pseudo-reference (where the blind pipeline loses accuracy); block length from
the glasses alone and the SLAM/OptiTrack pose comparison (glasses-only
operation); and the scene-geometry estimates: SLAM room volume, vision
distance and geometric DRR.

Deliberately excluded: the composite drive generator, because the files
actually played (118 s) were not produced by it -- the played drive signals
are part of the raw data.

## Setup

```bash
git clone <this repo> && cd aria-rir-thesis
git submodule update --init          # NOT --recursive: upstream's own submodules are not needed
cp config/paths.example.json config/paths.json   # then set raw_root and chime_atf_pkl
pip install -r requirements.txt
```

MATLAB R2025a with the Signal Processing, Statistics and Machine Learning, and
Symbolic Math toolboxes (the last is pulled in by the ITA toolbox).

## Order of operations

```bash
python build/build_atf.py                        # ATFs
python build/build_reference.py                  # omni references, all rooms
python build/build_take.py                       # every take in config/rooms.json
python build/build_array_sweeps.py lshape        # L-shape array sweeps
matlab -batch "cd pipeline; buildArrayReference({'office','corridor','reverb','meeting','smallmeeting'})"
matlab -batch "cd pipeline; buildSweepReference('lshape','lshape_maleSpeech_static')"
python run.py baseline                           # stage 2: the paper's four conditions on every take
python run.py rescore                            # every saved result -> results/pipeline/summary.csv
python run.py blocksweep                         # walking and static takes at 0.5x / 1.5x / 2x the rule block
python analysis/reproduction.py                  # RQ1 tables and figures -> results/reproduction/
python run.py anchors                            # RQ2 front-end anchors, takes with a protocol bearing
python run.py oracle                             # RQ2/RQ3 oracle candidates (exportInputs, dereverb/oracle.py, evalDereverb)
python analysis/decomposition.py                 # RQ2 movement x estimation -> results/decomposition/
python analysis/blocksweep.py                    # RQ2 block length -> results/blocksweep/
python analysis/frontend.py                      # RQ2 front end -> results/frontend/
python analysis/dereverb.py                      # RQ2/RQ3 oracle candidates -> results/dereverb/
```

`run.py` builds each stage's job list from `config/rooms.json`, skips jobs whose output
exists (so every stage resumes), and runs MATLAB jobs three at a time, one process each,
retrying a crashed job once; `--dry-run` prints the jobs, `--rooms` / `--movement` /
`--takes` narrow them. Logs go to `results/logs/<stage>/`.

## Scoring

The paper's three metrics, T20 and DRR on the order-0 (omnidirectional) channel
and tonal balance at the microphones:

* **T20** per octave band, 125 Hz-8 kHz (ITA toolbox, Lundeby), as a relative
  error against the omni reference. The comparison with the paper uses all
  seven bands, as the paper does. Every later analysis (where accuracy is
  lost, glasses-only operation, scene geometry) has one primary T20 measure:
  the mean |relative error| in the 500 Hz and 1 kHz bands, the bands of
  ISO 3382-1's mid-frequency reverberation time (T_mid). 500 Hz-2 kHz and
  all seven bands are reported beside it as sensitivity checks. The extreme
  bands carry large errors in every condition, which dilutes the differences
  between conditions.
* **DRR**, scored under two direct-sound windows (`drrWindows.m`): the
  paper's (peak -0.5 to +1 ms; `drrDbPaper`, `drrErrPaper`) and a wider one
  (-1 to +2.5 ms; `drrDb`, `drrErr`). Compare with the paper using its window;
  `drrWindows.m` explains why the wider one is used for this data's own
  comparisons.
* **Tonal balance (octave-band RMS)**: the level of the reverberant part per
  band (the quantity the paper's BRIR synthesis matches), scored **at the
  microphones**: each estimate is decoded to what the 7 microphones should
  record at the reference pose and compared with the measured microphone IRs.
  Blind estimates have an arbitrary absolute gain, so only the spectral shape
  is scored (the mean over bands removed). The paper scores this on the omni
  CH channel instead; on this array, under movement, that comparison is
  distorted by the CH encoding (the informed estimate came out worse than the
  blind one), so it is kept only as `rmsOmniErrMeanDb`.

## Where this deviates from the paper

All forced by the data rather than chosen; each is documented where it is
implemented.

* **ATFs**: CHiME Aria set (HATS, 1674 directions), horizontal ring; the paper
  used its own KEMAR measurements of its glasses.
* **Block length**: per room, the smallest 500 ms multiple >= 2 x the longest
  octave-band T20 of the omni reference, because the RLS returns half a block
  of RIR (an oracle input; see below).
* **References**: T20 from an omni on a stand at the listening position; DRR
  and the RMS spectral shape from array-side sweeps (median over takes),
  because a free-field omni reads several dB higher than a head-worn array at
  the same spot.
* **Pose**: Aria MPS closed-loop SLAM (no OptiTrack in four of the five rooms).
* **Clock rate**: the laptop and the glasses both sample at a nominal 48 kHz
  from independent clocks; no drift correction is applied (see
  `build/source_align.py` for why and for the error bound, <= 0.2 ms).
* **Source placement**: the drive signal is placed so its direct path lands at
  10 ms at the reference pose and validated before a take is written; two
  L-shape walking takes are placed from OptiTrack-tracked geometry instead.

## Oracle inputs

The baseline takes two inputs from outside the glasses, as a reproduction
under controlled conditions does:

* **Block length**, the RLS's main hyperparameter, is set from the omni
  reference measurement (2 x its longest octave-band T20).
* **Segmentation** -- the trim to the playback, the speech span, the window
  the reference pose is chosen in, and the span the noise window is chosen
  outside -- is located by aligning the known drive signal with the
  recording.

Everything else the blind conditions use comes from the glasses: array
audio, MPS pose (computed from the glasses' own cameras and IMU) and the
device calibration, plus the CHiME ATFs, a fixed prior for this glasses
model. The informed conditions also use the source signal, by definition.
A later glasses-only stage replaces both oracle inputs, block length from
the glasses data and segmentation from speech activity and pose, so that
ground truth is used only for scoring.

## Licence

The licence for this repository's own code is not chosen yet. The upstream
code it runs is **non-commercial academic use only** (see
`external/room-acoustic-matching/LICENSE`); it is included as an unmodified
submodule, not copied.
