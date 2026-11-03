# Blind room-acoustic estimation from moving smart glasses

Code for the thesis *Blind Room-Acoustic Estimation from Moving Smart Glasses*. It runs the
method of Deppisch et al. (2026), *Identification and Matching of Room Acoustics With Moving
Head-Worn Microphone Arrays* (EURASIP JASMP 2026:30) -- the **baseline method** -- on 35
recordings made with Project Aria Gen 1 glasses in five rooms, plus a sixth room for the pose
study, and answers three research questions:

* **RQ1** Does the baseline method reproduce with Aria Gen 1 data, and can it run on data
  captured by the glasses alone?
* **RQ2** Where does the baseline method lose accuracy under movement and blind estimation?
* **RQ3** Can scene geometry and vision from the glasses reduce the errors identified in RQ2?

The baseline method's algorithms (WPE dereverberation, MUSIC direction finding and an MPDR
beamformer, and the RLS with its motion-aware and omnidirectional models) run **unmodified**
from the authors' code, pinned as a submodule. This repository supplies everything around
them: converting the recordings into the input format of that code, the ground truth, the
scoring, and the analyses of the three research questions.

> **Status:** every step under "Order of operations" has been run from here. The baseline
> runs (the four conditions of Deppisch et al.) cover every take: the 35 takes of the five
> main rooms and the takes of the pose-study room. The block-length sweep, the runs at the
> glasses' own block, the reference substitution and the scene geometry cover every take of
> the five main rooms; the oracle study covers the 20 takes where the wearer does not walk,
> and the moving-delay analysis the 15 walking takes.

## Two stages

```
raw recordings                       take files                          results
(Aria VRS + MPS, OptiTrack,   --->   data/rooms/*.mat             --->   RIR estimates,
 drive + reference WAVs)      build/  (the baseline method's   pipeline/  T20, DRR and
                              Python   input format)            MATLAB     tonal-balance errors
```

`pipeline/` runs on the take files that `build/` writes. The raw recordings are not in the
repository; every file under `data/` and `results/` is rebuilt from them. The two stages
share no code; `build/` needs Python and Aria's tools, `pipeline/` needs only MATLAB.

The loudspeaker drive files are not generated here: the files that were played are part of
the raw data.

## What is in here

**Stage 1 -- `build/` (raw -> `.mat`)**

| file | role |
|---|---|
| `build_take.py` | one take file in the baseline method's input format: the trim to the playback, the source signal placed so its direct path arrives 10 ms into the take at the origin pose and validated before the take is written, the pose, the noise floor, and the in-take sweep's seven-channel array IR |
| `vrs_audio.py` | 7-channel audio and device calibration from the VRS |
| `pose_mps.py` | the MPS closed-loop SLAM trajectory moved to the array and re-expressed in the take's frame W' (centred on the array at the origin pose, x forward, z up); the origin pose is the mean pose over the stillest 2 s of the in-take sweep (on the takes recorded with the speech alone, of the speech, at a moment the wearer faces the loudspeaker) |
| `source_align.py` | GCC-PHAT placement of the drive signal (150 Hz-6 kHz, 8 s windows) |
| `noise.py` | the noise floor: the quietest 2 s of the recording outside playback, and its power spectrum |
| `build_reference.py` | the omnidirectional reference: three omni sweeps at the listening position, deconvolved, aligned and averaged -> T20, source distance (direct-path delay minus the playback chain's 1.14 ms latency), rule block |
| `build_array_sweeps.py` | the pose-study room's separate array sweeps -> 7-channel IRs |
| `build_atf.py` | the MCAS far-field ATFs -> the horizontal ring of 48 directions in the baseline method's layout, with the two rear channels swapped into the device's order, (0, 1, 2, 3, 4, 6, 5) |
| `fit_optitrack_link.py`, `pose_optitrack.py`, `optitrack_io.py` | the pose study: the OptiTrack-to-glasses link (a hand-eye calibration, fitted once on a separate calibration recording) and the OptiTrack pose of each twin |
| `geometry_placement.py` | source placement from the OptiTrack-tracked geometry, for the pose-study room's two walking takes that have no still window (see `config/rooms.json`) |
| `check_takes.py` | data check: one line per take, with the source placement re-validated on the written files -> `results/build/check_takes.txt` |
| `check_atf_order.py` | data check: the ATF channel order, every one of the 5040 assignments scored by how well the ATFs' pairwise time differences of arrival match the Project Aria CAD layout over all 1674 directions, plus the head-shadow check -> `results/build/atf_channel_order.md` |
| `paths.py` | merges `config/rooms.json` with the machine-specific `config/paths.json` |

**Stage 2 -- `pipeline/` (`.mat` -> results)**

The functions you call sit in `pipeline/` itself; the helpers they use are grouped in
subfolders, which `ensurePaths` (each entry point calls it first) puts on the path.

| entry point | role |
|---|---|
| `runTake.m`, `runAll.m` | the four conditions of Deppisch et al. (informed and blind estimation, each with the motion-aware and the omnidirectional model) on one take / all takes (`run.py baseline`) |
| `rescore.m` | re-scores saved RIRs into `results/pipeline/summary.csv` (`run.py rescore`) |
| `buildArrayReference.m`, `buildSweepReference.m` | the array reference for DRR and tonal balance, in the circular-harmonic domain: the median over the room's in-take sweeps (the pose-study room: its separate array sweeps) |
| `summariseBlockSweep.m` | RQ1, block length: each take at 0.5, 1.5 and 2x the rule block (`run.py blocksweep`) and at every block the glasses would set from their room box (`run.py sabine`), beside the baseline run's 1x, re-scored into `results/blocksweep/blocksweep.csv` |
| `frontEndAnchors.m` | RQ2, substituting the reference (`run.py anchors`): the RLS given each of six references, from the pseudo-reference without dereverberation, the published pseudo-reference and the oracle-steered pseudo-reference, to the perfect pseudo-reference and the true source at matched level and at its own level -> `results/frontend/` |
| `movingReference.m` | RQ2, the moving delay (post hoc, `run.py movingref`), on the walking takes: the true source with only the changing delay or only the changing 1/r gain; the motion-aware model with its translation switched off (`rlsFit`'s `motionAwareNoTrans`, not a published condition); and the perfect and published pseudo-references with the movement removed using the glasses' own geometry (acoustic bearing and vision distance) -> `results/movingref/` |
| `evalDereverb.m`, `exportInputs.m` | the oracle study's harness: a candidate's inputs exported as WAVs (optionally with the received source, which the oracle candidates use), its output scored with the beamformer steered along the oracle bearing |
| `sourceBearing.m` | RQ3 (`run.py geometry`): the acoustic bearing, MUSIC's estimates fitted to a single fixed source position using the tracked pose, at the origin pose |
| `refT20Repeatability.m` | each band's reference T20 on every omni sweep separately, with the scoring's estimator -> `results/reliability/` |
| `ensurePaths.m`, `repoRoot.m` | put the pinned dependencies and the subfolders on the path (and check nothing shadows them); the repository root |

| helpers | role |
|---|---|
| `core/`: `prepAriaTake.m` | loads a take and sets everything up as the baseline method's `estimateRirs.m` does |
| `core/`: `rlsFit.m`, `bfEstimate.m`, `getWpe.m`, `defaultWpeParams.m` | thin wrappers calling the upstream functions with the published parameters |
| `core/`: `chEncodeArrayIr.m` | array IRs encoded into the circular-harmonic domain (the array reference) |
| `scoring/`: `scoreRir.m` | T20, DRR and tonal balance of one estimate against the room's references |
| `scoring/`: `drrWindows.m`, `revBandLevels.m` | the DRR direct-sound windows (Deppisch et al.'s, and a wide one kept in the summaries), and the reverberant part's octave-band levels |
| `scoring/`: `chDecodeToMics.m`, `micRevShape.m` | an estimate decoded to the seven microphones at the origin pose, and each microphone's reverberant spectral shape: tonal balance is scored there |
| `diagnostics/`: `evalReference.m`, `scoreRefSig.m`, `rowsToTable.m` | one reference through both RLS models, scored as a signal (coherence with the source, early and late residual) and through its RIRs |
| `diagnostics/`: `receivedDirect.m`, `removeMotion.m` | the perfect pseudo-reference: the source as received at the moving glasses, direct path only (the changing delay and 1/r gain, or either alone); and its inverse, a reference with that movement removed |
| `diagnostics/`: `addOracleDoa.m`, `acousticCentre.m` | the oracle bearing: the source straight ahead of the origin pose at the omni-measured distance (the wearer faces the loudspeaker during the in-take sweep; for the corridor's static and head-only takes, which have no sweep, `faces_source_at_reference` in `config/rooms.json`), followed through the tracked pose; bearing and range per sample from the head's acoustic centre |
| `diagnostics/`: `snapToAtf.m`, `doaErrVsOracle.m` | MUSIC's direction errors against the oracle bearing, before and after the beamformer snaps to the ATF grid |
| `diagnostics/`: `beamform.m` | the MUSIC and MPDR front end steered along a given trajectory, via the one modified upstream function in `upstream_modified/` (see its README) |
| `diagnostics/`: `musicWorldFrame.m` | MUSIC's estimates fitted to one source position using the pose: the acoustic bearing (`sourceBearing.m`); not used to steer |

**The oracle study -- `dereverb/` (`run.py oracle`)**

`oracle.py` builds the oracle study's candidates: the oracle-steered pseudo-reference with one
component of the residual removed, using the known source. In nested order: the floor echo
(removed by filtering), the early reflections 2.5-20 ms after the direct path (by subtraction,
and by filtering), and the whole fixed response; and, for the interaction of the fixed and
non-fixed parts, the non-fixed part removed (the fixed part only). A candidate that removes the
floor echo with a single-echo model is also built but not reported. Each goes through
`evalDereverb` like any dereverberator; `analysis/dereverb.py` reads them.

**Scene geometry -- `geometry/` (`run.py geometry`)**

`scene.py` takes each take's MPS semi-dense point cloud (points with depth uncertainty below
5 cm) into the take's frame and derives, from the glasses alone:

* the room box: the cloud rotated about the vertical to the alignment that makes its
  horizontal histograms sharpest, then the 1st to 99th percentile of each axis, giving the
  volume and surface area;
* the vision distance: the points within +-9 deg of the acoustic bearing (`sourceBearing.m`),
  in a height band from 0.8 m below to 0.3 m above eye level and 0.6-20 m away, counted in
  30 cm windows along the bearing; the distance is the median horizontal distance of the
  nearest window at least half as dense as the densest one.

`analysis/geometric_drr.py` turns them, with the acoustic estimate's T20, into the geometric
DRR. `fig_vision.py` draws the vision distance for one take (top view, side view, and the
density rule along the bearing).

**Analysis -- `analysis/` (results -> tables and figures)**

Reads the result CSVs, and where it needs them the take and reference files and
`config/rooms.json`, so it needs Python with numpy, scipy and matplotlib, not MATLAB. Each
script writes a summary beside its tables and figures.

| file | role |
|---|---|
| `reproduction.py` | RQ1, against Deppisch et al.: their Table 2 (DRR), Fig. 7 (T20 per octave band) and Fig. 8 (tonal balance per octave band) per movement, the per-room errors, and tonal balance on the zeroth-order channel beside the microphones (Appendix C) -> `results/reproduction/` |
| `stats.py` | shared by the scripts below: the primary measure (T_mid), the band groups beside it (low, high, 8 kHz, all), and the paired tests (bootstrap CI of the median, exact sign and Wilcoxon signed-rank tests) |
| `pose.py` | RQ1, SLAM against OptiTrack on the pose-study room's walking and head-only takes: the yaw and horizontal translation the motion-aware model uses, the tracking agreement over the whole recording, and the change in the estimates between each take's SLAM and OptiTrack twins -> `results/pose/` |
| `reliability.py` | RQ1, the repeatability of the measured T20: its sweep-to-sweep spread, the Schroeder frequency (from the SLAM volume where no tape volume is given; a guide only), the informed and blind errors per band and per voice, and the speech's energy per band -> `results/reliability/` |
| `blocksweep.py` | RQ1, block length and the glasses' block: T20 error against block length per movement, band group and octave band, with DRR and tonal balance (signed per band) checked over the same range; the range of mean absorption for which the block the glasses set from their room box (Sabine) stays inside each tolerance, beside published values for dwellings; and the baseline method run at those blocks against the rule block (Appendix C.2) -> `results/blocksweep/` |
| `decomposition.py` | RQ2, movement and blind estimation: the cost of blind estimation, the cost of movement and the choice of model, per metric, paired (Appendix D.1) -> `results/decomposition/` |
| `frontend.py` | RQ2, substituting the reference: each reference's error against the measured values and its deviation from informed, the steps between references, and the figure of both models, not walking against walking -> `results/frontend/` |
| `dereverb.py` | RQ2, the oracle study: each candidate against the oracle-steered pseudo-reference on the deviation from informed, and the share of the headroom it recovers (Appendix D.2) -> `results/dereverb/` |
| `movingref.py` | RQ2, the moving delay (post hoc): the delay or the gain alone, the role of the pose, and its removal with the glasses' geometry; and the post hoc results of that removal (RQ3): front ends compared again when walking, the omnidirectional model's walking T_mid, and the geometric DRR against the corrected omnidirectional DRR -> `results/movingref/` |
| `geometric_drr.py` | RQ3, the geometric DRR: the diffuse-field model (SLAM volume, the acoustic estimate's T20, vision distance; its constant fitted leave-one-room-out on the four rooms recorded first, frozen for the held-out room) against the acoustic DRR, and the error budget as the measured inputs are replaced by the glasses' own -> `results/geometry/` (`drr_summary.md`, `fig_drr.png`) |
| `fig_rir.py` | background figures from one room's omnidirectional reference (default: meeting room): the impulse response with its direct sound, early reflections and tail, and the DRR window beside the energy decay curve and its T20 fit, each computed as the scoring computes it -> `results/background/` |

**Configuration**: `config/rooms.json` defines the dataset -- every room, every
take, the files each uses, and every exclusion with its reason.

**Rooms**: the five main rooms are used for all three research questions: the office,
corridor, reverberant room (`reverb`), meeting room (`meeting`) and small meeting room
(`smallmeeting`). The four recorded first are used for anything that is fitted or chosen.
The small meeting room (`"held_out": true`) was recorded after the analysis was settled. It
is pooled with the others wherever nothing is fitted, and held out wherever something is:
the geometric DRR's constant and the vision distance's cone width and density ratio are
fixed on the first four rooms and applied to it unchanged. It has no SM57 recording; the
SM57 near-field copy is not used for any result.

A sixth room, L-shaped and acoustically the corridor's end (`lshape`), has OptiTrack
coverage and is used only for the pose study. Each of its tracked walking and head-only
takes is built twice from identical audio, once with the SLAM pose and once with the
OptiTrack pose (the `*Opti` twins), both trimmed to the span OptiTrack tracks completely and
expressed relative to the same origin pose. It is left out of everything else, since
including it would count the corridor's space twice.

**External code**

| path | what | licence |
|---|---|---|
| `external/room-acoustic-matching` | the baseline method's code (Deppisch et al.), submodule pinned at `76c6ea3`, unmodified | Chalmers non-commercial academic licence |
| `pipeline/upstream_modified` | one of its functions, copied and modified to steer the beamformer along a given trajectory; used only for the pseudo-references steered along the oracle bearing | Chalmers non-commercial academic licence |
| `external/ita-toolbox` | ITA toolbox (T20 with Lundeby truncation), submodule pinned at `fbf29e1` | BSD-style (RWTH Aachen) |
| `thirdparty/stft` | T. Shlomo's STFT implementation, vendored (upstream pins it as a nested submodule at `d7993ac`, which is not fetched) | MIT |

## Setup

```bash
git clone <this repo> && cd aria-rir-thesis
git submodule update --init          # NOT --recursive: upstream's own submodules are not needed
cp config/paths.example.json config/paths.json   # then set raw_root and chime_atf_pkl (the MCAS ATF file)
pip install -r requirements.txt
```

MATLAB R2025a with the Signal Processing, Statistics and Machine Learning, and
Symbolic Math toolboxes (the last is pulled in by the ITA toolbox).

## Order of operations

```bash
python build/build_atf.py                        # ATFs, in the device's channel order
python build/build_reference.py                  # omnidirectional references, all rooms
python build/build_take.py                       # every take in config/rooms.json
python build/build_array_sweeps.py lshape        # the pose-study room's array sweeps
matlab -batch "cd pipeline; buildArrayReference({'office','corridor','reverb','meeting','smallmeeting'})"
matlab -batch "cd pipeline; buildSweepReference('lshape','lshape_maleSpeech_static')"
python build/check_atf_order.py                  # data checks (Appendix A) -> results/build/
python build/check_takes.py
python run.py baseline                           # the four conditions on every take
python run.py rescore                            # every saved result -> results/pipeline/summary.csv
python analysis/reproduction.py                  # RQ1 against Deppisch et al. -> results/reproduction/
python analysis/pose.py                          # RQ1 SLAM against OptiTrack -> results/pose/
matlab -batch "cd pipeline; refT20Repeatability"      # RQ1 repeatability of the measured T20
python analysis/reliability.py                   #   -> results/reliability/
python run.py blocksweep                         # RQ1 every take at 0.5x / 1.5x / 2x the rule block
python run.py geometry                           # RQ3 acoustic bearing, room box, vision distance -> results/geometry/
python run.py sabine                             # RQ1 every take at the glasses' own blocks (needs geometry)
python analysis/blocksweep.py                    # RQ1 block length and the glasses' block -> results/blocksweep/
python analysis/decomposition.py                 # RQ2 movement and blind estimation -> results/decomposition/
python run.py anchors                            # RQ2 substituting the reference
python analysis/frontend.py                      #   -> results/frontend/
python run.py oracle                             # RQ2 the oracle study (exportInputs, dereverb/oracle.py, evalDereverb)
python analysis/dereverb.py                      #   -> results/dereverb/
python analysis/geometric_drr.py                 # RQ3 the geometric DRR -> results/geometry/
python run.py movingref                          # RQ2 the moving delay, and its removal (post hoc; needs anchors and geometry)
python analysis/movingref.py                     #   -> results/movingref/
python geometry/fig_vision.py                    # RQ3 vision-distance figure, one take -> results/geometry/
python analysis/fig_rir.py                       # background figures -> results/background/
```

`run.py` builds each stage's job list from `config/rooms.json`, skips jobs whose output
exists (so every stage resumes), and runs MATLAB jobs three at a time, one process each,
retrying a crashed job once; `--dry-run` prints the jobs, `--rooms` / `--movement` /
`--takes` narrow them. Logs go to `results/logs/<stage>/`.

## Scoring

As in Deppisch et al., T20 and DRR are computed from the zeroth-order circular-harmonic
channel of each estimate; tonal balance is the exception.

* **T20** per octave band, 125 Hz-8 kHz (ITA toolbox: Schroeder decay with Lundeby
  truncation, falling back to T15 or T10 where T20 cannot be fitted), as a relative error
  against the measured T20 of the omnidirectional reference. The primary measure, chosen
  before the results, is T_mid: the mean |error| over the 500 Hz and 1 kHz bands, ISO
  3382-1's mid-frequency reverberation time. Beside it: low (125 and 250 Hz, warmth, as in
  Beranek's bass ratio), high (2 and 4 kHz, brilliance), 8 kHz on its own, and all seven
  bands, Deppisch et al.'s measure, used wherever a result sits beside theirs. A conclusion
  drawn on T_mid is stated as one. The large errors at the band edges are not measurement
  noise: the reference T20 repeats within 5 % across the three sweeps in every band, and
  informed estimates err 4-15 % (median) in every band (`analysis/reliability.py`), so they
  belong to blind estimation.
* **DRR**, with Deppisch et al.'s estimator (fourth-order band-pass, 100 Hz-8 kHz; the direct
  peak is the earliest of the three largest peaks in the first 20 ms; the reverberant part
  runs from the end of the direct window to the Lundeby point), in dB against the array
  reference. The thesis reports Deppisch et al.'s direct-sound window, 0.5 ms before to 1 ms
  after the direct peak (`drrDbPaper`, `drrErrPaper`). The summaries also keep a wide window
  (1 ms before to 2.5 ms after; `drrDb`, `drrErr`), which shows how much of a DRR error is
  the direct sound's spread (`drrWindows.m`).
* **Tonal balance**: each estimate is decoded to the seven microphones at the origin pose,
  and the octave-band levels of its reverberant part are compared with those of the measured
  microphone responses. The mean over bands is removed, since blind estimates have an
  arbitrary gain, and |error| is averaged over the microphones. This departs from Deppisch
  et al.'s zeroth-order-channel definition: under movement, that channel ranks informed
  motion-aware estimation below blind. Their definition is kept beside it
  (`rmsOmniErrMeanDb`, Appendix C.3).
* **Statistics** (`analysis/stats.py`): every comparison is paired, by take (by room and
  talker between movements); the median of the paired differences with a bootstrap 95 % CI
  (10 000 resamples), the number of pairs in which the error fell, and exact two-sided sign
  and Wilcoxon signed-rank tests, significant at p < 0.05. Beside Deppisch et al.'s results,
  their statistic: the mean and standard deviation over takes.

## Where this deviates from Deppisch et al.

The algorithms and their parameters are unmodified apart from the block length; each
difference is documented where it is implemented.

* **Device and ATFs**: Aria Gen 1 with the far-field ATFs released with the MCAS dataset,
  measured on a head and torso simulator (1674 directions; the horizontal ring of 48 is used,
  6 and 12 deg spacing), in place of their seven-microphone prototype and its KEMAR
  measurement. The channel order is settled from the ATFs themselves (`check_atf_order.py`):
  the two rear channels are swapped to match the device.
* **Ground truth**: T20 from an omnidirectional microphone at the listening position; DRR and
  tonal balance from the array's in-take sweeps (the median over the room's takes), where
  they used a 22-microphone equatorial array for both. An omnidirectional microphone on a
  stand reads several dB higher in DRR than a head-worn array at the same spot.
* **Block length**: the rule block, the smallest multiple of 500 ms at least twice the
  longest octave-band measured T20, since the RLS returns half a block of RIR, in place of
  per-room values chosen by hand (an oracle input; see below).
* **Forgetting factor**: the code's value, 1 for both models, where their text gives 0.99
  for the omnidirectional model.
* **Pose**: the glasses' MPS closed-loop SLAM in place of motion capture.
* **Speech**: two 60 s passages of anechoic speech from the EARS dataset, one male and one
  female talker, in place of their samples. The single-microphone variant is not run.
* **Tonal balance**: scored at the microphones (above).
* **Clock rate**: the laptop and the glasses both sample at a nominal 48 kHz from
  independent clocks; no drift correction is applied (see `build/source_align.py` for why
  and for the error bound, <= 0.2 ms).
* **Source placement**: the drive signal is placed so its direct path arrives 10 ms into the
  take at the origin pose, and the take is refused unless the placed direct path lands within
  24 samples of that target, with a correlation peak at least eight times the background,
  stable across two search widths. Two of the pose-study room's walking takes are placed from
  the OptiTrack-tracked geometry instead.

## Oracle inputs

The baseline method, as run here, takes two inputs from outside the glasses, as a
reproduction under controlled conditions does:

* **Block length**: the rule block, set from the measured T20 (`analysis/blocksweep.py`
  tests setting it from the glasses instead).
* **Segmentation**: the trim to the playback, the speech span, the window the origin pose is
  chosen in, and the span the noise window is chosen outside, located by aligning the known
  drive signal with the recording. A deployed system would find the speech with voice-activity
  detection instead.

Everything else the blind conditions use comes from the glasses (array audio, MPS pose,
computed from the glasses' own cameras and IMU, and the device calibration) or is a fixed
prior for the device (the ATFs). The informed conditions also use the source signal, by
definition.

## Licence

The licence for this repository's own code is not chosen yet. The baseline method's code is
**non-commercial academic use only** (see `external/room-acoustic-matching/LICENSE`); it is
included as an unmodified submodule, not copied.
