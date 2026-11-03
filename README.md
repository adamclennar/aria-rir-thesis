# Blind room-acoustic estimation from moving smart glasses

Code for the thesis *Blind Room-Acoustic Estimation from Moving Smart Glasses*. It runs the
room-acoustic estimation method of Deppisch et al. (2026), *Identification and Matching of Room
Acoustics With Moving Head-Worn Microphone Arrays* (the **baseline method**), on recordings from
Project Aria Gen 1 glasses, and analyses it in three parts, one for each research question:

* **RQ1**: does the method reproduce on Aria Gen 1 data, and can it run on the glasses' own SLAM
  pose and a block length set from their room map?
* **RQ2**: where does it lose accuracy under movement and blind estimation?
* **RQ3**: can the glasses' scene geometry predict the DRR more accurately than the baseline method?

The baseline method's code is included unmodified, as a pinned submodule; a modified copy of one of
its functions is used only for the RQ2 diagnostics (see [External code](#external-code)). This
repository converts the recordings into the baseline method's input format, builds the ground truth,
scores the estimates, and runs the analyses.

## Requirements

* Python 3.12 and the packages in `requirements.txt` (numpy, scipy, matplotlib, soundfile,
  projectaria-tools)
* MATLAB R2025a with the Signal Processing, Statistics and Machine Learning, and Symbolic Math
  toolboxes
* The raw recordings, which are not in this repository: the Aria VRS and MPS files, the OptiTrack
  data, the loudspeaker drive files and the measurement-microphone recordings
* The MCAS far-field ATF file (`aria_atfs.pkl`)

## Setup

```bash
git clone git@github.com:adamclennar/aria-rir-thesis.git && cd aria-rir-thesis
git submodule update --init                      # not --recursive
cp config/paths.example.json config/paths.json   # then set the two paths in it (below)
pip install -r requirements.txt
```

In `config/paths.json`, set `raw_root` to the folder holding the raw recordings and `chime_atf_pkl`
to the path of `aria_atfs.pkl`.

`run.py` starts MATLAB from `/Applications/MATLAB_R2025a.app/bin/matlab`. If MATLAB is installed
elsewhere, set the `MATLAB` environment variable to its path.

## How it fits together

```
raw recordings --build/--> data/ (take files) --pipeline/--> results/ --analysis/--> tables, figures
```

| folder | what it does |
|---|---|
| `build/` | Python: turns each recording into a take file in the baseline method's input format (audio, SLAM pose, the source signal aligned to the recording, and the noise floor); builds the omnidirectional ground truth and the ATFs; and runs two data checks |
| `pipeline/` | MATLAB: builds the ground truth at the array, runs the baseline method on each take and scores its T20, DRR and tonal balance; also runs the RQ2 diagnostics |
| `geometry/` | the room box and the source distance from the glasses' SLAM point cloud (RQ1 and RQ3) |
| `dereverb/` | the oracle study's candidate references (RQ2) |
| `analysis/` | Python: reads the results and writes each part's tables, figures and summary |
| `run.py` | runs the MATLAB stages as jobs, three at a time; finished jobs are skipped, so an interrupted stage resumes where it stopped |

`data/` and `results/` are not committed: everything in them is rebuilt from the raw recordings.

## Running it

Run these from the repository root, in this order. The `run.py` stages accept `--dry-run`,
`--rooms`, `--movement`, `--takes` and `--jobs`, and write their logs to `results/logs/<stage>/`.

```bash
# 1. take files and ground truth
python build/build_atf.py
python build/build_reference.py
python build/build_take.py
python build/build_array_sweeps.py lshape
matlab -batch "cd pipeline; buildArrayReference({'office','corridor','reverb','meeting','smallmeeting'})"
matlab -batch "cd pipeline; buildSweepReference('lshape','lshape_maleSpeech_static')"
python build/check_atf_order.py
python build/check_takes.py

# 2. the baseline method on every take, scored
python run.py baseline
python run.py rescore                                  # -> results/pipeline/summary.csv

# 3. RQ1
python analysis/reproduction.py
python analysis/pose.py
matlab -batch "cd pipeline; refT20Repeatability"
python analysis/reliability.py
python run.py blocksweep                               # 0.5, 1.5 and 2x the rule block
python run.py geometry                                 # bearing, room box, vision distance (also RQ3)
python run.py sabine                                   # the block lengths the glasses would set
python analysis/blocksweep.py

# 4. RQ2
python analysis/decomposition.py
python run.py anchors && python analysis/frontend.py
python run.py oracle && python analysis/dereverb.py
python run.py movingref && python analysis/movingref.py

# 5. RQ3 and the background figures
python analysis/geometric_drr.py
python geometry/fig_vision.py
python analysis/fig_rir.py
```

## Where each result comes from

Each output folder has a `summary.md` with the numbers quoted in the thesis (`drr_summary.md` for
the geometric DRR).

| result | script | output |
|---|---|---|
| Data checks (Appendix A) | `build/check_atf_order.py`, `build/check_takes.py` | `results/build/` |
| RQ1: against Deppisch et al.; per room and tonal balance (Appendix C.1, C.3) | `analysis/reproduction.py` | `results/reproduction/` |
| RQ1: SLAM against OptiTrack | `analysis/pose.py` | `results/pose/` |
| RQ1: repeatability of the measured T20 | `analysis/reliability.py` | `results/reliability/` |
| RQ1: block length and the glasses' block (Appendix C.2) | `analysis/blocksweep.py` | `results/blocksweep/` |
| RQ2: movement and blind estimation (Appendix D.1) | `analysis/decomposition.py` | `results/decomposition/` |
| RQ2: substituting the reference | `analysis/frontend.py` | `results/frontend/` |
| RQ2: the oracle study (Appendix D.2) | `analysis/dereverb.py` | `results/dereverb/` |
| RQ2: the moving delay | `analysis/movingref.py` | `results/movingref/` |
| RQ3: the geometric DRR | `analysis/geometric_drr.py`, `geometry/fig_vision.py` | `results/geometry/` |
| Background figures | `analysis/fig_rir.py` | `results/background/` |

## Dataset

The dataset has 35 takes in five rooms (`office`, `corridor`, `reverb`, `meeting`, `smallmeeting`),
with two talkers and three movements (static, head-only and walking). The small meeting room was
recorded last and is held out from everything that is fitted. A sixth room (`lshape`) has motion
capture and is used only for the pose study.

`config/rooms.json` defines the dataset: every room, take and file used, and every exclusion with
its reason.

## External code

| path | what | licence |
|---|---|---|
| `external/room-acoustic-matching` | the baseline method's code (Deppisch et al.), submodule at `76c6ea3`, unmodified | Chalmers non-commercial academic licence |
| `pipeline/upstream_modified` | a copy of one of its functions, changed only to accept a beamforming direction per sample; used only for the RQ2 diagnostics (see its README) | Chalmers non-commercial academic licence |
| `external/ita-toolbox` | the ITA Toolbox, used for T20; submodule at `fbf29e1` | BSD-style (RWTH Aachen) |
| `thirdparty/stft` | T. Shlomo's STFT, copied into this repository at the commit the baseline method's code pins (`d7993ac`) | MIT |

## Licence

The licence for this repository's own code has not been chosen yet. The baseline method's code is
for non-commercial academic use only (see `external/room-acoustic-matching/LICENSE`).
