# data/

Nothing here is committed; all of it is built from the raw recordings.

| folder | contents | built by |
|---|---|---|
| `rooms/` | one `<room>_<speaker>Speech_<motion>.mat` per take, in the input format of the baseline method's code, plus `<room>_reference.mat` (the omnidirectional reference: omni sweeps at the listening position) and `<room>_arrayReference.mat` (the array reference: the in-take sweeps, or the pose-study room's separate array sweeps) | `build/build_take.py`, `build/build_reference.py`, `pipeline/buildArrayReference.m`, `build/build_array_sweeps.py` + `pipeline/buildSweepReference.m` |
| `anechoic/` | the far-field ATFs released with the MCAS dataset, in the device's channel order and the baseline method's layout | `build/build_atf.py` |
| `cache/` | decoded VRS audio (`*_audio.npz`), keyed on the source file's size and mtime | `build/vrs_audio.py`, automatically |

The raw recordings (Aria VRS + MPS, OptiTrack CSVs, the loudspeaker drive files that were
played, and the reference-microphone WAVs) live outside the repository at `raw_root`
(`config/paths.json`); `config/rooms.json` names every file used, relative to it.
