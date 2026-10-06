# data/

Nothing here is committed; all of it is built.

| folder | contents | built by |
|---|---|---|
| `rooms/` | one `<room>_<speaker>Speech_<motion>.mat` per take (Deppisch-format take files), plus `<room>_reference.mat` (omni sweep at the listening position) and `<room>_arrayReference.mat` (array-side sweeps) | `build/build_take.py`, `build/build_reference.py`, `pipeline/buildArrayReference.m`, `build/build_array_sweeps.py` + `pipeline/buildSweepReference.m` |
| `anechoic/` | CHiME Aria ATFs in the paper's layout | `build/build_atf.py` |
| `cache/` | decoded VRS audio (`*_audio.npz`), keyed on the source file's size and mtime | `build/vrs_audio.py`, automatically |

The raw recordings (Aria VRS + MPS, OptiTrack CSVs, loudspeaker drive and
reference-microphone WAVs) live outside the repository at `raw_root`
(`config/paths.json`); `config/rooms.json` names every file used, relative to it.
