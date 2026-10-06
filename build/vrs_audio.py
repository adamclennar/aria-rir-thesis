"""Aria VRS -> 7-channel audio, plus the device->CPF calibration transform.

Format facts (verified on Gen 1 recordings with projectaria_tools 2.1.4):
  - audio chunks are flat interleaved int64 holding 32-bit values; full scale
    is 2^31-1
  - capture_timestamps_ns has one entry per sample frame, on the DEVICE clock
    - which is the same clock MPS trajectories are stamped in, so no sync step exists
    between audio and pose
  - channel k of the audio IS mic k of the device calibration

Decoding a VRS takes minutes, so the result is cached as .npz next to the
build cache to be reused on later runs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

FULL_SCALE = 2 ** 31 - 1


def load_audio(vrs_path: str | Path, cache_dir: str | Path | None = None):
    """Return (samples (N,7) float64 in [-1,1], timestamps_ns (N,) int64, fs)."""
    vrs_path = Path(vrs_path)
    cache = None
    if cache_dir is not None:
        cache = Path(cache_dir) / (vrs_path.stem + "_audio.npz")
        # Keyed on the SOURCE FILE's size and mtime as well as its name: a cache
        # keyed on the name alone serves the previous file's audio after a
        # rename, which once made two different takes look like one recording.
        st = vrs_path.stat()
        if cache.exists():
            z = np.load(cache)
            if ("src_size" in z and int(z["src_size"]) == st.st_size
                    and abs(float(z["src_mtime"]) - st.st_mtime) < 1):
                return z["samples"], z["timestamps_ns"], float(z["fs"])
            print(f"  cache for {vrs_path.name} is stale (source file changed), re-decoding")

    from projectaria_tools.core import data_provider

    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    if provider is None:
        raise RuntimeError(f"could not open {vrs_path}")
    sid = provider.get_stream_id_from_label("mic")
    if sid is None:
        raise RuntimeError(f"{vrs_path.name}: no microphone stream")
    cfg = provider.get_audio_configuration(sid)
    n_ch, fs = int(cfg.num_channels), float(cfg.sample_rate)

    chunks, stamps, muted = [], [], 0
    for i in range(provider.get_num_data(sid)):
        data, rec = provider.get_audio_data_by_index(sid, i)
        block = np.asarray(data.data)
        chunks.append(block.reshape(-1, n_ch) if block.ndim == 1 else block)
        stamps.append(np.asarray(rec.capture_timestamps_ns))
        muted += int(getattr(rec, "audio_muted", 0) != 0)
    if muted:
        print(f"  WARNING: {muted} chunks flagged audio_muted")

    samples = np.concatenate(chunks).astype(np.float64) / FULL_SCALE
    ts = np.concatenate(stamps).astype(np.int64)
    if ts.size != samples.shape[0]:
        raise RuntimeError("timestamp/sample count mismatch - interleaving misread")

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, samples=samples, timestamps_ns=ts, fs=fs,
                 src_size=st.st_size, src_mtime=st.st_mtime)
    return samples, ts, fs


def device_to_cpf(vrs_path: str | Path):
    """(R_device_cpf (3,3), t_device_cpf (3,)) from the recording's own
    factory calibration: p_device = R @ p_cpf + t.

    MPS reports the DEVICE frame; the ATFs and mic geometry live in a frame
    derived from CPF, so this is the link between the two.
    """
    from projectaria_tools.core import data_provider

    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    T = provider.get_device_calibration().get_transform_device_cpf()
    return (np.asarray(T.rotation().to_matrix(), dtype=float),
            np.asarray(T.translation(), dtype=float).reshape(3))
