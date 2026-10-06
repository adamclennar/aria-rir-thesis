"""silenceData / noisePower from the take's own quiet portion.

The paper stored 2 s of array silence and noisePower = mean over 8192-sample
blocks and all 7 channels of |FFT|^2 (checked against their files: ratio
1.05 to a plain block-averaged periodogram). The only consumer is
estimateRirs.m line 164, regulRls = 1e-2 * noisePower -- diagonal loading
of the RLS. Level within a few dB and spectral shape are all that matter.

No separate silence was recorded. build_take excludes the trimmed take
plus 1 s after the speech ends (the reverb of the last word has decayed by
then), and pick_silence takes the quietest `dur_s` window of what remains,
skipping the first second of the recording (button press). Taking the
quietest window rejects footsteps and frame handling on its own.

Where the window can come from depends on the protocol:
  * old-protocol takes: the trimmed take starts at the drive start, so the
    candidates are the lead-in before playback and the time after the
    speech (the drive file ends with 2 s of silence after the last word);
  * composite takes: the trimmed take starts 48.5 s into the drive file,
    near the end of the sweep, so the candidates are everything before
    that -- the walk-around, the clicks and the still period on the mark
    before the sweep (the sweep itself is never the quietest) -- and the
    time after the speech. The silence between the sweep and the speech
    lies inside the trimmed take and is never used.
"""
from __future__ import annotations

import numpy as np

NFFT = 8192


def pick_silence(x: np.ndarray, fs: float, exclude: tuple[int, int],
                 dur_s: float = 2.0) -> tuple[np.ndarray, int]:
    """Quietest dur_s window of multichannel x outside sample span `exclude`."""
    n, w = x.shape[0], int(dur_s * fs)
    cs = np.concatenate([[0.0], np.cumsum(np.mean(x ** 2, 1))])
    env = (cs[w:] - cs[:-w]) / w                  # mean power per window start
    ok = np.ones(len(env), bool)
    lo, hi = max(exclude[0] - w, 0), min(exclude[1], len(env))
    ok[lo:hi] = False
    ok[:int(fs)] = False                          # skip the very start (button press)
    if ok.sum() == 0:
        raise RuntimeError("no quiet window outside the drive span")
    env = np.where(ok, env, np.inf)
    i0 = int(np.argmin(env))
    return x[i0:i0 + w].copy(), i0


def noise_power(silence: np.ndarray, fs: float):
    """(fAxNoise (NFFT/2+1,), noisePower (NFFT/2+1,)) in the paper's units."""
    nb = silence.shape[0] // NFFT
    X = np.fft.rfft(silence[:nb * NFFT].reshape(nb, NFFT, -1), axis=1)
    P = (np.abs(X) ** 2).mean(axis=(0, 2))
    f = np.linspace(0, fs / 2, NFFT // 2 + 1)
    return f, P
