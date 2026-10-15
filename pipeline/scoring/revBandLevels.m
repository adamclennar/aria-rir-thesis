function L = revBandLevels(x, fs, t0Smp, endSmp, fc)
% Octave-band level (dB) of the reverberant part of an RIR -- the quantity
% behind the paper's "octave-band RMS energy" (Fig. 8), and the one its BRIR
% synthesis scales the noise tail to (synthIsoDiffRirChd): from the end of
% the paper's direct-sound window (peak +1 ms) to the Lundeby point.
%
%   t0Smp   direct peak (getDrr's second output)
%   endSmp  Lundeby point (getT20Ita's third output, as synthIsoDiffRirChd)
%   fc      octave centre frequencies, Hz
%
% Energy is summed from the FFT of the cut segment over fc/sqrt(2) to
% fc*sqrt(2), so no band filter rings across the cut into the direct sound.
% The level is absolute (arbitrary scale): blind estimates carry an unknown
% gain, so compare SHAPES -- scoreRir removes the mean over bands.
W = drrWindows();
post = W(strcmp({W.name}, 'paper')).postMs;
i0 = t0Smp + round(post/1000*fs) + 1;
L = NaN(1, numel(fc));
if ~(isfinite(t0Smp) && isfinite(endSmp)) || endSmp <= i0
    return
end
seg = x(i0:endSmp);
N = numel(seg);
P = abs(fft(seg)).^2;
P = P(1:floor(N/2)+1);
f = (0:floor(N/2)).'*fs/N;
for b = 1:numel(fc)
    in = f >= fc(b)/sqrt(2) & f < fc(b)*sqrt(2);
    if any(in); L(b) = 10*log10(sum(P(in))); end
end
end
