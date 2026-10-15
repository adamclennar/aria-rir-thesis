function S = micRevShape(x, fs, fc)
% Spectral shape of each microphone's reverberant part: octave-band levels
% (revBandLevels: peak +1 ms to the Lundeby point) with each microphone's
% mean over bands removed. x is numSamples x numMics; S is numMics x numBands.
%
% The direct peak and the Lundeby point are found per microphone, as for the
% omni channel in scoreRir (getDrr's peak, getT20Ita's end point), so a
% microphone further from the source does not leak its direct sound into
% the segment.
S = NaN(size(x,2), numel(fc));
for m = 1:size(x,2)
    [~, ~, iEnd] = getT20Ita(x(:,m), fs, 1, [125 8000]);
    [~, t0] = getDrr(x(:,m), fs, 0.5, 1, [100 8000]);
    L = revBandLevels(x(:,m), fs, t0, iEnd, fc);
    S(m,:) = L - mean(L, 'omitnan');
end
end
