function mics = chDecodeToMics(P, rir)
% What an estimated CH-domain RIR says each microphone should record at the
% reference pose: the inverse of chEncodeArrayIr.
%
% The RLS model is p = D * R(rotation) * T(translation) * h (fdChRlsRotTransAtf).
% At the reference pose R and T are the identity, so the microphones' transfer
% functions are simply D*h, per frequency. rir is the RLS output
% (filterLen x 2N+1 for the motion-aware model, filterLen x 1 for the omni
% model, whose array matrix is P.DOmni). Returns real microphone IRs,
% blockLen x numMics.
%
% Comparing these with the measured microphone IRs needs no encoding of the
% measurement, so it avoids the omni-channel mismatch between two differently
% regularised solvers (the RLS and chEncodeArrayIr) that the paper-style
% omni-channel comparison suffers under movement on this 7-mic array.
fftLen = P.blockLenSmp;
numPos = fftLen/2 + 1;
H = fft(rir, fftLen);
H = H(1:numPos, :);
M = zeros(numPos, P.numMics);
for ff = 1:numPos
    if size(H,2) == 1
        M(ff,:) = (P.DOmni(ff,:).' * H(ff)).';
    else
        M(ff,:) = (squeeze(P.D(ff,:,:)) * H(ff,:).').';
    end
end
mics = real(ifft([M; conj(flipud(M(2:end-1,:)))]));
end
