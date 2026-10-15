function [idx, snappedAziRad] = snapToAtf(P, aziRad)
% Which anechoic ATF does the beamformer end up steering at?
%
% musicAndBeamform does not steer at the estimated DoA: it picks the nearest
% direction on the measured ATF grid (P.atfAziRad) and steers there. On the
% 48-direction CHiME ring the gaps are 6 and 12 deg, so any DoA improvement
% finer than a grid step is invisible to the beamformer.
gridAzi = P.atfAziRad(:).';
[~, idx] = min(abs(wrapToPi(aziRad(:) - gridAzi)), [], 2);
snappedAziRad = P.atfAziRad(idx);
end
