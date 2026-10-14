function [pref, doaUsed] = beamform(P, sig, traj)
% The paper's MUSIC + MPDR front end without the RLS: the pseudo-reference
% from N x M array signals, steered by MUSIC (traj = [], as published) or
% along a per-sample DoA trajectory (N x 2 [azi ele], head frame), e.g. the
% protocol bearing (addOracleDoa).
% A trajectory needs the modified beamformer in upstream_modified/; with
% traj = [] its output is the upstream function's.
%
%   pref     N x 1 pseudo-reference, sample-aligned with sig
%   doaUsed  the DoA each beamformer block steered at, before the beamformer
%            snaps it to the ATF grid (snapToAtf)
arguments
    P struct
    sig double
    traj = []
end
ensurePaths();
[sb, doaUsed] = musicAndBeamformTimeDomainAdaptiveTraj(sig, P.bf.bfBlockLen, P.bf.bfHopSize, ...
    P.bf.doaObservationLenMs, P.atfsBf, P.atfGrid, P.bf, traj, false);
pref = sb(:);
n = size(sig,1);
pref = pref(1:min(end,n)); pref(end+1:n) = 0;
end
