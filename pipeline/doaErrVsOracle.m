function S = doaErrVsOracle(P, doaUsed)
% Score a per-frame DoA sequence against the protocol bearing (addOracleDoa), both
% as estimated and as actually steered after snapping to the ATF grid.
%   doaErrDeg      median |estimate - truth|
%   steerErrDeg    median |snapped estimate - truth|: what the beamformer used
%   steerFloorDeg  median |snapped truth - truth|: the grid's own floor
%   snapChangedPct frames steered at a different ATF than the truth would be
%   failPct        frames more than 90 deg off (front/back confusions)
S = struct('doaErrDeg',NaN,'doaBiasDeg',NaN,'doaP90Deg',NaN,'failPct',NaN, ...
           'snapChangedPct',NaN,'steerErrDeg',NaN,'steerFloorDeg',NaN);
if ~P.hasOracle || isempty(doaUsed); return; end
hop = P.bf.bfHopSize;
nF = size(doaUsed,1);
nSmp = numel(P.oracleAzi);
ctr = max(min((0:nF-1)'*hop + round(P.bf.bfBlockLen/2), nSmp), 1);
oracleF = P.oracleAzi(ctr);
est = doaUsed(:,1);
[iE,sE] = snapToAtf(P, est);
[iO,sO] = snapToAtf(P, oracleF);
e = wrapToPi(est - oracleF);
S.doaErrDeg      = median(abs(e))*180/pi;
S.doaBiasDeg     = angle(mean(exp(1i*e)))*180/pi;
S.doaP90Deg      = prctile(abs(e),90)*180/pi;
S.failPct        = 100*mean(abs(e) > pi/2);
S.snapChangedPct = 100*mean(iE ~= iO);
S.steerErrDeg    = median(abs(wrapToPi(sE - oracleF)))*180/pi;
S.steerFloorDeg  = median(abs(wrapToPi(sO - oracleF)))*180/pi;
end
