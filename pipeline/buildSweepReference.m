function buildSweepReference(room, anyTake)
% Reference for a room whose only valid RIR measurement is a set of
% STANDALONE array-side sweeps (build/build_array_sweeps.py), e.g. the
% L-shape room, whose omni "Sweep_Received" files turned out to be loopbacks.
%
% Each sweep's 7-channel IR is encoded into the CH domain with the same
% array matrix the RLS uses (chEncodeArrayIr), and T20 and DRR are read off
% the order-0 channel -- exactly how buildArrayReference treats the in-take
% sweeps of the composite rooms. Only the sweeps made under the takes'
% conditions (OptiTrack markers on the glasses) form the reference; the
% no-marker sweeps are kept alongside as a check on the markers' effect.
%
% Writes both files prepAriaTake reads (the first only when no omni
% reference exists), so the rest of the pipeline is unchanged:
%   <room>_reference.mat       IRs = CH order-0 IR of the marker sweep whose
%                              T20 is closest to the median (prepAriaTake
%                              derives rtRef from it); drrDb, distanceM (tape),
%                              blockLenMsSuggested; source = 'array sweeps'
%   <room>_arrayReference.mat  median T20 / DRR over the marker sweeps, per-
%                              sweep values, and the no-marker values
ensurePaths();
root = fileparts(fileparts(mfilename('fullpath')));
cfg = jsondecode(fileread(fullfile(root,'config','rooms.json')));
S = load(fullfile(root,'data','rooms',[room '_arraySweeps.mat']));
P = prepAriaTake(anyTake, 'runWpe', false, 'blockLenMs', 3000);   % 3 s FFT: the whole IR
fs = P.fs;
names = cellstr(S.names); mk = logical(S.markers(:));
n = numel(names); rt = zeros(n,7); drr = zeros(n,1); xs = cell(n,1);
for k = 1:n
    x = chEncodeArrayIr(P, S.IRs(:,:,k));
    xs{k} = x(:,1);
    [v, fAx] = getT20Ita(xs{k}, fs, 1, [125 8000]);
    dv = getDrr(xs{k}, fs, 1, 2.5, [100 8000]);
    rt(k,:) = v(:).'; drr(k) = dv(1);
    fprintf('  %-32s markers %d  T20 %s  DRR %+6.1f dB\n', names{k}, mk(k), mat2str(round(rt(k,:),2)), drr(k));
end
rtMed = median(rt(mk,:),1); drrMed = median(drr(mk));
[~, iRep] = min(sum((log(rt(mk,:)) - log(rtMed)).^2, 2));
iMk = find(mk); iRep = iMk(iRep);
blk = 500*ceil(2*max(rtMed)*1000/500);
fprintf('[%s] markers (reference): MEDIAN T20 %s  DRR %+.1f dB (spread %.1f dB) -> block %d ms\n', ...
    room, mat2str(round(rtMed,2)), drrMed, range(drr(mk)), blk);
fprintf('[%s] no markers (check):  MEDIAN T20 %s  DRR %+.1f dB\n', room, ...
    mat2str(round(median(rt(~mk,:),1),2)), median(drr(~mk)));

src = sprintf(['CH order-0 channel of %s (glasses worn on the mark facing the loudspeaker, OptiTrack ' ...
    'markers on, as in the takes), chosen as the marker sweep whose T20 is closest to the median of %d. ' ...
    'There is NO valid omni reference for this room (its Sweep_Received files are loopbacks); RT and DRR ' ...
    'are therefore array-side. distanceM is the tape measurement. Built %s by buildSweepReference.m.'], ...
    names{iRep}, sum(mk), datestr(now,'yyyy-mm-dd'));
ref = struct('IRs', xs{iRep}, 'fs', fs, 'fBands', fAx(:).', 'rt20Bands', rt(iRep,:), ...
    'drrDb', drrMed, 'distanceM', cfg.rooms.(room).distance_m, ...
    'distanceMeasuredM', cfg.rooms.(room).distance_m, 'blockLenMsSuggested', blk, ...
    'source', 'array sweeps', 'info', src);
% Only stand in for the omni reference when there is no real one: where an
% omni on a stand was measured, T20 comes from it, as in every other room.
refFile = fullfile(root,'data','rooms',[room '_reference.mat']);
hasOmni = exist(refFile,'file') && ~isfield(load(refFile,'source'),'source');
if hasOmni
    fprintf('[%s] omni reference exists -- left alone; writing the array reference only\n', room);
else
    save(refFile, '-struct', 'ref', '-v7');
end
aR = struct('rt20Bands', rtMed, 'drrDb', drrMed, 'rt20PerTake', rt(mk,:), 'drrPerTake', drr(mk), ...
    'takes', {names(mk)}, 'rt20NoMarkers', rt(~mk,:), 'drrNoMarkers', drr(~mk), ...
    'noMarkerSweeps', {names(~mk)}, 'fBands', fAx(:).', ...
    'info', sprintf(['Array reference for the %s room from %d standalone array-side sweeps with the ' ...
    'OptiTrack markers on (median); no-marker sweeps stored as a check. Built %s by buildSweepReference.m.'], ...
    room, sum(mk), datestr(now,'yyyy-mm-dd')));
save(fullfile(root,'data','rooms',[room '_arrayReference.mat']), '-struct', 'aR', '-v7');
end
