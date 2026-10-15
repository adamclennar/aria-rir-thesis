function S = sourceBearing(takeName)
% The glasses' own (blind) source bearing at the reference pose, for the
% geometry stage: MUSIC's per-block DoA from the baseline run, fitted to one
% source position with the tracked pose (musicWorldFrame, 'position'). The
% bearing is good to a few degrees; the fitted range is not used (it cannot be
% observed without translation), vision supplies the distance instead
% (geometry/scene.py).
%
% Also exports what the geometric DRR analysis needs from the pipeline: the
% omni reference's octave-band T20 (P.rtRef, the values scoreRir's T20 errors
% are relative to, so a blind T20 is rtRef*(1 + err/100) exactly), the
% array-reference DRR under both windows, and the acoustically measured
% source distance (reference.mat distanceM).
%
% Writes results/geometry/bearing/<take>.json.
ensurePaths();                                     % helpers live in subfolders
P = prepAriaTake(takeName, 'runWpe', false);
B = load(fullfile(P.root, 'results', 'pipeline', [takeName '.mat']), 'doaHeadRad');
[~, info] = musicWorldFrame(P, [B.doaHeadRad(:), zeros(numel(B.doaHeadRad), 1)], 'mode', 'position');
S = struct('take', takeName, 'room', P.room, 'bearing_deg', info.worldMeanDeg, 'fit_range_m', info.srcRangeM, ...
           'fit_resid_deg', info.residDeg, 'rt_ref_s', P.rtRef, 'band_hz', P.fAxM, ...
           'distance_m', P.distanceM, 'drr_ref_db', P.drrRef, 'drr_ref_paper_db', P.drrRefPaper);
d = fullfile(P.root, 'results', 'geometry', 'bearing');
if ~exist(d, 'dir'); mkdir(d); end
fid = fopen(fullfile(d, [takeName '.json']), 'w'); fwrite(fid, jsonencode(S)); fclose(fid);
end
