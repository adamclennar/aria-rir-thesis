function T = rescore(takes)
% Re-score saved pipeline results without re-running WPE, the beamformer or
% the RLS. Reads results/pipeline/<take>.mat, recomputes the metrics with the
% CURRENT scoreRir, and writes results/pipeline/summary.csv: one row per
% (take, condition), with DRR under both windows (drrWindows.m), the RMS
% error, and the per-band T20 error (t20Err_<Hz>, %) and RMS error
% (rmsErr_<Hz>, dB) the paper's Figs. 7 and 8 are drawn from.
%
% Why this exists: the direct-path check in the first version of scoreRir was
% the RIR's global maximum, which only finds the direct arrival in a dry room.
% In the reverberant room (DRR -6.3 dB) the largest sample of a 1.5 s estimate
% is a reflection at 54 ms or a patch of estimation noise near t = 0, while the
% direct path sits correctly at 8.5-9.9 ms a decibel or two down. Scoring is
% cheap; the ~30 min per take of estimation is not, so it is redone from the
% stored RIRs rather than from the audio.
ensurePaths();
root = fileparts(fileparts(mfilename('fullpath')));
resDir = fullfile(root,'results','pipeline');
if nargin < 1 || isempty(takes)
    f = dir(fullfile(resDir,'*Speech*.mat'));
    takes = erase({f.name}, '.mat');
    % baseline runs only: block-sweep results (<take>_blk<ms>.mat) are not
    % take names -- prepAriaTake would look for data/rooms/<take>_blk<ms>.mat
    takes = takes(~contains(takes, '_blk'));
end

rows = {};
for t = 1:numel(takes)
    fRes = fullfile(resDir, [takes{t} '.mat']);
    if ~exist(fRes,'file'); fprintf('[%s] no result file, skipped\n', takes{t}); continue; end
    R = load(fRes);
    P = prepAriaTake(takes{t}, 'runWpe', false, 'blockLenMs', R.blockLenMs);
    fprintf('\n[%s] blockLen %d ms | ref T20 %s | ref DRR %+.1f dB (%s) | direct path expected at %.1f ms\n', ...
        takes{t}, R.blockLenMs, mat2str(round(P.rtRef,2)), P.drrRef, P.drrRefSource, P.directPathSmp/P.fs*1e3);
    c = fieldnames(R.rir);
    for k = 1:numel(c)
        if startsWith(c{k}, 'blind'); expect = P.prePad; else; expect = P.directPathSmp; end
        S = scoreRir(P, R.rir.(c{k}), expect);
        conv = R.conv.(c{k});
        fprintf(['  %-22s direct %5.1f ms (%+5.1f dB vs global max at %6.1f ms)  ' ...
                 'T20 err med %5.1f%% mean %5.1f%% rms %5.1f%%  DRR err %+5.1f dB (paper window %+5.1f)  ' ...
                 'RMS err %4.1f dB  conv %d\n'], ...
            c{k}, S.directMs, S.directRelMaxDb, S.peakMs, S.rtErrMedian, S.rtErrMean, S.rtErrRms, ...
            S.drrErr, S.drrErrPaper, S.rmsErrMeanDb, conv.isConverged);
        rows(end+1,:) = [{takes{t}, c{k}, R.blockLenMs, S.directMs, S.directRelMaxDb, S.peakMs, ...
                         S.rtErrMedian, S.rtErrMean, S.rtErrRms, S.rtErrSigned, S.drrDb, S.drrErr, S.drrErrOmni, ...
                         S.drrDbPaper, S.drrErrPaper, S.rmsErrMeanDb, ...
                         conv.isConverged, R.doaWorldMedianDeg, R.doaWorldSpreadDeg, R.doaWorldWithin15Pct}, ...
                         num2cell(S.bandErr), num2cell(S.rmsBandErrDb)]; %#ok<AGROW>
    end
end
bands = arrayfun(@(f) sprintf('%d', round(f)), P.fAxM, 'uni', 0);
T = cell2table(rows, 'VariableNames', [{'take','condition','blockLenMs','directMs','directRelMaxDb', ...
    'globalMaxMs','rtErrMedian','rtErrMean','rtErrRms','rtErrSigned','drrDb','drrErr','drrErrOmni', ...
    'drrDbPaper','drrErrPaper','rmsErrMeanDb','converged', ...
    'doaWorldMedianDeg','doaWorldStdDeg','doaWithin15Pct'}, strcat('t20Err_', bands), strcat('rmsErr_', bands)]);
writetable(T, fullfile(resDir,'summary.csv'));
fprintf('\nwrote %s\n', fullfile(resDir,'summary.csv'));
end
