function T = summariseBlockSweep(takes)
% Score every block-length result of the given walking takes - the baseline
% (results/pipeline/<take>.mat, the room's rule block) and the sweep
% (<take>_blk<ms>.mat, from run.py blocksweep) - against the room
% references, and write one row per (take, block length, condition) to
% results/blocksweep/blocksweep.csv: the summary T20 errors (mean absolute,
% median, rms), every band's T20 and error, the DRR error under both
% windows (drrWindows.m) and the octave-band RMS error.
%
% Everything is re-scored from the saved RIRs rather than read from the run
% logs, so a reference rebuilt after a run is picked up, and the numbers do
% not exist only in a log file.
ensurePaths();
root = fileparts(fileparts(mfilename('fullpath')));
resDir = fullfile(root, 'results', 'pipeline');
if nargin < 1 || isempty(takes)
    f = dir(fullfile(resDir, '*_fullMove*_blk*.mat'));
    takes = unique(regexprep({f.name}, '_blk\d+\.mat$', ''));
end
rows = {};
for t = 1:numel(takes)
    room = regexp(takes{t}, '^[a-z]+', 'match', 'once');
    refS = load(fullfile(root, 'data', 'rooms', [room '_reference.mat']), 'blockLenMsSuggested');
    rule = double(refS.blockLenMsSuggested);
    files = [{[takes{t} '.mat']}, {dir(fullfile(resDir, [takes{t} '_blk*.mat'])).name}];
    for k = 1:numel(files)
        fp = fullfile(resDir, files{k});
        if ~exist(fp, 'file'), fprintf('[%s] missing %s, skipped\n', takes{t}, files{k}); continue, end
        R = load(fp);
        P = prepAriaTake(takes{t}, 'runWpe', false, 'blockLenMs', R.blockLenMs);
        c = fieldnames(R.rir);
        for m = 1:numel(c)
            if startsWith(c{m}, 'blind'); e = P.prePad; else; e = P.directPathSmp; end
            S = scoreRir(P, R.rir.(c{m}), e);
            rows(end+1, :) = [{takes{t}, room, R.blockLenMs, R.blockLenMs/rule, c{m}, ...
                S.rtErrMean, S.rtErrMedian, S.rtErrRms, S.drrDb, S.drrErr, ...
                S.drrDbPaper, S.drrErrPaper, S.rmsErrMeanDb}, ...
                num2cell(S.rt), num2cell(S.rtRef), num2cell(S.bandErr), num2cell(S.rmsBandErrDb)]; %#ok<AGROW>
        end
        fprintf('[%s] %5d ms (%.2fx rule) scored\n', takes{t}, R.blockLenMs, R.blockLenMs/rule);
    end
end
bands = arrayfun(@(f) sprintf('%d', round(f)), P.fAxM, 'uni', 0);
T = cell2table(rows, 'VariableNames', [{'take','room','blockLenMs','blockFactor','condition', ...
    'rtErrMean','rtErrMedian','rtErrRms','drrDb','drrErr','drrDbPaper','drrErrPaper','rmsErrMeanDb'}, ...
    strcat('T20_', bands), strcat('ref_', bands), strcat('err_', bands), strcat('rmsErr_', bands)]);
outDir = fullfile(root, 'results', 'blocksweep');
if ~exist(outDir, 'dir'); mkdir(outDir); end
writetable(T, fullfile(outDir, 'blocksweep.csv'));
fprintf('\nwrote %s (%d rows)\n', fullfile(outDir, 'blocksweep.csv'), height(T));
end
