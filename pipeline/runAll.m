function T = runAll(takes)
% Run runTake on every take (or the given stems) and tabulate
% results/pipeline/summary.csv.
ensurePaths();
root = repoRoot();
if nargin < 1 || isempty(takes)
    f = dir(fullfile(root,'data','rooms','*Speech*.mat'));
    takes = erase({f.name}, '.mat');
end
rows = {};
for t = 1:numel(takes)
    try
        R = runTake(takes{t});
    catch err
        fprintf('[%s] FAILED: %s\n', takes{t}, err.message); continue
    end
    c = fieldnames(R.score);
    for k = 1:numel(c)
        S = R.score.(c{k});
        rows(end+1,:) = {R.take, c{k}, R.blockLenMs, S.peakMs, S.rtErrMedian, S.rtErrMean, S.rtErrRms, S.rtErrSigned, S.drrDb, S.drrErr, ...
                         R.doaWorldMedianDeg, R.doaWorldSpreadDeg, R.doaWorldWithin15Pct}; %#ok<AGROW>
    end
end
T = cell2table(rows, 'VariableNames', {'take','condition','blockLenMs','peakMs','rtErrMedian','rtErrMean','rtErrRms','rtErrSigned','drrDb','drrErr','doaWorldMedianDeg','doaWorldStdDeg','doaWithin15Pct'});
writetable(T, fullfile(root,'results','pipeline','summary.csv'));
disp(T);
end
