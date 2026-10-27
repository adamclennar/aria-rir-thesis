function refT20Repeatability(rooms)
% How well each octave band's reference T20 is defined: the scoring's own
% estimator (getT20Ita, as prepAriaTake applies it to the mean of the omni
% sweeps) run on each sweep separately. The sweep-to-sweep spread is an upper
% bound on the uncertainty of the mean the scoring uses.
%
% Unlike the scoring, missing values are NOT interpolated: a NaN means T20,
% T15 and T10 all failed in that band.
%
% Writes results/reliability/ref_t20_per_sweep.csv: room, sweep (0 = the mean
% IR the scoring uses, 1..S = each sweep), T20 per octave band (s).
arguments
    rooms cell = {'office', 'corridor', 'reverb', 'meeting', 'smallmeeting'}
end
ensurePaths();                                     % helpers live in subfolders
root = repoRoot();
out = fullfile(root, 'results', 'reliability');
if ~exist(out, 'dir'); mkdir(out); end
fid = fopen(fullfile(out, 'ref_t20_per_sweep.csv'), 'w');
header = false;
for k = 1:numel(rooms)
    R = load(fullfile(root, 'data', 'rooms', [rooms{k} '_reference.mat']), 'IRs', 'IRsPerSweep', 'fs');
    X = [R.IRs(:, 1), R.IRsPerSweep];
    for s = 1:size(X, 2)
        [t, f] = getT20Ita(X(:, s), R.fs, 1, [125 8000], false);
        if ~header
            fprintf(fid, 'room,sweep%s\n', sprintf(',t20_%d', round(f)));
            header = true;
        end
        fprintf(fid, '%s,%d%s\n', rooms{k}, s - 1, sprintf(',%.4f', t));
    end
    fprintf('[%s] %d sweeps\n', rooms{k}, size(X, 2) - 1);
end
fclose(fid);
end
