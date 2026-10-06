function buildArrayReference(rooms)
% Per-room array reference, pooled over every take of that room.
%
% Each composite take carries its own 7-channel array IR from the sweep it
% played (the take's IRs field). All takes of a room were measured from the
% SAME mark with the SAME loudspeaker, so they are repeat measurements of one
% quantity -- pooling them is strictly better than trusting any single take.
%
% It also fixes a real sensitivity. The metrics are computed on the CH
% omnidirectional channel, which is dominated by whichever microphones face
% the source, so a take where the wearer's head sat a few degrees differently
% reads a different DRR: office_maleSpeech_static came out at -1.8 dB where
% its five siblings gave -4.2 to -4.8, because every other mic was 5-7 dB
% quieter relative to mic 0 in that take. The MEDIAN over takes is reported
% as the reference for that reason; the mean and the per-take values are
% stored alongside so the spread stays visible.
%
% Writes data/rooms/<room>_arrayReference.mat:
%   rt20Bands, drrDb        median over takes -- use these
%   rt20PerTake, drrPerTake  every take, for inspection
%   takes, fBands, fs, info
ensurePaths();
root = fileparts(fileparts(mfilename('fullpath')));
cfg = jsondecode(fileread(fullfile(root,'config','rooms.json')));
if nargin < 1 || isempty(rooms)
    rooms = fieldnames(cfg.rooms)';
end

for r = 1:numel(rooms)
    room = rooms{r};
    f = dir(fullfile(root,'data','rooms',[room '_*Speech*.mat']));
    stems = erase({f.name},'.mat');
    rt = []; drr = []; used = {};
    for t = 1:numel(stems)
        d = load(fullfile(root,'data','rooms',[stems{t} '.mat']),'IRs');
        if ~isfield(d,'IRs'); continue; end
        P = prepAriaTake(stems{t}, 'runWpe', false);
        x = chEncodeArrayIr(P, d.IRs);
        [v, fAx] = getT20Ita(x(:,1), P.fs, 1, [125 8000]);
        dv = getDrr(x(:,1), P.fs, 1, 2.5, [100 8000]);
        rt(end+1,:) = v(:).'; drr(end+1,1) = dv(1); used{end+1} = stems{t}; %#ok<AGROW>
        fprintf('  %-32s T20 %s  DRR %+6.1f dB\n', stems{t}, mat2str(round(v(:).',2)), dv(1));
    end
    if isempty(used)
        fprintf('[%s] no takes with an IRs field, skipped\n', room); continue
    end
    rtMed = median(rt,1); drrMed = median(drr);
    fprintf('[%s] %d takes | MEDIAN T20 %s  DRR %+.1f dB (mean %+.1f, spread %.1f dB)\n\n', ...
        room, numel(used), mat2str(round(rtMed,2)), drrMed, mean(drr), max(drr)-min(drr));
    info = sprintf(['Array reference for the %s room: the CH omnidirectional channel of the ' ...
        'in-take sweep IR, pooled over %d takes measured from the same mark. Median across ' ...
        'takes; per-take values in rt20PerTake/drrPerTake. Built %s by buildArrayReference.m.'], ...
        room, numel(used), datestr(now,'yyyy-mm-dd'));
    save(fullfile(root,'data','rooms',[room '_arrayReference.mat']), ...
        'rtMed','drrMed','rt','drr','used','fAx','info','-v7');
    % friendlier field names
    S = load(fullfile(root,'data','rooms',[room '_arrayReference.mat']));
    out = struct('rt20Bands',S.rtMed,'drrDb',S.drrMed,'rt20PerTake',S.rt,'drrPerTake',S.drr, ...
                 'takes',{S.used},'fBands',S.fAx(:).','info',S.info);
    save(fullfile(root,'data','rooms',[room '_arrayReference.mat']),'-struct','out','-v7');
end
end
