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
%   drrDbPaper, drrPerTakePaper  the same under the paper's direct-sound
%                           window (drrWindows.m)
%   revBandsDb              median over takes of the reverberant part's
%                           octave-band levels (revBandLevels), each take's
%                           mean over bands removed first: the spectral shape
%                           scoreRir's RMS error is measured against
%   revBandsPerTake         every take's mean-removed levels
%   takes, fBands, fs, info
ensurePaths();
root = fileparts(fileparts(mfilename('fullpath')));
cfg = jsondecode(fileread(fullfile(root,'config','rooms.json')));
if nargin < 1 || isempty(rooms)
    rooms = fieldnames(cfg.rooms)';
end
W = drrWindows();

for r = 1:numel(rooms)
    room = rooms{r};
    f = dir(fullfile(root,'data','rooms',[room '_*Speech*.mat']));
    stems = erase({f.name},'.mat');
    rt = []; drr = []; lev = []; used = {};
    for t = 1:numel(stems)
        d = load(fullfile(root,'data','rooms',[stems{t} '.mat']),'IRs');
        if ~isfield(d,'IRs'); continue; end
        P = prepAriaTake(stems{t}, 'runWpe', false);
        x = chEncodeArrayIr(P, d.IRs);
        [v, fAx, iEnd] = getT20Ita(x(:,1), P.fs, 1, [125 8000]);
        dv = zeros(1, numel(W));
        for w = 1:numel(W)
            [dd, t0] = getDrr(x(:,1), P.fs, W(w).preMs, W(w).postMs, W(w).bandHz); dv(w) = dd(1);
        end
        L = revBandLevels(x(:,1), P.fs, t0, iEnd, fAx);
        rt(end+1,:) = v(:).'; drr(end+1,:) = dv; lev(end+1,:) = L - mean(L,'omitnan'); used{end+1} = stems{t}; %#ok<AGROW>
        fprintf('  %-32s T20 %s  DRR %+6.1f dB (paper window %+6.1f)\n', stems{t}, mat2str(round(v(:).',2)), dv(1), dv(2));
    end
    if isempty(used)
        fprintf('[%s] no takes with an IRs field, skipped\n', room); continue
    end
    drrMed = median(drr, 1);
    fprintf('[%s] %d takes | MEDIAN T20 %s  DRR %+.1f dB (mean %+.1f, spread %.1f dB) | paper window %+.1f dB\n\n', ...
        room, numel(used), mat2str(round(median(rt,1),2)), drrMed(1), mean(drr(:,1)), max(drr(:,1))-min(drr(:,1)), drrMed(2));
    info = sprintf(['Array reference for the %s room: the CH omnidirectional channel of the ' ...
        'in-take sweep IR, pooled over %d takes measured from the same mark. Median across ' ...
        'takes; per-take values in rt20PerTake/drrPerTake. Built %s by buildArrayReference.m.'], ...
        room, numel(used), datestr(now,'yyyy-mm-dd'));
    out = struct('rt20Bands',median(rt,1),'drrDb',drrMed(1),'rt20PerTake',rt,'drrPerTake',drr(:,1), ...
                 'drrDbPaper',drrMed(2),'drrPerTakePaper',drr(:,2), ...
                 'revBandsDb',median(lev,1,'omitnan'),'revBandsPerTake',lev, ...
                 'takes',{used},'fBands',fAx(:).','info',info);
    save(fullfile(root,'data','rooms',[room '_arrayReference.mat']),'-struct','out','-v7');
end
end
