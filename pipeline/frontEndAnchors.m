function T = frontEndAnchors(takeName, opts)
% Where does the blind front end lose accuracy, and how much could a better
% one recover? The RLS is run with the REAL array signals against a ladder of
% pseudo-references, from today's blind front end up to a perfect one:
%
%   published              the baseline as published: WPE + MPDR steered by
%                          MUSIC (the baseline run's RIRs, not re-run)
%   wpe_oracle_steering    WPE + MPDR steered at the oracle/protocol bearing
%                          (addOracleDoa: the source straight ahead of the
%                          reference pose, followed through the tracked pose):
%                          steering taken out of the comparison
%   nowpe_oracle_steering  as wpe_oracle_steering without WPE: what the
%                          dereverberation does, with the steering identical
%   perfect_frontend       the source as it reaches the moving array -- range
%                          delay and 1/r (receivedDirect) -- at the level of the
%                          wpe_oracle_steering reference: a PERFECT front end
%                          (perfect dereverberation, steering and beamforming).
%                          The headroom anchor.
%   informed_lvl_matched   the true source at the same level: also free of the
%                          array's motion, so perfect_frontend vs
%                          informed_lvl_matched is what movement costs even a
%                          perfect front end
%   informed               the true source at its own level (the baseline run's
%                          RIRs, not re-run), as the paper's informed condition
%
% LEVEL. The RLS regularisation is an absolute power (1e-2 x the mics' noise
% power), so a reference's level sets how hard its estimate is regularised.
% The source signal sits 31-38 dB above the pseudo-reference's direct path,
% i.e. the informed condition runs with ~10^3 x weaker regularisation than the
% blind one. The synthetic references are therefore scaled by levelGain, the
% least-squares gain of the wpe_oracle_steering reference onto
% perfect_frontend: 'a better front end at today's output level'. informed vs
% informed_lvl_matched measures the regularisation effect on its own.
%
% Steering is held at the oracle bearing rather than left to MUSIC because
% changing the dereverberation changes the signal MUSIC estimates from, which
% would move the steering too; published vs wpe_oracle_steering shows how
% much the steering itself matters.
%
% Needs the source geometry, so only takes with an oracle bearing (addOracleDoa).
% Each condition is cached in results/frontend/<take>/<label>.mat (RIRs, scores,
% and the reference for beamformed conditions), so a run resumes; delete the
% folder to start a take again. Every cached condition of the take is written
% to results/frontend/<take>_anchors.csv, one row per condition and model, with
% the reference scores (scoreRefSig), the RIR scores (scoreRir) and, for the
% beamformed conditions, the DoA error against the oracle bearing.
arguments
    takeName (1,:) char
    opts.conditions (1,:) cell = {'published','wpe_oracle_steering','nowpe_oracle_steering','perfect_frontend', ...
                                  'informed_lvl_matched','informed'}
end
P = prepAriaTake(takeName);                         % WPE from the cache
P = addOracleDoa(P);
if ~P.hasOracle
    error('frontEndAnchors:noGeometry', '%s has no oracle bearing (addOracleDoa), so no source geometry for receivedDirect', takeName);
end
outDir = fullfile(P.root, 'results', 'frontend', takeName);
if ~exist(outDir, 'dir'); mkdir(outDir); end
fBase = fullfile(P.root, 'results', 'pipeline', [takeName '.mat']);
B = load(fBase, 'rir', 'pref', 'doaHeadRad', 'blockLenMs');
if B.blockLenMs ~= P.blockLenMs
    error('frontEndAnchors:block', '%s: baseline run at %d ms, rule block now %d ms', takeName, B.blockLenMs, P.blockLenMs);
end
s = P.d.sourceSig(:,1);
b = receivedDirect(P, s);
fprintf('[%s] block %d ms; r_max %.2f m, yaw %+.0f..%+.0f deg, source range %.2f..%.2f m\n', ...
    takeName, P.blockLenMs, max(vecnorm(P.d.translationXYZCart(:,1:2),2,2)), min(P.rotYaw)*180/pi, ...
    max(P.rotYaw)*180/pi, min(P.srcRangeM), max(P.srcRangeM));

want = @(c) any(strcmp(opts.conditions, c));
nan2 = struct('doaErrDeg', NaN, 'steerErrDeg', NaN);
if want('published')
    d = doaErrVsOracle(P, [B.doaHeadRad(:), zeros(numel(B.doaHeadRad),1)]);
    cached(outDir, 'published', @() ev(P, B.pref(:), 'published', 'alignment', 'array', ...
        'rirs', struct('motionAware', B.rir.blind_motionAware, 'omni', B.rir.blind_omni), ...
        'extra', ex('music', 'wpe', 1, d)));
end
if want('wpe_oracle_steering') || want('perfect_frontend') || want('informed_lvl_matched')
    cached(outDir, 'wpe_oracle_steering', @() blind(P, P.derev, P.oracleTraj, 'wpe_oracle_steering', 'oracle', 'wpe'));
end
if want('nowpe_oracle_steering')
    cached(outDir, 'nowpe_oracle_steering', @() blind(P, P.d.arraySigs, P.oracleTraj, 'nowpe_oracle_steering', 'oracle', 'none'));
end
if want('perfect_frontend') || want('informed_lvl_matched')
    L = load(fullfile(outDir, 'wpe_oracle_steering.mat'), 'pref');
    a = levelGain(P, b, double(L.pref));
    fprintf('[%s] level: the wpe_oracle_steering reference''s direct path at %.4f x the source (%.1f dB)\n', takeName, a, 20*log10(a));
    if want('perfect_frontend')
        cached(outDir, 'perfect_frontend', @() ev(P, a*b, 'perfect_frontend', 'alignment', 'source', ...
            'carriesMotion', true, 'extra', ex('perfect', 'perfect', a, nan2)));
    end
    if want('informed_lvl_matched')
        cached(outDir, 'informed_lvl_matched', @() ev(P, a*s, 'informed_lvl_matched', 'alignment', 'source', ...
            'extra', ex('none', 'none', a, nan2)));
    end
end
if want('informed')
    cached(outDir, 'informed', @() ev(P, s, 'informed', 'alignment', 'source', ...
        'rirs', struct('motionAware', B.rir.informed_motionAware, 'omni', B.rir.informed_omni), ...
        'extra', ex('none', 'none', 1, nan2)));
end

% every condition cached for this take, not only this call's
f = dir(fullfile(outDir, '*.mat'));
rows = {};
for k = 1:numel(f)
    L = load(fullfile(outDir, f(k).name), 'rows');
    rows = [rows, L.rows]; %#ok<AGROW>
end
T = rowsToTable(rows);
T.atfMatchAziDeg = repmat(P.atfMatchAziRad*180/pi, height(T), 1);    % cross-check of the oracle bearing
writetable(T, fullfile(P.root, 'results', 'frontend', [takeName '_anchors.csv']));
fprintf('[%s] front-end anchors: %d rows written\n', takeName, height(T));
end

function e = ex(steering, dereverb, gain, d)
% steering: music | oracle bearing | perfect | none (no beamformer);
% dereverb: wpe | none | perfect; gain: level applied to the reference
% (levelGain, or 1 = native level); d: doaErrVsOracle of the steering used
e = struct('steering', string(steering), 'dereverb', string(dereverb), 'levelGain', gain, ...
    'doaErrDeg', d.doaErrDeg, 'steerErrDeg', d.steerErrDeg);
end

function [rows, rirs, pref] = blind(P, sig, traj, label, steering, dereverb)
t0 = tic;
[pref, doaUsed] = beamform(P, sig, traj);
fprintf('[%s] %s: beamformer %.1f min\n', P.takeName, label, toc(t0)/60);
[rows, rirs] = evalReference(P, pref, label, 'alignment', 'array', ...
    'extra', ex(steering, dereverb, 1, doaErrVsOracle(P, doaUsed)));
end

function [rows, rirs, pref] = ev(varargin)
% evalReference, plus the empty third output the cache expects (only the
% beamformed conditions store their reference)
[rows, rirs] = evalReference(varargin{:});
pref = [];
end

function a = levelGain(P, b, y)
% least-squares gain of the pseudo-reference y onto the received direct
% sound b, after removing the bulk lag
n = size(P.d.arraySigs,1); y = y(1:min(end,n)); y(end+1:n) = 0;
[c, l] = xcorr(y, b, round(0.05*P.fs)); [~, i] = max(abs(c));
y = circshift(y, -l(i));
a = abs((b'*y)/(b'*b));
end

function cached(outDir, label, fn)
f = fullfile(outDir, [label '.mat']);
if exist(f, 'file'); fprintf('%s: cached\n', label); return; end
[rows, rirs, pref] = fn();
rirs = structfun(@single, rirs, 'uni', 0);
pref = single(pref);
save(f, 'rows', 'rirs', 'pref', '-v7.3');
end
