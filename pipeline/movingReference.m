function T = movingReference(takeName, opts)
% Does the array's motion count twice when the reference already carries it?
%
% When the wearer walks, the front-end anchors (frontEndAnchors) give the
% motion-aware model T_mid estimates ~11 pp apart (median deviation) with the
% perfect front end -- the source as received at the moving array, range delay
% and 1/r -- and with the dry source at the same level, while two dry
% references that differ only in level agree to ~2 pp. Any reference built
% from the moving array's signals carries that motion; the motion-aware RLS
% also translates the field with the tracked pose (fdChRlsRotTransAtf: one mean
% rotation and translation per block). Two tests, the RLS against the real
% array signals as in frontEndAnchors, every synthetic reference at the
% anchors' level (levelGain):
%
% 1. Which part of the motion matters, and can the glasses remove it?
%   received_delay_only       the source with the range delay only, no 1/r
%   received_gain_only        the source with the 1/r gain only, no delay
%   perfect_frontend_demoved  the perfect front end with the motion removed
%                             (removeMotion) from the GLASSES' OWN source
%                             position: their acoustic bearing
%                             (sourceBearing.m) at the vision distance
%                             (geometry/scene.py), and SLAM. With the oracle
%                             geometry this is the dry source by construction,
%                             so it is not run; the residual is printed instead
%   published_demoved         the published pseudo-reference with the same
%                             correction: the practical version
%   all four with both models.
% 2. Does the model's translation count it a second time? The motion-aware
%    model with the translation switched off and the rotation kept (rlsFit
%    'motionAwareNoTrans') on the anchors' perfect_frontend,
%    informed_lvl_matched and published references
%    (perfect_frontend_notrans, informed_lvl_matched_notrans,
%    published_notrans; the rows keep the anchors' labels, with the model
%    motionAwareNoTrans). With the anchors' runs (translation on) these make
%    the 2 x 2: reference carries the motion x model translates.
%
% Needs the take's anchors (the wpe_oracle_steering reference sets the level)
% and its geometry (run.py geometry: bearing and vision distance). Each
% condition is cached in results/movingref/<take>/<condition>.mat, so a run
% resumes; every cached condition's rows go to
% results/movingref/<take>_movingref.csv (opts.outDir: both into that folder
% instead, for a trial run).
arguments
    takeName (1,:) char
    opts.conditions (1,:) cell = {'received_delay_only','received_gain_only','perfect_frontend_demoved', ...
                                  'published_demoved','perfect_frontend_notrans', ...
                                  'informed_lvl_matched_notrans','published_notrans'}
    opts.models (1,:) cell = {'motionAware','omni'}     % for test 1
    opts.outDir (1,:) char = ''
end
ensurePaths();
P = prepAriaTake(takeName, 'runWpe', false);
P = addOracleDoa(P);
if ~P.hasOracle
    error('movingReference:noGeometry', '%s has no oracle bearing (addOracleDoa)', takeName);
end
outDir = opts.outDir;
if isempty(outDir); outDir = fullfile(P.root, 'results', 'movingref', takeName); end
if ~exist(outDir, 'dir'); mkdir(outDir); end
fW = fullfile(P.root, 'results', 'frontend', takeName, 'wpe_oracle_steering.mat');
if ~exist(fW, 'file')
    error('movingReference:anchors', '%s: no wpe_oracle_steering reference; run the front-end anchors first', takeName);
end
W = load(fW, 'pref');
B = load(fullfile(P.root, 'results', 'pipeline', [takeName '.mat']), 'pref', 'blockLenMs');
if B.blockLenMs ~= P.blockLenMs
    error('movingReference:block', '%s: baseline run at %d ms, rule block now %d ms', takeName, B.blockLenMs, P.blockLenMs);
end

s = P.d.sourceSig(:,1);
b = receivedDirect(P, s);
a = levelGain(P, b, double(W.pref));
srcO = P.srcPos(1:2);
srcG = glassesSource(P);
[pc, d] = acousticCentre(P);
rO = vecnorm(srcO - pc, 2, 2);  rG = vecnorm(srcG - pc, 2, 2);
dO = (rO - norm(srcO - [-d 0]))/343*1e3;  dG = (rG - norm(srcG - [-d 0]))/343*1e3;
% how well the motion can be taken out, written into every row: the glasses' delay error against the
% oracle's, and what is left of the perfect front end with the motion removed, re the dry source
G = struct('rangeChangeMaxMs', max(abs(dO)), 'glassesDelayErrRmsMs', rms(dG - dO), ...
    'demovedResidOracleDb', residualDb(removeMotion(P, b, srcO), s), ...
    'demovedResidGlassesDb', residualDb(removeMotion(P, b, srcG), s));
fprintf(['[%s] block %d ms, level %.4f (%.1f dB) | source: oracle (%.2f, %.2f) m, glasses (%.2f, %.2f) m | ' ...
    'range change %+.2f..%+.2f ms, glasses minus oracle %.3f ms rms (max %.3f)\n'], takeName, P.blockLenMs, a, ...
    20*log10(a), srcO, srcG, min(dO), max(dO), G.glassesDelayErrRmsMs, max(abs(dG - dO)));
fprintf('[%s] perfect front end with the motion removed, re the dry source: oracle %.1f dB, glasses %.1f dB\n', ...
    takeName, G.demovedResidOracleDb, G.demovedResidGlassesDb);
ex = @(signal, motion, removed, gain) extra(signal, motion, removed, gain, G);

want = @(c) any(strcmp(opts.conditions, c));
nt = {'motionAwareNoTrans'};
go = @(cond, ref, label, align, motion, models, x) cached(outDir, cond, @() evalReference(P, ref, label, ...
    'alignment', align, 'carriesMotion', motion, 'models', models, 'extra', x));
if want('received_delay_only')
    go('received_delay_only', a*receivedDirect(P, s, P.distanceM, 'parts', 'delay'), 'received_delay_only', ...
        'source', true, opts.models, ex('source', 'delay', 'none', a));
end
if want('received_gain_only')
    go('received_gain_only', a*receivedDirect(P, s, P.distanceM, 'parts', 'gain'), 'received_gain_only', ...
        'source', false, opts.models, ex('source', 'gain', 'none', a));
end
if want('perfect_frontend_demoved')
    go('perfect_frontend_demoved', removeMotion(P, a*b, srcG), 'perfect_frontend_demoved', ...
        'source', false, opts.models, ex('source', 'both', 'glasses', a));
end
if want('published_demoved')
    go('published_demoved', removeMotion(P, double(B.pref(:)), srcG), 'published_demoved', ...
        'array', false, opts.models, ex('published', 'both', 'glasses', 1));
end
if want('perfect_frontend_notrans')
    go('perfect_frontend_notrans', a*b, 'perfect_frontend', 'source', true, nt, ex('source', 'both', 'none', a));
end
if want('informed_lvl_matched_notrans')
    go('informed_lvl_matched_notrans', a*s, 'informed_lvl_matched', 'source', false, nt, ex('source', 'none', 'none', a));
end
if want('published_notrans')
    go('published_notrans', double(B.pref(:)), 'published', 'array', true, nt, ex('published', 'both', 'none', 1));
end

% every condition cached for this take, not only this call's
f = dir(fullfile(outDir, '*.mat'));
rows = {};
for k = 1:numel(f)
    L = load(fullfile(outDir, f(k).name), 'rows');
    rows = [rows, L.rows]; %#ok<AGROW>
end
T = rowsToTable(rows);
if isempty(opts.outDir); fCsv = fullfile(P.root, 'results', 'movingref', [takeName '_movingref.csv']);
else; fCsv = fullfile(outDir, [takeName '_movingref.csv']); end
writetable(T, fCsv);
fprintf('[%s] moving reference: %d rows written\n', takeName, height(T));
end

function e = extra(signal, motion, removed, gain, G)
% signal: the signal the reference is built from (source | published);
% motion: the array motion it carries (both | delay | gain | none) before any
% removal; removed: whose geometry took it out again (none | glasses);
% gain: level applied (levelGain, or 1 = native); G: the take's motion-removal
% checks, the same in every row
e = struct('signal', string(signal), 'motion', string(motion), 'removed', string(removed), 'levelGain', gain);
for f = fieldnames(G)'
    e.(f{1}) = G.(f{1});
end
end

function xy = glassesSource(P)
% The glasses' own source position in W': their acoustic bearing at the
% reference pose (sourceBearing.m) at the vision distance (geometry/scene.py),
% both measured from the head's acoustic centre there
J = jsondecode(fileread(fullfile(P.root, 'results', 'geometry', 'bearing', [P.takeName '.json'])));
S = readtable(fullfile(P.root, 'results', 'geometry', 'scene.csv'), 'TextType', 'string');
r = S.r_vision_m(S.take == P.takeName);
if isempty(r) || ~isfinite(r(1))
    error('movingReference:vision', '%s: no vision distance in results/geometry/scene.csv (run.py geometry)', P.takeName);
end
[~, d] = acousticCentre(P);
xy = [-d 0] + r(1)*[cosd(J.bearing_deg) sind(J.bearing_deg)];
end

function v = residualDb(y, s)
% energy of y - s re s, over the samples both cover
n = min(numel(y), numel(s));
v = 10*log10(sum((y(1:n) - s(1:n)).^2)/sum(s(1:n).^2));
end

function a = levelGain(P, b, y)
% as frontEndAnchors: least-squares gain of the pseudo-reference y onto the
% received direct sound b, after removing the bulk lag
n = size(P.d.arraySigs,1); y = y(1:min(end,n)); y(end+1:n) = 0;
[c, l] = xcorr(y, b, round(0.05*P.fs)); [~, i] = max(abs(c));
y = circshift(y, -l(i));
a = abs((b'*y)/(b'*b));
end

function cached(outDir, cond, fn)
f = fullfile(outDir, [cond '.mat']);
if exist(f, 'file'); fprintf('%s: cached\n', cond); return; end
[rows, rirs] = fn();
rirs = structfun(@single, rirs, 'uni', 0);
save(f, 'rows', 'rirs', '-v7.3');
end
