function T = evalDereverb(takeName, candidate, opts)
% Score one dereverberation candidate on one take, with the beamformer
% steered at the protocol bearing (addOracleDoa) so only the dereverberator
% differs between candidates: left to MUSIC, the steering would move with
% whatever signal the candidate hands it. Takes with a protocol bearing only.
%
%   candidate  'wpe'   the published WPE (the baseline)
%              'none'  no dereverberation
%              a .wav or .mat (variable y) file:
%                N x 7  PRE-beamformer: replaces WPE; beamformed here
%                N x 1  POST-beamformer: replaces the pseudo-reference itself
%              Files must be sample-aligned with the take: same length and
%              start as exportInputs' WAVs. A post candidate's bulk delay
%              against <take>_ref_wpe.wav is measured and REPORTED; it is
%              removed only if opts.realign (a fixed delay is harmless to
%              the RLS up to a few ms, but a large one eats the RIR window).
%
%   opts.inputGain  LEVEL. The RLS is not gain-invariant: its regularisation
%              is an absolute power (1e-2 x the mics' noise power), so a
%              reference 11.6x louder (exportInputs peak-normalises) sees
%              134x weaker regularisation -- feeding the exported WPE
%              reference straight back in moved motion-aware T20 error from
%              21.6% to 38.2% with the reference scores identical. File
%              candidates are therefore divided by the gain exportInputs
%              applied to the signal they were made from:
%                'auto'      the array gain for 7 ch, the ref_wpe gain for 1 ch
%                'ref_none'  for a post-filter run on <take>_ref_none.wav
%                a number    divide by it (1 = the file is at natural level)
%              A dereverberator that renormalises its own output level must
%              be undone the same way, or it is scored partly on its gain.
%
% Scores (evalReference): refCoh / refCohComp / per-octave coherence /
% refFiltVarDb / early and late residual for the reference; T20, DRR (both
% windows) and tonal-balance error for both RLS models. The 'wpe' and 'none'
% candidates are frontEndAnchors' wpe_oracle_steering and
% nowpe_oracle_steering, so a candidate is read against those and against
% the take's perfect_frontend.
% Writes results/dereverb/eval/<take>_<label>.{mat,csv}; reruns are cached.
arguments
    takeName (1,:) char
    candidate (1,:) char
    opts.label (1,:) char = ''
    opts.realign (1,1) logical = false
    opts.inputGain = 'auto'
    opts.force (1,1) logical = false   % ignore a cached result and re-run
end
ensurePaths();                                     % helpers live in subfolders
[~, base, ext] = fileparts(candidate);
label = opts.label;
if isempty(label)
    if any(strcmp(candidate, {'wpe','none'})); label = candidate; else; label = erase(base, [takeName '_']); end
end
% WPE is needed for the baseline and, for a post candidate, as the reference
% its delay is measured against; it is cached for every built take
P = prepAriaTake(takeName, 'runWpe', ~strcmp(candidate, 'none'));
P = addOracleDoa(P);
if ~P.hasOracle
    error('evalDereverb:noGeometry', '%s has no protocol bearing (addOracleDoa) to steer at', takeName);
end
outDir = fullfile(P.root, 'results', 'dereverb', 'eval');
if ~exist(outDir, 'dir'); mkdir(outDir); end
fOut = fullfile(outDir, sprintf('%s_%s', takeName, label));
if exist([fOut '.mat'], 'file') && ~opts.force
    L = load([fOut '.mat'], 'T'); T = L.T; disp(T(:, {'label','model','refCohEff','refEarlyDb','rtErrMean','drrErr'}));
    return
end

traj = P.oracleTraj;
lagSmp = NaN;
switch candidate
    case 'wpe';  pref = beamform(P, P.derev, traj);       stage = "pre";
    case 'none'; pref = beamform(P, P.d.arraySigs, traj); stage = "pre";
    otherwise
        y = readCandidate(candidate, ext);
        n = size(P.d.arraySigs, 1);
        if abs(size(y,1) - n) > round(0.01*P.fs)
            error('evalDereverb:length', '%s has %d samples, the take %d: not sample-aligned', ...
                candidate, size(y,1), n);
        end
        y = y(1:min(end,n), :); y(end+1:n, :) = 0;
        g = exportGain(P, opts.inputGain, size(y,2));
        y = y/g;
        fprintf('[%s] %s: divided by the export gain %.3f\n', takeName, label, g);
        if size(y,2) == size(P.d.arraySigs, 2)
            pref = beamform(P, y, traj); stage = "pre";
        elseif size(y,2) == 1
            stage = "post";
            ref0 = beamform(P, P.derev, traj);
            [c, lags] = xcorr(y, ref0, round(0.05*P.fs), 'normalized');
            [~, ix] = max(abs(c)); lagSmp = lags(ix);
            fprintf('[%s] %s: bulk delay vs the WPE reference %d samples (%.2f ms)\n', ...
                takeName, label, lagSmp, lagSmp/P.fs*1e3);
            if opts.realign && lagSmp ~= 0
                y = circshift(y, -lagSmp);
                if lagSmp > 0; y(end-lagSmp+1:end) = 0; else; y(1:-lagSmp) = 0; end
            end
            pref = y;
        else
            error('evalDereverb:channels', '%s has %d channels: expected %d (pre) or 1 (post)', ...
                candidate, size(y,2), size(P.d.arraySigs,2));
        end
end
[rows, rirs] = evalReference(P, pref, label, 'alignment', 'array', ...
    'extra', struct('stage', stage, 'candidate', string(candidate), 'lagSmp', lagSmp));
T = rowsToTable(rows);
writetable(T, [fOut '.csv']);
pref = single(pref);
rirs = structfun(@single, rirs, 'uni', 0);   % kept so metrics can be re-scored without the RLS
save([fOut '.mat'], 'T', 'pref', 'rirs', '-v7.3');
end

function g = exportGain(P, spec, nCh)
if isnumeric(spec); g = spec; return; end
G = jsondecode(fileread(fullfile(P.root, 'results', 'dereverb', 'inputs', [P.takeName '_gains.json'])));
switch spec
    case 'auto'
        if nCh > 1; g = G.array; else; g = G.ref_wpe; end
    otherwise
        g = G.(spec);
end
end

function y = readCandidate(f, ext)
switch lower(ext)
    case '.wav'; y = audioread(f);
    case '.mat'; L = load(f, 'y'); y = L.y;
    otherwise; error('evalDereverb:format', 'unsupported candidate format %s', ext);
end
y = double(y);
end
