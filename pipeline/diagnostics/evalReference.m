function [rows, rirs] = evalReference(P, ref, label, opts)
% Run the RLS (both signal models) against one reference signal and score
% the reference and the RIRs. The common core of the front-end anchors
% (frontEndAnchors) and the dereverberator harness (evalDereverb).
%
%   ref        N x 1, sample-aligned with P.d.arraySigs
%   opts.alignment
%     'array'  ref is derived from the array signals (a beamformer output or a
%              post-filter of one), as in the blind pipeline: passed to the
%              RLS unpadded, so the RIR's direct path lands at the 5 ms pre-pad.
%     'source' ref is aligned like sourceSig (direct path arrives at the array
%              10 ms later): pre-padded like P.sourceSigPre, as in the informed
%              pipeline, so the direct path lands at P.directPathSmp.
%   opts.rirs  optional struct with precomputed RIRs per model (e.g. the
%              baseline run's); the RLS is skipped for those models.
%   opts.carriesMotion  whether ref carries the array's motion (range delay,
%              1/r) like a beamformer output -- decides which source model it
%              is scored against (scoreRefSig). Default: true for 'array'
%              alignment, false for 'source'; references built from
%              receivedDirect are 'source'-aligned but carry the motion.
%
% rows: one struct per model with the reference scores (identical for both,
% the reference does not depend on the model) and the RIR scores (scoreRir:
% T20 per band, DRR under both windows, tonal balance at the microphones).
arguments
    P struct
    ref (:,1) double
    label (1,:) char
    opts.alignment (1,:) char {mustBeMember(opts.alignment,{'array','source'})} = 'array'
    opts.models (1,:) cell = {'motionAware','omni'}
    opts.rirs struct = struct()
    opts.extra struct = struct()     % extra scalar fields copied into every row
    opts.carriesMotion = []
end
if isempty(opts.carriesMotion); opts.carriesMotion = strcmp(opts.alignment, 'array'); end
n = size(P.d.arraySigs,1);
ref = ref(1:min(end,n)); ref(end+1:n) = 0;
ref(~isfinite(ref)) = 0;
R = scoreRefSig(P, ref, 'carriesMotion', opts.carriesMotion);
switch opts.alignment
    case 'array';  refRls = ref;                         expect = P.prePad;
    case 'source'; refRls = [zeros(P.prePad,1); ref];    expect = P.directPathSmp;
end
rows = {}; rirs = struct();
for m = 1:numel(opts.models)
    mdl = opts.models{m};
    t0 = tic;
    if isfield(opts.rirs, mdl)
        rir = opts.rirs.(mdl); conv = struct('isConverged', NaN);
    else
        [rir, conv] = rlsFit(P, refRls, '', mdl);
    end
    rirs.(mdl) = rir;
    S = scoreRir(P, rir, expect);
    r = struct('take', string(P.takeName), 'label', string(label), 'model', string(mdl), ...
        'alignment', string(opts.alignment), 'blockLenMs', P.blockLenMs, ...
        'carriesMotion', opts.carriesMotion, 'refCohEff', R.refCohEff, ...
        'refCoh', R.refCoh, 'refCohComp', R.refCohComp, 'refErrDb', R.refErrDb, ...
        'refFiltVarDb', R.refFiltVarDb, 'refEarlyDb', R.refEarlyDb, 'refLateDb', R.refLateDb, ...
        'refTailDb', R.refTailDb, 'refFloorDb', R.refFloorDb, ...
        'rtErrMean', S.rtErrMean, 'rtErrRms', S.rtErrRms, 'rtErrMedian', S.rtErrMedian, ...
        'rtErrSigned', S.rtErrSigned, 'drrDb', S.drrDb, 'drrErr', S.drrErr, ...
        'drrDbPaper', S.drrDbPaper, 'drrErrPaper', S.drrErrPaper, 'rmsErrMeanDb', S.rmsErrMeanDb, ...
        'directMs', S.directMs, 'converged', conv.isConverged, 'rlsMin', toc(t0)/60);
    fe = fieldnames(opts.extra);
    for k = 1:numel(fe); r.(fe{k}) = opts.extra.(fe{k}); end
    % per-band vectors last: rowsToTable spreads them over one column per band
    r.refCohBands = R.refCohBands;
    r.refFiltVarBands = R.refFiltVarBands;
    r.t20Err = S.bandErr;
    r.rmsAbsErr = S.rmsBandAbsErrDb;
    rows{end+1} = r; %#ok<AGROW>
    fprintf(['[%s] %-18s %-11s coh %.3f drift %4.2f | ref early %+5.1f late %+5.1f (floor %+5.1f) dB' ...
        ' | T20 err %5.1f%% | DRR err %+5.1f dB (paper window %+5.1f) | %.1f min\n'], P.takeName, label, mdl, ...
        R.refCohEff, R.refFiltVarDb, R.refEarlyDb, R.refLateDb, R.refFloorDb, S.rtErrMean, S.drrErr, ...
        S.drrErrPaper, r.rlsMin);
end
end
