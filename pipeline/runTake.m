function R = runTake(takeName, opts)
% The paper's conditions on one of our takes, with the upstream code:
%
%   informed  motion-aware / omni   RLS driven by the true source signal
%   blind     motion-aware / omni   WPE -> MUSIC + MPDR beamformer -> RLS
%
% Saves results/pipeline/<take>.mat (RIRs, scores, per-block DoA, the
% pseudo-reference) and prints one gate line per condition. On a static take
% the informed motion-aware RIR is the build check: its peak must sit at
% P.directPathSmp (~9.3 ms) and its T20 must track the SM57 reference.
arguments
    takeName (1,:) char
    opts.blockLenMs (1,1) double = NaN
    opts.atfFile (1,:) char = 'ariaAtfsHorizontal.mat'
    opts.sourceField (1,:) char = 'sourceSig'
    opts.tag (1,:) char = ''
end
P = prepAriaTake(takeName, 'blockLenMs', opts.blockLenMs, 'atfFile', opts.atfFile, ...
                 'sourceField', opts.sourceField);
R = struct('take', takeName, 'blockLenMs', P.blockLenMs, 'atfFile', P.atfFile, ...
           'sourceField', P.sourceField, 'fAx', P.fAxM, 'rtRef', P.rtRef, 'drrRef', P.drrRef, 'drrRefOmni', P.drrRefOmni, ...
           'drrRefSource', P.drrRefSource, 'distanceM', P.distanceM, ...
           'directPathMs', P.directPathSmp/P.fs*1e3);
models = {'motionAware','omni'};

% ---- informed -------------------------------------------------------------
for m = 1:2
    [rir, conv] = rlsFit(P, P.sourceSigPre, 'informed', models{m});
    R.rir.(['informed_' models{m}]) = rir;
    R.conv.(['informed_' models{m}]) = conv;
    R.score.(['informed_' models{m}]) = scoreRir(P, rir);
end

% ---- blind ----------------------------------------------------------------
[rirs, pref, doa, conv] = bfEstimate(P, P.derev, [], 'blind', models);
for m = 1:2
    R.rir.(['blind_' models{m}]) = rirs.(models{m});
    R.conv.(['blind_' models{m}]) = conv.(models{m});
    R.score.(['blind_' models{m}]) = scoreRir(P, rirs.(models{m}));
end
R.pref = pref;
R.doaHeadRad = doa(:,1);                                     % MUSIC azimuth per beamformer block, head frame
ctr = min(size(P.d.arraySigs,1), round((0:numel(R.doaHeadRad)-1)'*P.bf.bfHopSize + P.bf.bfBlockLen/2));
ctr = max(ctr,1);
R.doaBlockCentreS = ctr/P.fs;
yawAt = P.d.rotationYawPitchRollRad(ctr,1);
R.doaWorldRad = wrapToPi(R.doaHeadRad + yawAt);              % de-rotated by tracked yaw: constant if MUSIC is right
inSpeech = R.doaBlockCentreS > P.speechSpanS(1) & R.doaBlockCentreS < P.speechSpanS(2);
z = exp(1i*R.doaWorldRad(inSpeech));
R.doaWorldMedianDeg = angle(median(real(z)) + 1i*median(imag(z)))*180/pi;
R.doaWorldSpreadDeg = 180/pi*sqrt(-2*log(abs(mean(z))));    % circular std
R.doaWorldWithin15Pct = 100*mean(abs(wrapToPi(R.doaWorldRad(inSpeech) - R.doaWorldMedianDeg*pi/180)) < 15*pi/180);

% ---- report ---------------------------------------------------------------
fprintf('\n[%s] blockLen %d ms | ref T20 %s (omni) | ref DRR %+.1f dB (%s)\n', ...
    takeName, P.blockLenMs, mat2str(round(P.rtRef,2)), P.drrRef, P.drrRefSource);
c = fieldnames(R.score);
for k = 1:numel(c)
    S = R.score.(c{k});
    fprintf('  %-22s peak %5.1f ms  T20 %s  err med %5.1f%% mean %5.1f%% rms %5.1f%%  DRR %+5.1f dB (err %+5.1f)  conv %d\n', ...
        c{k}, S.peakMs, mat2str(round(S.rt,2)), S.rtErrMedian, S.rtErrMean, S.rtErrRms, S.drrDb, S.drrErr, R.conv.(c{k}).isConverged);
end
fprintf('  MUSIC DoA in world frame: median %+.0f deg, circular std %.0f deg, %.0f%% of blocks within 15 deg\n', ...
    R.doaWorldMedianDeg, R.doaWorldSpreadDeg, R.doaWorldWithin15Pct);

out = fullfile(P.root,'results','pipeline');
if ~exist(out,'dir'); mkdir(out); end
save(fullfile(out,[takeName opts.tag '.mat']), '-struct', 'R', '-v7.3');
end
