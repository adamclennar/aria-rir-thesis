function [rir, pref, doaUsed, conv] = bfEstimate(P, inputSig, doaTraj, tag, models)
% Beamform with the paper's MUSIC + MPDR front end (upstream
% musicAndBeamformTimeDomainAdaptive, unmodified) and run the RLS on the
% resulting pseudo-reference. doaTraj = [] estimates the DoA with MUSIC, as in
% the paper; a constant 1 x 2 [azi ele] steers at a fixed oracle direction.
arguments
    P struct
    inputSig double
    doaTraj = []
    tag (1,:) char = ''
    models = {'motionAware'}   % cellstr; 'omni' shares the whole front end, so
                               % asking for both costs one extra RLS and no more
end
ensurePaths();
if size(doaTraj,1) > 1
    error('bfEstimate:traj', ['a time-varying DoA needs a modified beamformer, which is not ' ...
          'part of the baseline (upstream accepts [] or one constant direction)']);
end
t0 = tic;
[sb, doaUsed] = musicAndBeamformTimeDomainAdaptive(inputSig, P.bf.bfBlockLen, ...
    P.bf.bfHopSize, P.bf.doaObservationLenMs, P.atfsBf, P.atfGrid, P.bf, doaTraj, false);
tBf = toc(t0);

pref = sb(:);
if ischar(models); models = {models}; end
rir = struct(); conv = struct();
for m = 1:numel(models)
    [rir.(models{m}), conv.(models{m})] = rlsFit(P, pref, '', models{m});
    conv.(models{m}).bfMin = tBf/60;
    if ~isempty(tag)
        fprintf('[%s] %-26s %-12s bf %.1f min, rls %.1f min, converged %d @ %.1f s\n', ...
            P.takeName, tag, models{m}, tBf/60, conv.(models{m}).rlsMin, ...
            conv.(models{m}).isConverged, conv.(models{m}).convTimeSec);
    end
end
if numel(models) == 1
    % NOTE: the return TYPE depends on how many models were asked for -- a bare
    % matrix for one, a struct for several.
    rir = rir.(models{1});
    conv = conv.(models{1});
end
end
