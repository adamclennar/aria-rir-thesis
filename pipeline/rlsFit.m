function [rir, conv] = rlsFit(P, pref, tag, model)
% Run the motion-aware RLS against a given pseudo-reference.
%
% fdChRlsRotTransAtf asserts that the array signals, the reference signal AND
% all three tracking vectors are the same length. Truncating only the first two
% -- which is what you do naturally when the reference comes back a different
% length from the beamformer -- trips that assert. Everything is cut here in one
% place so no caller has to remember.
arguments
    P struct
    pref (:,1) double
    tag (1,:) char = ''
    model (1,:) char {mustBeMember(model,{'motionAware','omni'})} = 'motionAware'
end
mn = min(size(P.sigsPre,1), numel(pref));
p = pref(1:mn); p(~isfinite(p)) = 0;
switch model
    case 'motionAware'
        ord = P.chOrder; Dm = P.D;
        rot = P.rotYawPre(1:mn); tAzi = P.transAziPre(1:mn); tRad = P.transRadPre(1:mn);
    case 'omni'
        % estimateRirs.m's blind-omni condition: zeroth order, the omni slice of
        % the array matrix, and zeroed motion. The omni model deliberately
        % ignores array movement and absorbs it as time variation of the field.
        % It shares the ENTIRE front end with the motion-aware model -- same WPE,
        % same beamformer, same pseudo-reference -- so scoring both costs one
        % extra RLS and nothing else.
        ord = 0; Dm = P.DOmni;
        z = zeros(mn,1); rot = z; tAzi = z; tRad = z;
end
t0 = tic;
[rir,~,~,isConv,convSec] = fdChRlsRotTransAtf(P.sigsPre(1:mn,:), p, P.fs, ...
    ord, P.chDef, Dm, rot, tAzi, tRad, ...
    P.regulRls, 1, P.blockLenSmp, P.hopSizeSmp, P.filterLenSmp, P.winR, true);
conv = struct('isConverged', isConv, 'convTimeSec', convSec, 'rlsMin', toc(t0)/60, ...
              'model', model);
if ~isempty(tag)
    fprintf('[%s] %-28s %-12s rls %.1f min, converged %d @ %.1f s\n', P.takeName, tag, ...
        model, conv.rlsMin, isConv, convSec);
end
end
