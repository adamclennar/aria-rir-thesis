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
% Start where the reference does. Upstream's early stop compares successive
% filter estimates in dB; a block whose reference is EXACTLY zero leaves the
% estimate at exactly zero, the change is -Inf dB, and the RLS reports
% "converged" after one block with an empty RIR. That happens on the
% old-protocol takes, whose drive file opens with ~2 s of digital silence
% (a 2 s block lies wholly inside it). Silent reference samples carry no
% information, so dropping them -- keeping the pre-pad before the first
% non-zero sample -- changes nothing else: a reference that does not start
% with exact zeros (every blind pseudo-reference, every composite take) is
% used from sample 1 as before.
i0 = find(p ~= 0, 1);
if isempty(i0)
    error('rlsFit:silentReference', '%s: the reference signal is all zeros', P.takeName);
end
s0 = max(1, i0 - P.prePad);
switch model
    case 'motionAware'
        ord = P.chOrder; Dm = P.D;
        rot = P.rotYawPre(s0:mn); tAzi = P.transAziPre(s0:mn); tRad = P.transRadPre(s0:mn);
    case 'omni'
        % estimateRirs.m's blind-omni condition: zeroth order, the omni slice of
        % the array matrix, and zeroed motion. The omni model deliberately
        % ignores array movement and absorbs it as time variation of the field.
        % It shares the ENTIRE front end with the motion-aware model -- same WPE,
        % same beamformer, same pseudo-reference -- so scoring both costs one
        % extra RLS and nothing else.
        ord = 0; Dm = P.DOmni;
        z = zeros(mn-s0+1,1); rot = z; tAzi = z; tRad = z;
end
t0 = tic;
[rir,~,~,isConv,convSec] = fdChRlsRotTransAtf(P.sigsPre(s0:mn,:), p(s0:mn), P.fs, ...
    ord, P.chDef, Dm, rot, tAzi, tRad, ...
    P.regulRls, 1, P.blockLenSmp, P.hopSizeSmp, P.filterLenSmp, P.winR, true);
convSec = convSec + (s0-1)/P.fs;      % back on the take's own time axis
conv = struct('isConverged', isConv, 'convTimeSec', convSec, 'rlsMin', toc(t0)/60, ...
              'model', model, 'startSec', (s0-1)/P.fs);
if ~isempty(tag)
    fprintf('[%s] %-28s %-12s rls %.1f min, converged %d @ %.1f s\n', P.takeName, tag, ...
        model, conv.rlsMin, isConv, convSec);
end
end
