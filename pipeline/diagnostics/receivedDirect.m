function b = receivedDirect(P, s, rRef, opts)
% The direct sound of s AS RECEIVED at the moving array: delayed by the
% change in source-to-array range and scaled by 1/r, relative to the range at
% `rRef` (default: the reference pose, where build_take placed sourceSig).
%
% This is what a PERFECT front end (perfect dereverberation, perfect
% steering, distortionless beamformer) would hand the RLS: an MPDR output is
% distortionless for the direct path, so its direct component carries the
% real propagation delay and spreading loss to the array, not the source as
% emitted. It is the zero-residual end of the dose-response study, and the
% source model that refFiltVarDb measures a pseudo-reference against.
%
% Needs P.srcRangeM (addOracleDoa), which is measured from the acoustic
% centre (acousticCentre), not the tracked CPF.
%
% opts.parts  'both' (the received direct sound), or only the range 'delay'
%             or only the 1/r 'gain': movingReference.m separates the two.
arguments
    P struct
    s (:,1) double
    rRef (1,1) double = P.distanceM
    opts.parts (1,:) char {mustBeMember(opts.parts,{'both','delay','gain'})} = 'both'
end
c = 343;
n = numel(s);
r = P.srcRangeM(1:n);
dly = (r - rRef)/c*P.fs;                     % samples later than at the reference pose
b = s;
if ~strcmp(opts.parts, 'gain'); b = interp1((1:n)', s, (1:n)' - dly, 'spline', 0); end
if ~strcmp(opts.parts, 'delay'); b = b .* (rRef./r); end
end
