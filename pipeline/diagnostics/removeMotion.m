function y = removeMotion(P, x, srcPos)
% The array's motion relative to a fixed source taken OUT of a reference
% signal: the inverse of receivedDirect. x is shifted earlier by the change in
% source-to-array range since the reference pose and scaled by r/rRef, so its
% direct component no longer moves with the wearer.
%
% Any reference built from the moving array's own signals (a beamformer
% output, or the perfect front end) carries that motion; the dry source does
% not. The motion-aware RLS translates the field with the tracked pose either
% way, and movingReference.m tests whether the motion then counts twice.
%
%   srcPos  [1 x 2] source position in the take's frame W' (x forward at the
%           reference pose): the oracle's P.srcPos(1:2) (addOracleDoa), or the
%           glasses' own estimate, their acoustic bearing at the vision
%           distance (movingReference.m). Ranges are from the head's acoustic
%           centre (acousticCentre), as receivedDirect's, so a pure head turn
%           moves nothing.
%
% First order in the range rate: the shift is taken at the output time, not
% the emission time, which at walking speeds differs by under half a sample.
arguments
    P struct
    x (:,1) double
    srcPos (1,2) double
end
[pc, d] = acousticCentre(P);
r = vecnorm(srcPos - pc, 2, 2);
rRef = norm(srcPos - [-d 0]);               % the acoustic centre at the reference pose (yaw 0)
n = min(numel(x), numel(r));
dly = (r(1:n) - rRef)/343*P.fs;             % samples later than at the reference pose
y = interp1((1:numel(x))', x, (1:n)' + dly, 'spline', 0) .* (r(1:n)/rRef);
y(end+1:numel(x)) = 0;
end
