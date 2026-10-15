function [traj, info] = musicWorldFrame(P, doaFrames, opts)
% Stabilise MUSIC's DoA using two things the published pipeline ignores: the
% source does not move, and the head pose is known at every sample.
%
% Not used to steer: the diagnostics steer at the protocol bearing
% (addOracleDoa), and steering barely changes the RIR estimates. Kept as the
% glasses' own (blind) source bearing, which the vision distance needs (RQ3).
%
% MUSIC estimates a bearing in the HEAD frame, independently in every frame.
% The head rotates, so a static source produces a bearing that sweeps, and every
% frame re-estimates it from scratch. Rotating each estimate into the world
% frame with the tracked yaw removes the motion; what is left scatters about a
% fixed source and can be averaged. The average is rotated back into the head
% frame to give a per-sample steering trajectory.
%
% Two ways of averaging (opts.mode):
%
%   'bearing'   average the world-frame BEARING (the first version).
%               Exact for rotation only. When the wearer walks, the bearing
%               from the array to a static source genuinely changes -- 1 m
%               sideways at 2 m range is ~27 deg -- so a whole-take average is
%               wrong on walking takes; a finite window (opts.windowSec)
%               follows it partly.
%   'position'  fit the source POSITION: the point in the horizontal plane
%               whose bearing from the tracked array position best matches
%               every frame's world-frame bearing (L1 on the wrapped angle
%               error, so front/back failures cannot drag it). The bearing is
%               then recomputed from the tracked position at every sample, so
%               translation is handled as well as rotation. When the wearer
%               barely translates the range is poorly determined, but so is its
%               effect on the bearing, so the trajectory is still right.
%
%   opts.windowSec  'bearing' mode only. Inf = one average over the take.
%   opts.robust     'bearing' mode: circular median rather than circular mean.
arguments
    P struct
    doaFrames (:,:) double
    opts.mode (1,:) char {mustBeMember(opts.mode,{'bearing','position'})} = 'bearing'
    opts.windowSec (1,1) double = Inf
    opts.robust (1,1) logical = true
end
hop = P.bf.bfHopSize;
nF = size(doaFrames,1);
nSmp = size(P.d.arraySigs,1);
ctr = max(min((0:nF-1)'*hop + round(P.bf.bfBlockLen/2), nSmp), 1);
yaw = P.rotYaw;
world = wrapToPi(doaFrames(:,1) + yaw(ctr));          % head -> world
frameOfSmp = min(floor((0:nSmp-1)'/hop) + 1, nF);

info.mode = opts.mode;
info.windowSec = opts.windowSec;
info.worldSpreadDeg = median(abs(wrapToPi(world - circAvg(world, true))))*180/pi;

switch opts.mode
    case 'bearing'
        if isinf(opts.windowSec)
            wAvg = repmat(circAvg(world, opts.robust), nF, 1);
        else
            w = max(3, round(opts.windowSec*P.fs/hop));
            half = floor(w/2);
            wAvg = zeros(nF,1);
            for k = 1:nF
                wAvg(k) = circAvg(world(max(1,k-half):min(nF,k+half)), opts.robust);
            end
        end
        headAzi = wrapToPi(wAvg(frameOfSmp) - yaw);
        info.worldMeanDeg = circAvg(world, opts.robust)*180/pi;
        info.srcPos = [NaN NaN];
        fprintf('[%s] world-frame DoA (bearing, %s): source %+.1f deg, MUSIC scatter about it %.1f deg\n', ...
            P.takeName, winStr(opts.windowSec), info.worldMeanDeg, info.worldSpreadDeg);

    case 'position'
        % bearings are referenced to the ATFs' centre, not the tracked CPF
        [pos, dOff] = acousticCentre(P);
        pF = pos(ctr,:);
        s = fitSourcePosition(pF, world);
        v = s - pos;
        headAzi = wrapToPi(atan2(v(:,2), v(:,1)) - yaw);
        vF = s - pF;
        resid = wrapToPi(atan2(vF(:,2), vF(:,1)) - world);
        info.srcPos = s;
        c0 = [-dOff 0];                                   % acoustic centre at the reference
        info.worldMeanDeg = atan2(s(2)-c0(2), s(1)-c0(1))*180/pi;  % bearing from the reference pose
        info.srcRangeM = norm(s - c0);
        info.residDeg = median(abs(resid))*180/pi;
        fprintf('[%s] world-frame DoA (position): source at %+.1f deg, %.2f m from the reference pose; MUSIC scatter about the fit %.1f deg\n', ...
            P.takeName, info.worldMeanDeg, info.srcRangeM, info.residDeg);
end
traj = [headAzi, zeros(nSmp,1)];
end

function s = fitSourcePosition(pF, world)
% Coarse polar grid search then a local refinement, minimising the sum of
% absolute wrapped bearing errors. Grid: 1 deg x 5 cm, 1-8 m. The lower bound
% matters on takes with little translation, where range is unobservable: an
% unbounded fit wandered to 0.26 m on a static take, and at that range 3 cm of
% sway already swings the steering by 6 deg. A source closer than 1 m would be
% outside the far-field ATFs (measured at 1.53 m) anyway.
az = (-180:1:179)*pi/180;
rg = 1.0:0.05:8;
[AZ, RG] = meshgrid(az, rg);
cand = [RG(:).*cos(AZ(:)), RG(:).*sin(AZ(:))];
s = bestOf(cand, pF, world);
% refine: 0.1 deg x 1 cm around the coarse optimum
r0 = norm(s); a0 = atan2(s(2), s(1));
[AZ, RG] = meshgrid(a0 + (-2:0.1:2)*pi/180, max(1.0, r0 + (-0.1:0.01:0.1)));
cand = [RG(:).*cos(AZ(:)), RG(:).*sin(AZ(:))];
s = bestOf(cand, pF, world);
end

function s = bestOf(cand, pF, world)
cost = zeros(size(cand,1),1);
chunk = 2000;
for i = 1:chunk:size(cand,1)
    j = i:min(i+chunk-1, size(cand,1));
    dx = cand(j,1).' - pF(:,1);                     % nF x nCand
    dy = cand(j,2).' - pF(:,2);
    cost(j) = sum(abs(wrapToPi(atan2(dy, dx) - world)), 1).';
end
[~, ix] = min(cost);
s = cand(ix,:);
end

function m = circAvg(a, robust)
if robust
    % circular median: the sample minimising the summed angular distance
    % (O(n^2), fine for a few thousand frames)
    d = sum(abs(wrapToPi(a - a.')), 1);
    [~,ix] = min(d); m = a(ix);
else
    m = angle(mean(exp(1i*a)));
end
end

function s = winStr(w)
if isinf(w); s = 'whole take'; else; s = sprintf('%g s window', w); end
end
