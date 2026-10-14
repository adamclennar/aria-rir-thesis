function P = addOracleDoa(P)
% Per-sample source bearing and range in the head frame, from the recording
% protocol rather than from an acoustic estimate: the oracle the diagnostics
% steer the beamformer at and build the perfect front end from.
%
% The loudspeaker was not tracked, so its position is reconstructed from two
% facts about the REFERENCE pose, which is the origin of the take's world
% frame (build_take puts referencePos = 0 and reference yaw = 0 there):
%
%   bearing   0 deg. The reference pose is the stillest 2 s of the in-take
%             sweep, when the wearer stands on the mark FACING the loudspeaker,
%             and the world frame's x axis is the wearer's forward direction
%             there. So the source is straight ahead of the reference pose.
%   range     the source-to-mark distance measured with the omni on the mark
%             (reference.mat distanceM: the sweep's delay on the laptop's own
%             clock minus the calibrated playback latency). The glasses cannot
%             give it: their clock is not shared with the laptop's, so the
%             delay in their recording mixes clock offset, latency and travel.
%
% The source position is then fixed in the world frame, and the bearing and
% range are recomputed at every sample from the tracked pose -- from the
% array's ACOUSTIC centre (acousticCentre: 8 cm behind the tracked CPF), the
% point the ATFs and so the beamformer's steering are referenced to. Its error
% is how squarely the wearer faced the loudspeaker: MUSIC on the static takes
% puts the source within -4..+3 deg of straight ahead.
%
% A first version took the bearing from the in-take sweep instead, matching
% the 7-channel direct path against every dense dummy-head ATF. The match is
% weak (the best direction beats the next by 0.1-4%) and put the source
% 30-114 deg off on 3 of 31 takes where the wearer faced it. It is still
% computed and reported (atfMatchAziRad, atfMatchScore, atfMatchMarginPct) as
% a cross-check, and used for nothing.
%
% Takes qualify when their reference pose faces the loudspeaker: every take
% with an in-take sweep (an IRs field), and the takes without one that
% config/rooms.json marks faces_source_at_reference -- the corridor's static
% takes (the stillest moment of the speech, standing on the mark) and its
% head-only takes (reference instants fixed by hand where the pose faces the
% loudspeaker, found from MUSIC and the tracked yaw, so for those two the
% bearing is not independent of MUSIC). Elsewhere the reference pose comes
% from the speech, where the wearer need not face the loudspeaker; those
% takes get P.hasOracle = false and empty oracle fields. Without a sweep there
% is no ATF cross-check (NaN).
%
% Adds: hasOracle, srcAziRefRad (0), srcPos [1x3], oracleAzi [Nx1],
%       oracleTraj [Nx2], srcRangeM [Nx1], atfMatchAziRad, atfMatchScore,
%       atfMatchMarginPct
P.hasOracle = false;
P.oracleAzi = []; P.oracleTraj = []; P.srcPos = []; P.srcAziRefRad = NaN; P.srcRangeM = [];
P.atfMatchAziRad = NaN; P.atfMatchScore = NaN; P.atfMatchMarginPct = NaN;
if isfield(P.d, 'IRs')
    P = atfMatch(P);
elseif ~facesSourceAtReference(P)
    fprintf('[%s] no in-take sweep, reference pose not known to face the source: no protocol bearing\n', P.takeName);
    return
end

% Source straight ahead of the reference pose, distanceM from the acoustic
% centre, which at the reference (yaw 0 by construction) is at (-d, 0).
P.srcAziRefRad = 0;
[pc, dOff] = acousticCentre(P);
P.srcPos = [-dOff 0 0] + P.distanceM*[1 0 0];
v = P.srcPos(1:2) - pc;
P.oracleAzi = wrapToPi(atan2(v(:,2), v(:,1)) - P.rotYaw);
P.oracleTraj = [P.oracleAzi, zeros(size(P.oracleAzi))];
P.srcRangeM = vecnorm(v,2,2);
P.hasOracle = true;
fprintf('[%s] protocol bearing: source straight ahead of the reference pose at %.2f m (dense-ATF match %+.0f deg, score %.2f, margin %.1f%%)\n', ...
    P.takeName, P.distanceM, P.atfMatchAziRad*180/pi, P.atfMatchScore, P.atfMatchMarginPct);
end

function tf = facesSourceAtReference(P)
% config/rooms.json: rooms.<room>.takes.<take>.faces_source_at_reference
cfg = jsondecode(fileread(fullfile(P.root, 'config', 'rooms.json')));
tf = false;
if isfield(cfg.rooms, P.room) && isfield(cfg.rooms.(P.room).takes, P.takeName)
    tc = cfg.rooms.(P.room).takes.(P.takeName);
    tf = isfield(tc, 'faces_source_at_reference') && isequal(tc.faces_source_at_reference, true);
end
end

function P = atfMatch(P)
% Cross-check only: the source bearing at the reference pose from the in-take
% sweep's direct path, matched against every dense ATF.
fs = P.fs;
A = load(fullfile(P.root,'data','anechoic','ariaAtfsHorizontalDense.mat'));
aziGrid = -A.rotationAnglesDeg(:)*pi/180;          % same convention as P.atfAziRad

% Direct-path segment of the measured array IR: -1 ms .. +2 ms around the
% strongest sample over all mics WITHIN THE FIRST 8 ms (build_take cuts the IR
% with its direct peak at 5 ms; a later, louder reflection on a shadowed mic
% would otherwise win).
irs = P.d.IRs;
[~,pk] = max(max(abs(irs(1:round(0.008*fs),:)),[],2));
seg = irs(max(1,pk-round(0.001*fs)) : min(size(irs,1), pk+round(0.002*fs)), :);

% Per-frequency normalised inner product between the measured and each ATF's
% 7-mic pattern, 200-6000 Hz: the SPATIAL pattern (inter-mic level and phase),
% not the spectrum. The magnitude is taken per frequency before summing, so
% the arbitrary bulk delay between the segment and the ATFs cancels.
nfft = 512;
f = (0:nfft/2)'/nfft*fs; sel = f >= 200 & f <= 6000;
Sm = fft(seg, nfft); Sm = Sm(sel,:); Sm = Sm./vecnorm(Sm,2,2);
Aa = fft(A.irs, nfft); Aa = Aa(sel,:,:);
nDir = size(Aa,3); sc = zeros(nDir,1);
for k = 1:nDir
    Av = Aa(:,:,k); Av = Av./vecnorm(Av,2,2);
    sc(k) = sum(abs(sum(conj(Av).*Sm, 2)));
end
[scS, ord] = sort(sc, 'descend');
P.atfMatchAziRad = aziGrid(ord(1));
P.atfMatchScore = scS(1)/nnz(sel);                 % 1 = perfect match
P.atfMatchMarginPct = 100*(scS(1)-scS(2))/scS(1);
end
