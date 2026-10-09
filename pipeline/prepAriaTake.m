function P = prepAriaTake(takeName, opts)
% Load one of our take files and derive everything the paper's pipeline
% needs, mirroring estimateRirs.m step for step. Differences from the paper's
% setup, all forced by the data rather than chosen:
%
%   * ATFs are the CHiME Aria set (data/anechoic/ariaAtfsHorizontal.mat),
%     48 directions on the horizontal ring, in the paper's file layout.
%   * Block length comes from the measured RT of the room's SM57 reference
%     (blockLenMsSuggested = smallest 500 ms multiple >= 2 x max T20),
%     because the RLS RIR window is half the block. Override with blockLenMs.
%   * The reference is a single omni at the LISTENING position (not the
%     paper's order-10 EMA), so RT and DRR are both scored, with the caveat
%     that a free-field omni is not the array's order-0 channel on a head.
%     No EMA-to-CH conversion is needed.
%   * There is no array RIR at the reference pose (paper field IRs), so no
%     frame offset / oracle DoA is derived here.
%
% Everything downstream (WPE, MUSIC/MPDR beamformer, motion-aware and omni
% RLS) is the vendored upstream code, unchanged.
arguments
    takeName (1,:) char
    opts.blockLenMs (1,1) double = NaN
    opts.filterLenMs (1,1) double = NaN
    opts.atfFile (1,:) char = 'ariaAtfsHorizontal.mat'
    opts.runWpe (1,1) logical = true
    opts.sourceField (1,:) char = 'sourceSig'   % informed reference: 'sourceSig' (drive) or 'sourceSigSm57'
end
ensurePaths();
root = fileparts(fileparts(mfilename('fullpath')));
P = struct(); P.takeName = takeName; P.root = root;

d = load(fullfile(root,'data','rooms',[takeName '.mat']));
P.d = d; fs = double(d.fs); P.fs = fs;
P.numMics = size(d.arraySigs,2);
P.room = regexp(takeName,'^[a-z]+','match','once');
P.speechSpanS = d.meta.speech_span_s(:).';

% ---- reference RIR and block length ----------------------------------------
refFile = fullfile(root,'data','rooms',[P.room '_reference.mat']);
if ~exist(refFile,'file')
    % Only buildSweepReference gets here: it needs P.D to encode the array
    % sweeps that BECOME the reference, so the reference cannot exist yet.
    if isnan(opts.blockLenMs)
        error('prepAriaTake:noReference', '%s has no reference file; pass blockLenMs', P.room);
    end
    P.refFile = ''; P.rirRef = []; P.rtRef = []; P.fAxM = []; P.drrRefOmni = NaN;
    P.drrRef = NaN; P.rtRefArray = []; P.drrRefSource = 'none'; P.distanceM = NaN;
    P.drrRefPaper = NaN; P.revRefDb = [];
    P.blockLenMs = opts.blockLenMs;
else
refS = load(refFile);
P.refFile = [P.room '_reference.mat'];
P.rirRef = refS.IRs(:,1);
[P.rtRef, P.fAxM] = getT20Ita(P.rirRef, fs, 1, [125 8000]);   % same estimator as for the estimates
P.rtRef = P.rtRef(:).'; P.fAxM = P.fAxM(:).';
P.drrRefOmni = double(refS.drrDb);
% DRR is scored against the ARRAY reference where one exists: the pooled
% CH-domain DRR of the in-take sweeps, median over every take of the room.
% A free-field omni on a stand is the wrong instrument for it -- it reads
% several dB higher than a head-worn array at the same spot, so scoring
% against it made every condition look like it under-estimated DRR by 4-10 dB
% when the true errors are within a decibel or two. RT still comes from the
% omni: a decay slope is a room property and transfers between instruments.
% The octave-band RMS error is scored against the array reference too, for
% the same reason (same microphones, same head): its spectral shape is
% revBandsDb. A reference built before these fields existed leaves them
% NaN / empty, so the scores come out NaN rather than wrong.
arrayRefFile = fullfile(root,'data','rooms',[P.room '_arrayReference.mat']);
P.drrRefPaper = NaN; P.revRefDb = [];
if exist(arrayRefFile,'file')
    aR = load(arrayRefFile);
    P.drrRef = double(aR.drrDb);
    P.rtRefArray = double(aR.rt20Bands(:)).';
    P.drrRefSource = sprintf('array, pooled over %d takes', numel(aR.drrPerTake));
    if isfield(aR,'drrDbPaper'); P.drrRefPaper = double(aR.drrDbPaper); end
    if isfield(aR,'revBandsDb'); P.revRefDb = double(aR.revBandsDb(:)).'; end
else
    P.drrRef = P.drrRefOmni;
    P.rtRefArray = [];
    P.drrRefSource = 'omni (no array reference for this room)';
end
P.distanceM = double(refS.distanceM);
P.blockLenMs = double(refS.blockLenMsSuggested);
if ~isnan(opts.blockLenMs); P.blockLenMs = opts.blockLenMs; end
end

% ---- ATFs ------------------------------------------------------------------
atfS = load(fullfile(root,'data','anechoic',opts.atfFile));
P.atfFile = opts.atfFile;
P.atfIrs = atfS.irs;
P.atfAziRad = -atfS.rotationAnglesDeg(:)*pi/180;   % stored as -sourceAzimuth, as in the paper
P.numAtfDirs = size(P.atfIrs,3);
P.atfGrid = [P.atfAziRad, pi/2*ones(P.numAtfDirs,1)];

% ---- WPE parameters (published) -------------------------------------------
P.wpe = defaultWpeParams(fs);

% ---- RLS setup (estimateRirs.m lines 114-131) ------------------------------
P.chOrder = 2; P.chDef = 'complex';
P.blockLenSmp = P.blockLenMs/1000*fs;
fAxR = linspace(0,fs/2,P.blockLenSmp/2+1)';
[D,~,~] = getChdArrayEncDecAtf(fft(P.atfIrs,P.blockLenSmp), P.atfAziRad, fs, ...
    P.chOrder, P.chDef, P.blockLenSmp, 2048, 1e-2);
P.D = D; P.DOmni = D(:,:,1);
P.hopSizeSmp = round(P.blockLenSmp/4);
P.filterLenSmp = P.blockLenSmp/2;
if ~isnan(opts.filterLenMs); P.filterLenSmp = round(opts.filterLenMs/1000*fs); end
P.winR = sqrt(hann(P.blockLenSmp,'periodic'));
P.regulRls = 1e-2 .* interp1(d.fAxNoise(:), d.noisePower(:), fAxR);

% ---- tracking (estimateRirs.m lines 134-140) --------------------------------
translation = d.translationXYZCart - d.referencePos;
P.rotYaw = d.rotationYawPitchRollRad(:,1) - d.referenceRotationAziEleRad(1);
[P.transAzi,~,P.transRad] = cart2sph(translation(:,1), translation(:,2), zeros(size(translation,1),1));
P.prePad = round(0.005*fs);
P.sigsPre     = [zeros(P.prePad,P.numMics); d.arraySigs];
P.rotYawPre   = [zeros(P.prePad,1); P.rotYaw];
P.transAziPre = [zeros(P.prePad,1); P.transAzi];
P.transRadPre = [zeros(P.prePad,1); P.transRad];
P.sourceField = opts.sourceField;
P.sourceSigPre = [zeros(P.prePad,1); d.(opts.sourceField)(:)];
% Where the informed RIR's direct peak lands: the source was placed with its
% direct path 10 ms into the take; rlsFit pre-pads array AND source alike, so
% that delay is preserved, minus the ATFs' own onset (~0.7 ms at 1.53 m),
% which the array matrix carries and the RLS absorbs as an advance. The
% blind RIRs peak at the 5 ms pre-pad instead (pseudo-reference is derived
% from the unpadded array signals, as in estimateRirs.m).
[~,atfOnset] = max(max(abs(P.atfIrs(:,:,1)),[],2));
P.atfOnsetSmp = atfOnset;
P.directPathSmp = double(d.meta.pre_delay_samples) - atfOnset;

% ---- beamformer (estimateRirs.m lines 76-86) --------------------------------
P.bf.fs = fs; P.bf.musicFreqRange = [100,1000]; P.bf.bfType = 'MPDR';
P.bf.doaObservationLenMs = 400; P.bf.regulConst = 1e-2;
P.bf.bfBlockLen = 2048; P.bf.bfHopSize = 1024;
P.atfsBf = fft(P.atfIrs, P.bf.bfBlockLen);

fprintf('[%s] blockLen %d ms (RIR window %d ms), %d ATF dirs, ref T20 %s s (omni), ref DRR %+.1f dB [%s] at %.2f m\n', ...
    takeName, P.blockLenMs, round(P.filterLenSmp/fs*1000), P.numAtfDirs, mat2str(round(P.rtRef,2)), ...
    P.drrRef, P.drrRefSource, P.distanceM);

% ---- dereverberated signals (cached) ----------------------------------------
if opts.runWpe
    P.derev = getWpe(P, P.wpe, 'published', fullfile(root,'results','cache'));
end
end
