function [sigBfd, medianDoaAziEleRad] = musicAndBeamformTimeDomainAdaptiveTraj(sig, stftBlockLenSmp, stftHopSizeSmp, ...
    doaObservationLenMs, arrayTransferFunctions, atfGridAziZenRad, params, trueDoaAziEleRad, plotFoundDoas)
% MODIFIED COPY of musicAndBeamformTimeDomainAdaptive.m from Deppisch et al.'s
% room-acoustic-matching (external/room-acoustic-matching/code/lib, commit
% 76c6ea3), used under the Chalmers Non-Commercial Software License Agreement
% for non-commercial academic purposes only; see README.md in this folder.
% Modified 2026-09-26 (aria_rir), copied here 2026-10-11. The only change: the
% oracle DoA may be a per-sample TRAJECTORY.
%
% Upstream passes the same `trueDoaAziEleRad` to every block, which is correct
% for a static array but wrong here: the head rotates through +/-180 deg, so the
% source bearing in the array frame sweeps continuously. Passing a constant
% would steer the beamformer at a fixed direction for the whole take and would
% understate the oracle condition badly.
%
% trueDoaAziEleRad may be:
%   []            -> estimate the DoA with MUSIC (upstream behaviour)
%   1 x 2         -> constant oracle DoA (upstream behaviour)
%   numSamples x 2 -> per-sample oracle DoA, sampled at each block's centre

if nargin < 7 || isempty(plotFoundDoas)
    plotFoundDoas = false;
end

stftWin = sqrt(hann(stftBlockLenSmp, 'periodic'));
fftLen = stftBlockLenSmp;
numPosFreqs = fftLen/2+1;
sigStft = stft(sig,stftWin,stftHopSizeSmp,fftLen,params.fs); % fftLen x numBlocks x numChannels
numBlocks = size(sigStft,2);
numMics = size(sigStft,3);

numFramesPerDoa = ceil((doaObservationLenMs/1000 * params.fs) / stftHopSizeSmp); % how many blocks should we observe for each doa estimate
if mod(numFramesPerDoa,2) == 0 % make odd number for symmetric centering
    numFramesPerDoa = numFramesPerDoa+1;
end
numAnalysisFramesOnEitherSide = (numFramesPerDoa-1)/2;

if ~isfield(params,'fftLen')
    params.fftLen = stftBlockLenSmp;
end

stftBfd = zeros(fftLen/2+1, numBlocks);
medianDoaAziEleRad = zeros(numBlocks,2);
for ii = 1:numBlocks
    % the frame we currently estimate the DOA for is ii, we center the
    % analysis around ii
    analysisStartFrame = max(ii-numAnalysisFramesOnEitherSide,1);
    analysisEndFrame = min(ii+numAnalysisFramesOnEitherSide,numBlocks);

    if ii < numAnalysisFramesOnEitherSide+1
        centerFrameIdx = ii;
    else
        centerFrameIdx = numAnalysisFramesOnEitherSide+1;
    end

    currSigStft = sigStft(1:numPosFreqs,analysisStartFrame:analysisEndFrame,:); % frames used in current estimation
    % Sample the oracle trajectory at this block's centre. Block ii spans
    % samples starting at (ii-1)*stftHopSizeSmp, matching the STFT framing.
    if isempty(trueDoaAziEleRad) || size(trueDoaAziEleRad,1) == 1
        currTrueDoa = trueDoaAziEleRad;
    else
        smpIdx = min(max((ii-1)*stftHopSizeSmp + round(stftBlockLenSmp/2), 1), size(trueDoaAziEleRad,1));
        currTrueDoa = trueDoaAziEleRad(smpIdx,:);
    end
    [currStftBfd, medianDoaAziEleRad(ii,:)] = musicAndBeamform(currSigStft, arrayTransferFunctions(1:numPosFreqs,:,:), atfGridAziZenRad, params, currTrueDoa, plotFoundDoas);
    stftBfd(:,ii) = currStftBfd(:,centerFrameIdx);
end

sigBfd = istft([stftBfd; conj(flipud(stftBfd(2:end-1,:)))], stftWin, stftHopSizeSmp, 'symmetric');


