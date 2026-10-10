function S = scoreRir(P, rir, expectSmp)
% RT and DRR error of one estimate against the omni reference at the
% listening position, per octave band,
% plus the direct-path peak position (a build check: with sourceSig placed
% so the direct path arrives at 10 ms and 5 ms of pre-padding, the informed
% RIR's peak must sit at P.directPathSmp).
%
%   S.bandErr      signed % error per band (positive = RT over-estimated)
%   S.rtErrMedian  median |bandErr|  (the paper's statistic)
%   S.rtErrMean    mean   |bandErr|
%   S.rtErrRms     rms     bandErr
%   S.peakMs        time of the RIR's global |max|, ms (omni CH channel)
%   S.directMs      time of the largest peak INSIDE a window around the
%                   expected direct path, ms -- this is the build check
%   S.directRelMaxDb  that peak's level relative to the global max
%
% Use directMs, not peakMs. The global max only finds the direct arrival in a
% dry room: in the reverberant room (DRR -6.3 dB) the largest sample of a
% 1.5 s estimate is a reflection ~54 ms in, or a patch of estimation noise
% near t = 0, while the direct path sits correctly at 8.5-9.9 ms a decibel or
% two below it. Reporting the global max there suggested a broken build when
% the build was fine.
%   S.drrDb        DRR of the estimate
%   S.drrErr       DRR error vs P.drrRef, dB (positive = too direct). That is
%                  the pooled ARRAY reference where the room has one -- the
%                  like-for-like instrument.
%   S.drrErrOmni   the same against the free-field omni, kept for continuity
%                  with results produced before the array reference existed.
%                  The reference is a free-field omni and the estimate is the
%                  array's order-0 channel on a head, so a bias of a decibel
%                  or two is expected; compare conditions, not absolutes.
%   S.drrDbPaper, S.drrErrPaper  the same two under the paper's direct-sound
%                  window. drrDb/drrErr are the repo window; drrWindows.m
%                  defines both and says when to use which.
%   S.rmsMicErrDb  tonal-balance (octave-band RMS) error AT THE MICROPHONES,
%                  numMics x numBands: the estimate decoded to each
%                  microphone (chDecodeToMics), the spectral shape of its
%                  reverberant part (micRevShape: band levels, mean over bands
%                  removed) minus the measured microphone's (the array
%                  reference's revShapeMics). Only the shape is scored -- a
%                  blind estimate's absolute gain is arbitrary, and stimuli
%                  are loudness-normalised anyway. Positive = too much energy
%                  in that band.
%   S.rmsBandErrDb    signed error per band, mean over the microphones (bias)
%   S.rmsBandAbsErrDb |error| per band, mean over the microphones
%   S.rmsErrMeanDb    |error|, mean over microphones and bands
%   S.rmsOmniBandErrDb, S.rmsOmniErrMeanDb  the paper's definition (Fig. 8),
%                  the same shape error on the omni CH channel against the
%                  CH-encoded reference. Kept for comparison only: on this
%                  7-mic array, under movement, the omni channel of the RLS
%                  estimate and of the separately regularised encoding
%                  disagree where the microphone signals agree.
x = real(rir(:,1));
[rt, fAx, iEnd] = getT20Ita(x, P.fs, 1, [125 8000]);
e = 100*(rt(:).' - P.rtRef)./P.rtRef;
S.fAx = fAx(:).'; S.rt = rt(:).'; S.rtRef = P.rtRef;
S.bandErr = e;
S.rtErrMedian = median(abs(e),'omitnan');
S.rtErrMean = mean(abs(e),'omitnan');
S.rtErrRms = sqrt(mean(e.^2,'omitnan'));
S.rtErrSigned = median(e,'omitnan');
[~,pk] = max(abs(x)); S.peakMs = pk/P.fs*1000;
% direct path: the largest peak within +/-4 ms of where the source placement
% says it must be (P.directPathSmp), which is a real check rather than a
% search over the whole response
% The expected arrival differs by condition: the INFORMED RIR is referenced
% to sourceSig, whose direct path was placed at pre_delay (so ~9.3 ms after
% the ATF onset is removed), while the BLIND RIR is referenced to a pseudo-
% reference derived from the UNPADDED array signals, so its direct path sits
% at the pre-pad (5 ms). Callers pass the right one; defaulting to the
% informed position mis-centres the window on blind estimates and clips the
% true peak, which is what a first version of this check did.
if nargin < 3 || isempty(expectSmp); expectSmp = P.directPathSmp; end
halfWin = round(0.004*P.fs);
lo = max(1, expectSmp - halfWin);
hi = min(numel(x), expectSmp + halfWin);
[vD, iD] = max(abs(x(lo:hi)));
S.directMs = (lo + iD - 1)/P.fs*1000;
S.directRelMaxDb = 20*log10(vD/max(abs(x)));
t0 = NaN;
for w = drrWindows()
    try
        [dv, t0] = getDrr(x, P.fs, w.preMs, w.postMs, w.bandHz); v = dv(1);
    catch
        v = NaN;
    end
    S.(['drrDb' w.suffix]) = v;
    S.(['drrErr' w.suffix]) = v - P.(['drrRef' w.suffix]);
end
S.drrErrOmni = S.drrDb - P.drrRefOmni;

% tonal balance at the microphones: decode the estimate to what each
% microphone should record at the reference pose and compare each one's
% spectral shape with the measured one
if isempty(P.revRefMics)
    E = NaN(P.numMics, numel(fAx));
else
    E = micRevShape(chDecodeToMics(P, rir), P.fs, fAx) - P.revRefMics;
end
S.rmsMicErrDb = E;
S.rmsBandErrDb = mean(E, 1, 'omitnan');
S.rmsBandAbsErrDb = mean(abs(E), 1, 'omitnan');
S.rmsErrMeanDb = mean(abs(E), 'all', 'omitnan');

% the paper's definition, on the omni channel: kept for comparison only
L = revBandLevels(x, P.fs, t0, iEnd, fAx);
if isempty(P.revRefDb)
    d = NaN(size(L));
else
    d = L - P.revRefDb; d = d - mean(d, 'omitnan');
end
S.rmsOmniBandErrDb = d;
S.rmsOmniErrMeanDb = mean(abs(d), 'omitnan');
end
