function S = scoreRefSig(P, pref, opts)
% Quality of the pseudo-reference signal: magnitude-squared coherence with the
% true source signal, plus a motion-
% compensated variant for walking takes.
%
% Coherence is invariant to any fixed linear filter, gain and delay, so it
% measures what is left over: reverberation, steering error and noise. The
% beamformer output is the source seen through the array response of whatever
% direction it steered at, so a waveform error would penalise a clean estimate
% simply for being filtered.
%
%   refCoh      mean coherence over `band`, 1 = perfect
%   refErrDb    10*log10(1-refCoh), residual power fraction in dB
%   refCohComp  the same after warping sourceSig by the tracked change in
%               source-array range (needs P.srcRangeM from addOracleDoa). The
%               range must be from the ACOUSTIC centre (acousticCentre), not
%               the tracked CPF: from the CPF, head turns alone predict a
%               15-sample delay swing the signal does not have, and the warp
%               made coherence WORSE (office headMove 0.40 -> 0.26).
%
% Why the compensated version. The pseudo-reference is the direct sound AS
% received at the moving array, so it carries the propagation delay r(t)/c.
% Walking 1 m changes that delay by ~3 ms, which rotates the cross-spectrum
% phase by tens of radians over the take at 1 kHz; Welch averaging then
% cancels it and the plain refCoh of every walking take collapses to ~0.07
% however good the beamformer is. Warping the source by the known geometric
% delay removes that, leaving what the DoA and dereverberation stages control.
% Within one take both versions rank conditions fairly (the delay is common to
% all conditions); across takes only refCohComp is comparable.
%
% Ceiling: even a perfect beamformer output is the source through a slowly
% time-varying array response, so neither reaches 1. Compare conditions.
%
% Coherence alone is not enough to rank references for the RLS: it is blind
% to WHICH linear filter relates the reference to the source, and the RLS
% divides that filter out of its estimate. A filter that drifts over the take
% (steering hopping between ATFs, adaptive beamformer weights) makes the RLS
% average inconsistent references while barely moving coherence. Hence:
%
%   refCohBands   [1 x 7] coherence per octave (125 Hz .. 8 kHz), from the
%                 motion-compensated comparison when available. The broadband
%                 mean is ~95% bins above 500 Hz; the RT metric weights the
%                 octaves equally.
%   refFiltVarDb  median over octaves of the std across 8 s chunks (4 s hop,
%                 speech span only) of the band level of the reference's
%                 transfer function from the source -- 0 dB for a reference
%                 whose filter never changes. Measured against the source as a
%                 perfect front end would deliver it (receivedDirect: range
%                 delay AND 1/r level), so walking itself does not count.
%   refFiltVarBands [1 x 7] the same per octave.
%   refCohEff     the coherence that applies to THIS reference: refCohComp if
%                 it carries the array's motion (a beamformer output, or a
%                 reference built from receivedDirect), refCoh if it does not
%                 (sourceSig, the SM57 at the loudspeaker). refCohBands and
%                 refFiltVar* are computed against the same source model.
%                 Comparing a reference against the wrong model is not subtle:
%                 the SM57 scores 0.97 plain and 0.09 motion-compensated.
%
% Coherence also cannot see residual reverberation that is a fixed filtered
% copy of the source -- which is what dereverberation removes. On a walking
% office take WPE halved the motion-aware T20 error while moving coherence
% from 0.186 to 0.198. So the reference's own impulse response from the
% source is estimated too (long-window spectral division against the same
% source model, 100-8000 Hz, 2.7 s window, acausal part used as the noise
% floor), and its tail measured against its direct path:
%
%   refEarlyDb    energy 2.5-20 ms after the direct peak, dB re the direct
%                 path (-1..+2.5 ms, the pipeline's DRR window)
%   refLateDb     energy 20 ms - 1 s after it, dB re the direct path, with
%                 the estimate's noise floor subtracted
%   refTailDb     both together; -refTailDb is the reference's own DRR
%   refFloorDb    the noise floor's energy over the late window, same units:
%                 refLateDb cannot be resolved much below it
arguments
    P struct
    pref (:,1) double
    opts.band (1,2) double = [100 8000]
    opts.carriesMotion (1,1) logical = true
end
band = opts.band;
s = P.d.sourceSig(:,1);
[S.refCoh, S.lagSmp, cxyPlain, f] = coh(P, s, pref, band);
S.refErrDb = 10*log10(max(1 - S.refCoh, eps));
S.band = band;
S.refCohComp = NaN;
S.fBands = [125 250 500 1000 2000 4000 8000];
S.refFiltVarDb = NaN; S.refFiltVarBands = nan(1,7);
S.carriesMotion = opts.carriesMotion;
cxy = cxyPlain; srcModel = s;
if isfield(P, 'srcRangeM') && ~isempty(P.srcRangeM)
    c = 343;
    n = numel(s);
    r = P.srcRangeM(1:n);
    dly = (r - median(r))/c*P.fs;            % samples, relative to the take's median range
    sw = interp1((1:n)', s, (1:n)' - dly, 'spline', 0);
    [S.refCohComp, ~, cxyComp] = coh(P, sw, pref, band);
    S.rangeSpanM = max(r) - min(r);
    if opts.carriesMotion
        cxy = cxyComp; srcModel = receivedDirect(P, s);
    end
end
if opts.carriesMotion && ~isnan(S.refCohComp); S.refCohEff = S.refCohComp; else; S.refCohEff = S.refCoh; end
if ~opts.carriesMotion || isfield(P, 'srcRangeM')
    [S.refFiltVarDb, S.refFiltVarBands] = filtVar(P, srcModel, pref, S.fBands);
    [S.refEarlyDb, S.refLateDb, S.refTailDb, S.refFloorDb] = refIr(P, srcModel, pref, band);
else
    S.refEarlyDb = NaN; S.refLateDb = NaN; S.refTailDb = NaN; S.refFloorDb = NaN;
end
S.refCohBands = arrayfun(@(fc) mean(cxy(f >= fc/sqrt(2) & f < min(fc*sqrt(2), P.fs/2))), S.fBands);
end

function [early, late, tail, floorDb] = refIr(P, x, pref, band)
% Impulse response from the source model x to the reference, and the energy
% in its tail relative to its direct path.
fs = P.fs;
[x, y] = alignBulk(P, x, pref);
nfft = 2^17;                                       % 2.7 s: longer than any band's RT
[H, f] = tfestimate(x, y, hann(nfft,'periodic'), nfft/2, nfft, fs);
w = zeros(size(f));                                % raised-cosine band-pass, 1/2-octave edges
inb = f >= band(1) & f <= band(2); w(inb) = 1;
lo = f >= band(1)/sqrt(2) & f < band(1); w(lo) = 0.5 - 0.5*cos(pi*log2(f(lo)/(band(1)/sqrt(2)))/0.5);
hi = f > band(2) & f <= min(band(2)*sqrt(2), fs/2); w(hi) = 0.5 + 0.5*cos(pi*log2(f(hi)/band(2))/0.5);
H = H.*w;
h = real(ifft([H; conj(flipud(H(2:end-1)))]));
h = circshift(h, nfft/2);                           % lag 0 at the centre
c0 = nfft/2 + 1;
win = c0 + (-round(0.01*fs):round(0.01*fs));       % the bulk lag is removed: direct within +-10 ms
[~, i] = max(abs(h(win))); pk = win(1) + i - 1;
e = @(a, b) sum(h(pk + (round(a*fs):round(b*fs))).^2);
Ed = e(-0.001, 0.0025);
Ee = e(0.0025, 0.020);
lateLen = round(0.98*fs);
El = e(0.020, 1.0);
% noise floor: the true response is causal, so the acausal part (-1.2..-0.2 s)
% is estimation noise; scale its energy density to the late window
acaus = h(pk - round(1.2*fs) : pk - round(0.2*fs));
Ef = mean(acaus.^2)*lateLen;
late = 10*log10(max(El - Ef, Ef*1e-2)/Ed);
early = 10*log10(Ee/Ed);
tail = 10*log10((Ee + max(El - Ef, 0))/Ed);
floorDb = 10*log10(Ef/Ed);
end

function [x, y] = alignBulk(P, x, y)
n = min(numel(x), numel(y));
x = x(1:n); y = y(1:n); y(~isfinite(y)) = 0;
[c,lags] = xcorr(y - mean(y), x - mean(x), round(0.05*P.fs), 'normalized');
[~,ix] = max(abs(c)); lag = lags(ix);
if lag > 0
    y = y(lag+1:end); x = x(1:numel(y));
elseif lag < 0
    x = x(-lag+1:end); y = y(1:numel(x));
end
end

function [v, vb] = filtVar(P, b, pref, fBands)
% Spread of the reference's per-octave level relative to b over 8 s chunks.
n = min(numel(b), numel(pref));
b = b(1:n); y = pref(1:n); y(~isfinite(y)) = 0;
[c,lags] = xcorr(y - mean(y), b - mean(b), round(0.05*P.fs), 'normalized');
[~,ix] = max(abs(c)); lag = lags(ix);
if lag > 0
    y = y(lag+1:end); b = b(1:numel(y));
elseif lag < 0
    b = b(-lag+1:end); y = y(1:numel(b));
end
win = round(8*P.fs); hop = round(4*P.fs); nfft = 8192;
sp = round(P.speechSpanS*P.fs);
starts = max(1, sp(1)) : hop : min(numel(y), sp(2)) - win;
L = nan(numel(starts), numel(fBands));
for k = 1:numel(starts)
    i = starts(k):starts(k)+win-1;
    [H, f] = tfestimate(b(i), y(i), hann(nfft,'periodic'), nfft/2, nfft, P.fs);
    for j = 1:numel(fBands)
        sel = f >= fBands(j)/sqrt(2) & f < min(fBands(j)*sqrt(2), P.fs/2);
        L(k,j) = 10*log10(mean(abs(H(sel)).^2));
    end
end
vb = std(L, 0, 1, 'omitnan');
v = median(vb, 'omitnan');
end

function [m, lag, cxy, f] = coh(P, s, pref, band)
n = min(numel(s), numel(pref));
s = s(1:n); y = pref(1:n);
y(~isfinite(y)) = 0;
% Remove the bulk delay so the coherence estimate is not smeared across Welch
% segments.
maxLag = round(0.05*P.fs);
[c,lags] = xcorr(y - mean(y), s - mean(s), maxLag, 'normalized');
[~,ix] = max(abs(c));
lag = lags(ix);
if lag > 0
    y = y(lag+1:end); s = s(1:numel(y));
elseif lag < 0
    s = s(-lag+1:end); y = y(1:numel(s));
end
% The Welch window must be comfortably longer than the response relating the
% two signals, or a perfectly linear relation still scores below 1.
nfft = 8192;
[cxy, f] = mscohere(y, s, hann(nfft,'periodic'), nfft/2, nfft, P.fs);
sel = f >= band(1) & f <= band(2);
m = mean(cxy(sel), 'omitnan');
end
