function w = defaultWpeParams(fs, mods)
% The published WPE settings from estimateRirs.m, with optional modifiers so a
% sweep can move along a dereverberation-strength axis using settings a real
% algorithm could actually be run at.
%
%   mods.predOrderScale   multiply every prediction order (1 = published,
%                         0 = no prediction at all i.e. WPE disabled)
%   mods.predictionDelayMs  prediction delay (published 20 ms)
%   mods.numIterations    published 2
%   mods.regulWeight      published 1e-4
arguments
    fs (1,1)
    mods struct = struct()
end
blockLenStft = 2048; hopStft = 256;
w.blockLenCovMs = 1500; w.adaptiveCovInit = false;
w.predictionDelayMs = 20; w.regulWeight = 1e-4;
w.numIterations = 2; w.covSmoothingLen = 1;
w.blockLenStft = blockLenStft; w.win = sqrt(hann(blockLenStft,'periodic'));
w.hopSizeStft = hopStft; w.fftLen = blockLenStft; w.fs = fs;
w.overlapFactCov = 0.5;

predOrderScale = 1;
if isfield(mods,'predictionDelayMs'); w.predictionDelayMs = mods.predictionDelayMs; end
if isfield(mods,'numIterations');     w.numIterations     = mods.numIterations;     end
if isfield(mods,'regulWeight');       w.regulWeight       = mods.regulWeight;       end
if isfield(mods,'predOrderScale');    predOrderScale      = mods.predOrderScale;    end

fAxW = linspace(0,fs/2,blockLenStft/2+1)';
p = zeros(blockLenStft/2+1,1);
p(fAxW<500) = 32; p(fAxW>=500 & fAxW<1000) = 28;
p(fAxW>=1000 & fAxW<8000) = 24; p(fAxW>=8000) = 12;
w.predOrders = max(1, round(p * predOrderScale));   % order 1 is the weakest usable filter
w.predOrderScale = predOrderScale;

w.numBlocksCovEst = round((w.blockLenCovMs/1000*fs - (blockLenStft-hopStft))/hopStft);
w.predDelay = round(w.predictionDelayMs/1000/(hopStft/fs));
end
