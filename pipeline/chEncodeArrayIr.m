function xCh = chEncodeArrayIr(P, IRs)
% Put a measured 7-channel array IR into the CH domain the estimates live in.
%
% P.D is the array matrix the RLS itself uses (numFreqs x numMics x numChs):
% mics = D * ch. So the measured IR is encoded by solving that per frequency,
% regularised as the encoder is. Returns the full CH IR; channel 1 is the
% omnidirectional component the metrics are computed on.
%
% Note this is NOT getChdArrayEncDecAtf's EncShort, which is a time-domain
% filter -- using it as a frequency-domain matrix gives a flat T20 and a
% -48 dB DRR, i.e. every band failing.
fftLen = P.blockLenSmp;
numPos = fftLen/2+1;
numChs = 2*P.chOrder+1;
irMic = zeros(fftLen, size(IRs,2));
n = min(size(IRs,1), fftLen);
irMic(1:n,:) = IRs(1:n,:);
IRf = fft(irMic);
chRef = zeros(numPos, numChs);
regul = 1e-2;
for ff = 1:numPos
    Dm = squeeze(P.D(ff,:,:));                       % numMics x numChs
    chRef(ff,:) = ((Dm'*Dm + regul*trace(Dm'*Dm)/numChs*eye(numChs)) \ (Dm' * IRf(ff,:).')).';
end
xCh = real(ifft(getChFreqDomainConjugate(chRef)));
end
