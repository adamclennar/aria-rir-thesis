function exportInputs(takeName, opts)
% Write the signals an external dereverberator needs, as 48 kHz 32-bit
% float WAVs in results/dereverb/inputs/:
%
%   <take>_array.wav     7 ch, the raw array signals (P.d.arraySigs). A
%                        PRE-beamformer candidate replaces WPE: return 7 ch,
%                        same length, same start.
%   <take>_ref_wpe.wav   1 ch, today's pseudo-reference: WPE + MPDR steered at
%                        the protocol bearing (addOracleDoa). A POST-beamformer
%                        candidate dereverberates this: return 1 ch, same
%                        length, same start.
%   <take>_ref_none.wav  1 ch, the same beamformer on the raw array signals
%                        (no WPE), for post-filters that should replace WPE
%                        rather than follow it.
%   <take>_received.wav  1 ch, only with opts.received: the source as it
%                        reaches the moving array (receivedDirect), i.e. the
%                        perfect front end's output. INFORMED: for oracle
%                        candidates that measure what the reference contains;
%                        a blind candidate must not read it.
%
% opts.array / opts.refNone (default true) skip the 7-channel and no-WPE files
% when a candidate does not need them.
%
% Every file is peak-normalised to 0.9 (the metrics are gain-invariant); the
% gains are in <take>_gains.json. Then score with
%   evalDereverb('<take>', 'results/dereverb/candidates/<take>_<label>.wav')
arguments
    takeName (1,:) char
    opts.array (1,1) logical = true
    opts.refNone (1,1) logical = true
    opts.received (1,1) logical = false
end
ensurePaths();                                     % helpers live in subfolders
P = addOracleDoa(prepAriaTake(takeName));         % WPE from cache
if ~P.hasOracle
    error('exportInputs:noGeometry', '%s has no protocol bearing (addOracleDoa) to steer at', takeName);
end
d = fullfile(P.root, 'results', 'dereverb', 'inputs');
if ~exist(d, 'dir'); mkdir(d); end
traj = P.oracleTraj;
g = struct();
if opts.array;   g.array    = put(fullfile(d, [takeName '_array.wav']), P.d.arraySigs, P.fs); end
g.ref_wpe = put(fullfile(d, [takeName '_ref_wpe.wav']), beamform(P, P.derev, traj), P.fs);
if opts.refNone; g.ref_none = put(fullfile(d, [takeName '_ref_none.wav']), beamform(P, P.d.arraySigs, traj), P.fs); end
if opts.received
    g.received = put(fullfile(d, [takeName '_received.wav']), receivedDirect(P, P.d.sourceSig(:,1)), P.fs);
end
fid = fopen(fullfile(d, [takeName '_gains.json']), 'w'); fwrite(fid, jsonencode(g)); fclose(fid);
fprintf('[%s] wrote inputs to %s\n', takeName, d);
end

function gain = put(f, x, fs)
gain = 0.9/max(abs(x(:)));
audiowrite(f, single(x*gain), fs, 'BitsPerSample', 32);
end
