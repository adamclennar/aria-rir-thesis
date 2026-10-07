function y = getWpe(P, wpeParams, tag, cacheDir)
% Run (or load) multichannel WPE. This is the expensive stage -- roughly 8 min
% on a 60 s take -- so every sweep caches it keyed on the parameter set.
arguments
    P struct
    wpeParams struct
    tag (1,:) char = 'published'
    cacheDir (1,:) char = ''
end
if isempty(cacheDir)
    y = runIt(P, wpeParams, tag);
    return
end
if ~exist(cacheDir,'dir'); mkdir(cacheDir); end
f = fullfile(cacheDir, sprintf('%s_wpe_%s.mat', P.takeName, tag));
% Fingerprint of the signal the cache was computed FROM. A length check alone
% is not enough: every take is 66 s, so a cache computed from a differently
% trimmed arraySigs (an earlier build of the same take) passes it and is used
% silently. That leaves the informed conditions correct and the blind ones
% quietly wrong, because only the blind path goes through WPE.
fp = sigFingerprint(P.d.arraySigs);
if exist(f,'file')
    L = load(f);
    % EXACT match. The fingerprint is computed the same way from the same
    % stored doubles, so identical signals always match exactly. (A relative
    % tolerance scaled by max(abs(fp)) was used here before; the first entry of
    % fp is the sample count, ~3e6, which made the tolerance ~3 -- larger than
    % any change a re-trim makes to the sum, the energy or the end samples --
    % so a cache from a differently trimmed take of the same length passed.)
    if ~isfield(L,'fp') || ~isequal(size(L.y), size(P.d.arraySigs)) || ~isequal(L.fp, fp)
        fprintf('[%s] WPE(%s) cache is STALE (built from different array signals), recomputing\n', ...
            P.takeName, tag);
    else
        fprintf('[%s] WPE(%s) loaded from cache\n', P.takeName, tag);
        y = L.y;
        return
    end
end
y = runIt(P, wpeParams, tag);
save(f,'y','wpeParams','fp','-v7.3');
end

function fp = sigFingerprint(x)
% Cheap, deterministic summary of a signal: shape, energy, and its ends.
fp = [size(x,1), size(x,2), sum(x(:)), sum(x(:).^2), x(1,:), x(end,:)];
end

function y = runIt(P, wpeParams, tag)
fprintf('[%s] WPE(%s) predOrder x%.2f delay %d ms iters %d ...', P.takeName, tag, ...
    wpeParams.predOrderScale, wpeParams.predictionDelayMs, wpeParams.numIterations);
t0 = tic;
y = adaptiveGwpeDerev(P.d.arraySigs, wpeParams);
n = size(P.d.arraySigs,1);
if size(y,1) < n
    y(end+1:n,:) = 0;      % istft can return a slightly shorter signal
elseif size(y,1) > n
    y = y(1:n,:);
end
fprintf(' %.1f min\n', toc(t0)/60);
end
