function ensurePaths()
% Put the pinned upstream code (Deppisch et al., external/room-acoustic-matching,
% unmodified), the pinned ITA toolbox and the vendored STFT on the path, and
% CHECK that those copies are the ones MATLAB resolves -- a startup.m that
% adds another ITA toolbox or another stft would otherwise win silently.
%
% The pipeline calls stft(x, win, hop, nfft, fs), the third-party signature,
% so that stft MUST shadow MATLAB's own (which reads a bare numeric argument
% as a sample rate and errors). addpath prepends, so the copies added here win.
%
% The pipeline's own code: the entry points in pipeline/ itself, and the
% helpers in core/ (the paper's method on a take), scoring/ (the metrics) and
% diagnostics/ (the front-end and geometry diagnostics). Entry points call
% ensurePaths before any helper, so 'cd pipeline; runTake(...)' still works.
%
% upstream_modified/ holds the one modified upstream function (the beamformer
% steered along a DoA trajectory), used only by the diagnostics; see the
% README there.
persistent done
if ~isempty(done); return; end
here = fileparts(mfilename('fullpath'));
root = fileparts(here);
ext = fullfile(root, 'external');
addpath(genpath(fullfile(ext, 'ita-toolbox')));
addpath(fullfile(ext, 'room-acoustic-matching', 'code', 'lib'));
addpath(fullfile(root, 'thirdparty', 'stft'));
addpath(fullfile(here, 'upstream_modified'));
addpath(fullfile(here, 'core'), fullfile(here, 'scoring'), fullfile(here, 'diagnostics'));
addpath(here);
expect = {'stft', fullfile(root,'thirdparty','stft'); ...
          'fdChRlsRotTransAtf', fullfile(ext,'room-acoustic-matching'); ...
          'ita_roomacoustics', fullfile(ext,'ita-toolbox')};
for k = 1:size(expect,1)
    w = which(expect{k,1});
    if ~startsWith(w, expect{k,2})
        error('ensurePaths:shadow', '%s resolves to %s, expected under %s', expect{k,1}, w, expect{k,2});
    end
end
warning('off','all');
done = true;
end
