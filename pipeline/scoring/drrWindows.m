function W = drrWindows()
% The direct-sound windows DRR is scored under, in one place so the
% estimates and the references can never disagree on the definition.
% Every scoring pass computes DRR under ALL of them.
%
%   repo   peak -1 to +2.5 ms  the wide window: this repository's primary definition
%   paper  peak -0.5 to +1 ms  Deppisch et al. (2026), Sec. 5.2
%
% Both are otherwise upstream getDrr: 4th-order 100-8000 Hz band-pass,
% direct peak = earliest of the 3 largest peaks in the first 20 ms, and the
% reverberant part runs from the end of the direct window to the Lundeby
% point. The peak and the Lundeby point do not depend on the window, so the
% definitions differ only in which samples count as direct.
%
% Why two. Use the paper's window wherever a number sits next to the paper's
% (Table 2). The wide window stays primary for this data's own comparisons:
% in the working repository's dose-response study, under the paper's window
% even a perfect pseudo-reference scored ~2.3 dB median DRR error on the
% walking takes (0.9 dB under the wide window), because the estimates'
% direct sound is spread over a few ms differently from the reference's,
% and that spread then swamps the differences between front ends.
%
%   suffix  appended to the field names: drrDb (wide), drrDbPaper (paper's)
W = struct('name',   {'repo', 'paper'}, ...
           'preMs',  {1,      0.5}, ...
           'postMs', {2.5,    1}, ...
           'suffix', {'',     'Paper'}, ...
           'bandHz', {[100 8000], [100 8000]});
end
