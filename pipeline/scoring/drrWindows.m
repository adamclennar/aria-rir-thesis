function W = drrWindows()
% The direct-sound windows DRR is scored under, in one place so the
% estimates and the references can never disagree on the definition.
% Every scoring pass computes DRR under ALL of them.
%
%   repo   peak -1 to +2.5 ms  the wide window, kept in the summaries only
%   paper  peak -0.5 to +1 ms  Deppisch et al. (2026), Sec. 5.2: the one reported
%
% Both are otherwise upstream getDrr: 4th-order 100-8000 Hz band-pass,
% direct peak = earliest of the 3 largest peaks in the first 20 ms, and the
% reverberant part runs from the end of the direct window to the Lundeby
% point. The peak and the Lundeby point do not depend on the window, so the
% definitions differ only in which samples count as direct.
%
% Why two. The thesis reports DRR under the paper's window only, so every
% number is defined as the reference study defines it, comparisons with its
% Table 2 included. The wide window is kept beside it in the summaries: in
% the working repository's dose-response study, under the paper's window even
% a perfect pseudo-reference scored ~2.3 dB median DRR error on the walking
% takes (0.9 dB under the wide window), because the estimates' direct sound
% is spread over a few ms differently from the reference's. The wide window
% shows how much of a DRR error is that spread.
%
%   suffix  appended to the field names: drrDb (wide), drrDbPaper (paper's)
W = struct('name',   {'repo', 'paper'}, ...
           'preMs',  {1,      0.5}, ...
           'postMs', {2.5,    1}, ...
           'suffix', {'',     'Paper'}, ...
           'bandHz', {[100 8000], [100 8000]});
end
