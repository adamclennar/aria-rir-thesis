# Modified upstream code

`musicAndBeamformTimeDomainAdaptiveTraj.m` is a modified copy of
`musicAndBeamformTimeDomainAdaptive.m` from Deppisch et al.'s
room-acoustic-matching repository (`external/room-acoustic-matching`, commit
`76c6ea3`). The only change: the beamformer can be steered along a per-sample
DoA trajectory (the true source bearing, or the world-frame position fit),
where upstream accepts MUSIC or one constant direction. With MUSIC or a
constant direction its output is upstream's. Modified 2026-09-26, copied here
2026-10-11.

The baseline reproduction does not use it (`bfEstimate.m` calls the upstream
function). Only the diagnostics that steer along a trajectory do
(`beamform.m`).

This file is used and distributed under the Chalmers University of Technology
Non-Commercial Software License Agreement (`external/room-acoustic-matching/LICENSE`), for non-commercial
academic purposes only, under the same terms as the original. Section 5 of
that agreement:

```
5.  Disclaimers.
THE SOFTWARE IS PROVIDED "AS IS," WITH ALL FAULTS, DEFECTS AND ERRORS, AND WITHOUT WARRANTY OF ANY KIND. LICENSOR
DISCLAIMS ALL WARRANTIES (EXPRESS OR IMPLIED AND ARISING BY LAW OR OTHERWISE) REGARDING THE SOFTWARE AND ITS
PERFORMANCE OR SUITABILITY FOR YOUR INTENDED USE, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTY OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, OR NONINFRINGEMENT. LICENSOR SHALL HAVE NO LIABILITY OF ANY KIND FOR THE USE OF OR
INABILITY TO USE THE SOFTWARE OR FOR ANY LOSS OF DATA.
NEITHER LICENSOR NOR ANY CONTRIBUTOR TO THE SOFTWARE WILL BE LIABLE FOR ANY DAMAGES RELATED TO THE SOFTWARE OR THIS
AGREEMENT, INCLUDING DIRECT, INDIRECT, SPECIAL, CONSEQUENTIAL OR INCIDENTAL DAMAGES, TO THE MAXIMUM EXTENT THE LAW
PERMITS, NO MATTER WHAT LEGAL THEORY IT IS BASED ON. ALSO, YOU MUST PASS THIS LIMITATION OF LIABILITY ON WHENEVER YOU
DISTRIBUTE THE SOFTWARE OR DERIVATIVE WORKS.
We have no duty of reasonable care or lack of negligence, and we are not obligated to (and will not) provide technical
support for the Software. If you breach this License or if you sue anyone over any intellectual property regarding the
Software, or another party’s use thereof, this License and your rights herein shall terminate automatically.  You will
immediately cease all use and distribution of the Software and destroy any copies or portions of the Software in your
possession.
```
