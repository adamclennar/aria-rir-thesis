function [pc, d] = acousticCentre(P)
% Horizontal position of the array's ACOUSTIC centre per sample, world frame.
%
% translationXYZCart tracks the Aria CPF (central pupil frame, between the
% eyes -- see build/pose_mps.py). The ATFs, and so the beamformer's delay
% reference and the CH expansion the motion-aware RLS assumes, are centred on
% the head: the HATS sat on the turntable axis during the CHiME measurement.
% The CPF sits in FRONT of that centre, so a pure head turn swings the tracked
% point along an arc and reads as translation that never happened.
%
% d was measured, not assumed: warping sourceSig by the source-to-array range
% change and maximising coherence with the pseudo-reference over d peaks at
% 0.08 m on all three takes tried (office headMove 0.26 -> 0.48, office
% fullMove 0.12 -> 0.24, meeting headMove 0.23 -> 0.44, d = 0 -> 0.08; 2 cm
% steps). That is the CPF-to-head-centre distance of an adult head.
%
% NOTE the RLS (prepAriaTake -> transAzi/transRad) still uses the CPF, as
% upstream estimateRirs.m does. Only the diagnostics that need the source
% geometry use the acoustic centre (addOracleDoa, musicWorldFrame).
d = 0.08;
p = P.d.translationXYZCart(:,1:2);
psi = P.rotYaw;
pc = p - d*[cos(psi), sin(psi)];
end
