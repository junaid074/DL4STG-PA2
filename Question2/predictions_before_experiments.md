# Predictions written before the corresponding experiments
# (Assignment requires these unedited in the report; commit before running those sections.)

## Q2.5 — before Testing the Baseline (mirror / rotation / speed)

I am writing this before any mirror, rotation, or speed-up test.

**Mirror.** I do not think model A will pass. The motion rule (straight lines + wall bounces) is the same after a flip, but the digits themselves look different, and nothing in training pushed the network to treat a mirrored 2 like a 2. So I expect a non-zero consistency gap and a clear rise in transformed error.

**Rotation.** Same story, probably worse. A 90° turn changes appearance more than a mirror, and if the training flow directions are not perfectly uniform around the circle, rotated trajectories may also look a bit unfamiliar. I expect CR and ER both higher than the mirror numbers.

**Speed.** Reach of model A looks like about 8 original pixels per time step (patch 4, two 3×3 ConvLSTMs on the folded grid). If a typical digit only moves a couple of pixels per frame at 1×, then 2× is maybe still inside that ballpark and 3× is where it should fall apart — blur, a digit that stalls, or one that gets left behind. Copy-last will look bad at every speed; A should beat it at 1× and maybe 2×, then collapse toward something useless at 3×.

## Q2.11 — before training B–D

Written before training B, C, or D.

**B (symmetry aug)** should mainly help the mirror/rotation failures — appearance-tied motion — so CM/CR and EM/ER should drop. It will not help speed.

**C (speed aug)** should help the unfamiliar-fast-motion story at 2×/3×, but only up to what the architecture can reach. If 3× is still beyond ~8 px/step, C will stay weak there.

**D (larger reach + speed aug)** should be the one that actually handles 3×, if the speed failure was architectural reach rather than just missing data. That is the comparison I care about: C bad at 3× and D good would point to reach, not data.
