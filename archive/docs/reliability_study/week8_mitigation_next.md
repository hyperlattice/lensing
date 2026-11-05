# Week 8 protocol record — targeted nuisance randomization

Week 7 was closed before this protocol was frozen. The Week 8 intervention addressed the largest supported synthetic penalties measured through Week 5: pixel scale, half exposure, and doubled read noise. Week 7 contributed no mitigation target because its independent SIE path reproduced the reference.

Protocol v1.8 froze a single training change: retain the 700 canonical training parents and add one deterministic randomized descendant per parent. Pixel scale was drawn from 0.04/0.05/0.06 arcsec per pixel with matching 320/256/214 native image shapes to preserve the measured angular field of view; exposure multiplier was drawn from 0.5/1.0, and read-noise multiplier from 1/2 using seed 6031. The 150 validation parents, three training seeds, architecture, and optimizer remained fixed.

The frozen evaluation covered canonical, 0.04, 0.06, half-exposure, and doubled-read-noise validation images. Predictions were saved before target joins, and paired changes used a 10,000-resample parent bootstrap.

The frozen rule passed: equal-weight stress MAE changed by -0.0290 arcsec (95% interval [-0.0351, -0.0231]), and canonical MAE changed by -0.0109 arcsec (upper bound -0.0031). The conditional scalar pixel-scale input extension was not run. Week 8 is closed; see [`week8_mitigation_status.md`](week8_mitigation_status.md) and [`week9_hybrid_next.md`](week9_hybrid_next.md) for results and the next phase.
