# Multi-node iterative auto-localization (3D)

Iterative scheme (Kulkarni et al. 2009, Section IV, in 3-D): every unknown node that has at least 4 references (beacons or already-settled nodes) within radio range runs the single-node localizer on its noisy ranges to the 6 nearest references; settled nodes serve as references (through their *estimated* positions) in the next round, until no new node settles.

| setting | value |
|---|---|
| localizer | `std` (20 particles x 60 iterations) |
| noise | `gaussian:0.5` |
| field [m] | 100 x 100 x 30 |
| nodes / beacons / radio range | 50 / 10 / 50 m |
| min / max references | 4 / 6 (non-coplanar reference sets only) |
| max rounds | 10 |
| seed | 1 |

## Results

| metric | value |
|---|---|
| nodes N | 50 |
| beacons M | 10 |
| radio range r [m] | 50 |
| field [m] | 100 x 100 x 30 |
| localizer | StandardPSO |
| noise | gaussian(0.5) |
| rounds run | 2 |
| localized nodes N_L | 50 |
| not localized N_NL | 0 |
| coverage (settled / N) | 100.0 % |
| E_l = mean squared error over localized nodes [m^2] | 32.932 |
| RMSE over localized nodes [m] | 5.739 |
| total fitness evaluations | 61000 |
| runtime [s] | 0.205 |

| round | newly settled | settled total | mean references in range | evaluations |
|---|---|---|---|---|
| 1 | 37 | 37 | 4.54 | 45140 |
| 2 | 13 | 50 | 13.85 | 15860 |

Metrics: N_NL = nodes not localized, E_l = mean squared error over the localized nodes (eq. 4 of the paper with the z term), RMSE = sqrt(E_l), coverage = settled / N. Ranges are simulated from the true positions; errors of settled nodes propagate to the nodes that use them as references, so later rounds are usually less accurate than the first one.
