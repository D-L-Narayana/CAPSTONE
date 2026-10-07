# Monte-Carlo parameter study

200 random nodes per setting, uniform in the 60 x 60 x 20 m field; Gaussian ranging noise; anchors 1-4 are the fixed non-coplanar corners of the baseline scenario, extra anchors are random. Base setting: sigma = 0.5 m, 4 anchors, N = 20, T = 60. Both PSO variants clip particles to the field; LSQ is the linearised trilateration solve, LSQ + Gauss-Newton refines it (<= 10 steps).

## Ranging noise sigma [m]

| value | method | RMSE [m] | mean err [m] | p90 err [m] | fitness evals | runtime [ms] | memory [floats] |
|---|---|---|---|---|---|---|---|
| 0.1 | Simplified PSO | 7.856 | 5.486 | 14.378 | 1200 | 3.192 | 120 |
| 0.1 | Standard PSO | 5.878 | 2.646 | 11.769 | 1220 | 3.843 | 183 |
| 0.1 | Least-squares trilateration | 1.172 | 0.984 | 1.967 | 0 | 0.204 | 15 |
| 0.1 | LSQ + Gauss-Newton | 0.621 | 0.426 | 1.003 | 7 | 3.742 | 15 |
| 0.25 | Simplified PSO | 8.070 | 5.839 | 14.458 | 1200 | 3.921 | 120 |
| 0.25 | Standard PSO | 6.059 | 3.167 | 11.791 | 1220 | 4.034 | 183 |
| 0.25 | Least-squares trilateration | 2.931 | 2.461 | 4.920 | 0 | 0.052 | 15 |
| 0.25 | LSQ + Gauss-Newton | 1.935 | 1.225 | 2.624 | 8 | 0.578 | 15 |
| 0.5 | Simplified PSO | 8.379 | 6.253 | 14.651 | 1200 | 3.240 | 120 |
| 0.5 | Standard PSO | 6.533 | 3.983 | 12.931 | 1220 | 15.474 | 183 |
| 0.5 | Least-squares trilateration | 5.860 | 4.920 | 9.850 | 0 | 0.082 | 15 |
| 0.5 | LSQ + Gauss-Newton | 4.174 | 2.678 | 6.526 | 8 | 0.771 | 15 |
| 1 | Simplified PSO | 8.555 | 6.705 | 14.687 | 1200 | 4.246 | 120 |
| 1 | Standard PSO | 6.978 | 4.921 | 13.205 | 1220 | 3.825 | 183 |
| 1 | Least-squares trilateration | 11.717 | 9.834 | 19.716 | 0 | 0.038 | 15 |
| 1 | LSQ + Gauss-Newton | 7.366 | 4.971 | 10.590 | 9 | 0.622 | 15 |
| 2 | Simplified PSO | 9.324 | 7.750 | 15.231 | 1200 | 2.950 | 120 |
| 2 | Standard PSO | 7.772 | 6.126 | 14.203 | 1220 | 14.231 | 183 |
| 2 | Least-squares trilateration | 23.424 | 19.648 | 39.043 | 0 | 0.053 | 15 |
| 2 | LSQ + Gauss-Newton | 11.187 | 8.374 | 18.656 | 9 | 0.641 | 15 |

## Number of anchors

| value | method | RMSE [m] | mean err [m] | p90 err [m] | fitness evals | runtime [ms] | memory [floats] |
|---|---|---|---|---|---|---|---|
| 4 | Simplified PSO | 8.379 | 6.253 | 14.651 | 1200 | 3.342 | 120 |
| 4 | Standard PSO | 6.533 | 3.983 | 12.931 | 1220 | 4.131 | 183 |
| 4 | Least-squares trilateration | 5.860 | 4.920 | 9.850 | 0 | 0.052 | 15 |
| 4 | LSQ + Gauss-Newton | 4.174 | 2.678 | 6.526 | 8 | 1.035 | 15 |
| 5 | Simplified PSO | 7.693 | 5.282 | 14.577 | 1200 | 7.396 | 120 |
| 5 | Standard PSO | 5.683 | 3.110 | 8.000 | 1220 | 10.051 | 183 |
| 5 | Least-squares trilateration | 4.204 | 3.264 | 7.038 | 0 | 0.118 | 18 |
| 5 | LSQ + Gauss-Newton | 2.828 | 1.730 | 4.014 | 8 | 1.666 | 18 |
| 6 | Simplified PSO | 6.424 | 4.247 | 12.296 | 1200 | 6.579 | 120 |
| 6 | Standard PSO | 3.051 | 1.558 | 2.609 | 1220 | 27.407 | 183 |
| 6 | Least-squares trilateration | 3.066 | 2.333 | 4.869 | 0 | 0.291 | 21 |
| 6 | LSQ + Gauss-Newton | 2.124 | 1.436 | 2.825 | 8 | 1.691 | 21 |
| 8 | Simplified PSO | 5.804 | 3.806 | 10.445 | 1200 | 11.915 | 120 |
| 8 | Standard PSO | 2.698 | 1.276 | 1.544 | 1220 | 5.419 | 183 |
| 8 | Least-squares trilateration | 2.136 | 1.672 | 3.140 | 0 | 0.035 | 27 |
| 8 | LSQ + Gauss-Newton | 1.147 | 0.892 | 1.876 | 8 | 0.381 | 27 |
| 10 | Simplified PSO | 4.947 | 3.026 | 8.056 | 1200 | 3.365 | 120 |
| 10 | Standard PSO | 2.372 | 1.055 | 1.695 | 1220 | 6.942 | 183 |
| 10 | Least-squares trilateration | 1.609 | 1.369 | 2.627 | 0 | 0.534 | 33 |
| 10 | LSQ + Gauss-Newton | 0.835 | 0.705 | 1.302 | 8 | 3.602 | 33 |

## Swarm size N

| value | method | RMSE [m] | mean err [m] | p90 err [m] | fitness evals | runtime [ms] | memory [floats] |
|---|---|---|---|---|---|---|---|
| 5 | Simplified PSO | 10.205 | 8.179 | 17.581 | 300 | 3.791 | 30 |
| 5 | Standard PSO | 7.081 | 4.974 | 14.115 | 305 | 2.189 | 48 |
| 10 | Simplified PSO | 9.136 | 7.091 | 16.950 | 600 | 1.700 | 60 |
| 10 | Standard PSO | 6.785 | 4.317 | 13.589 | 610 | 2.101 | 93 |
| 20 | Simplified PSO | 8.379 | 6.253 | 14.651 | 1200 | 1.765 | 120 |
| 20 | Standard PSO | 6.533 | 3.983 | 12.931 | 1220 | 2.269 | 183 |
| 40 | Simplified PSO | 7.027 | 4.794 | 14.312 | 2400 | 2.359 | 240 |
| 40 | Standard PSO | 6.133 | 3.833 | 11.786 | 2440 | 3.050 | 363 |
| 80 | Simplified PSO | 6.370 | 4.282 | 12.064 | 4800 | 2.996 | 480 |
| 80 | Standard PSO | 5.160 | 3.044 | 7.228 | 4880 | 10.834 | 723 |

## Iterations T

| value | method | RMSE [m] | mean err [m] | p90 err [m] | fitness evals | runtime [ms] | memory [floats] |
|---|---|---|---|---|---|---|---|
| 10 | Simplified PSO | 8.755 | 7.088 | 14.441 | 200 | 0.342 | 120 |
| 10 | Standard PSO | 7.064 | 4.948 | 11.984 | 220 | 0.462 | 183 |
| 20 | Simplified PSO | 8.532 | 6.574 | 14.698 | 400 | 0.942 | 120 |
| 20 | Standard PSO | 6.790 | 4.391 | 12.320 | 420 | 0.998 | 183 |
| 30 | Simplified PSO | 8.383 | 6.275 | 14.653 | 600 | 1.172 | 120 |
| 30 | Standard PSO | 6.616 | 4.104 | 13.046 | 620 | 1.059 | 183 |
| 60 | Simplified PSO | 8.379 | 6.253 | 14.651 | 1200 | 1.401 | 120 |
| 60 | Standard PSO | 6.533 | 3.983 | 12.931 | 1220 | 2.362 | 183 |
| 120 | Simplified PSO | 8.381 | 6.254 | 14.653 | 2400 | 6.872 | 120 |
| 120 | Standard PSO | 6.505 | 3.951 | 12.934 | 2420 | 4.361 | 183 |

## Error CDF (sigma = 0.5 m, 4 anchors)

| method | RMSE [m] | median [m] | p90 [m] | fitness evals | memory [floats] |
|---|---|---|---|---|---|
| Simplified PSO (20 x 60) | 8.379 | 4.660 | 14.651 | 1200 | 120 |
| Standard PSO (20 x 60) | 6.533 | 1.674 | 12.931 | 1220 | 183 |
| Least-squares trilateration | 5.860 | 4.478 | 9.850 | 0 | 15 |
| LSQ + Gauss-Newton | 4.174 | 1.479 | 6.526 | 8 | 15 |
