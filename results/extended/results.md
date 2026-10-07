# Monte-Carlo parameter study

200 random nodes per setting, uniform in the 60 x 60 x 20 m field; Gaussian ranging noise; anchors 1-4 are the fixed non-coplanar corners of the baseline scenario, extra anchors are random. Base setting: sigma = 0.5 m, 4 anchors, N = 20, T = 60. Both PSO variants clip particles to the field; LSQ is the linearised trilateration solve, LSQ + Gauss-Newton refines it (<= 10 steps).

Extended methods: AMCMPSO (an interpretation of the adaptive mean / centre-of-mass idea of the base paper, not a reproduction of it), Standard PSO with a relative tolerance stop (1 % over 10 iterations) and the warm-started variants (particle 0 seeded with the LSQ + Gauss-Newton estimate). The CRLB column is the mean Cramer-Rao lower bound on the RMS position error at the sampled true positions for Gaussian noise of the given sigma; it depends on the anchor geometry only and is a reference, not a method.

## Ranging noise sigma [m]

| value | method | RMSE [m] | mean err [m] | p90 err [m] | CRLB [m] | fitness evals | iterations | runtime [ms] | memory [floats] |
|---|---|---|---|---|---|---|---|---|---|
| 0.1 | Simplified PSO | 7.856 | 5.486 | 14.378 | 0.503 | 1200 | 60.0 | 7.012 | 120 |
| 0.1 | Standard PSO | 5.878 | 2.646 | 11.769 | 0.503 | 1220 | 60.0 | 17.259 | 183 |
| 0.1 | Least-squares trilateration | 1.172 | 0.984 | 1.967 | 0.487 | 0 | 0.0 | 0.262 | 15 |
| 0.1 | LSQ + Gauss-Newton | 0.621 | 0.426 | 1.003 | 0.487 | 7 | 7.0 | 0.841 | 15 |
| 0.1 | AMCMPSO (interpretation) | 5.289 | 2.215 | 8.585 | 0.503 | 1220 | 60.0 | 16.096 | 186 |
| 0.1 | Standard PSO + tolerance stop | 6.105 | 2.872 | 11.901 | 0.503 | 919 | 45.0 | 3.706 | 183 |
| 0.1 | Standard PSO + LSQ/GN warm start | 0.592 | 0.421 | 1.048 | 0.503 | 1220 | 60.0 | 7.381 | 183 |
| 0.1 | AMCMPSO (interpretation) + warm start | 0.592 | 0.421 | 1.048 | 0.503 | 1220 | 60.0 | 9.598 | 186 |
| 0.25 | Simplified PSO | 8.070 | 5.839 | 14.458 | 1.256 | 1200 | 60.0 | 3.010 | 120 |
| 0.25 | Standard PSO | 6.059 | 3.167 | 11.791 | 1.256 | 1220 | 60.0 | 8.729 | 183 |
| 0.25 | Least-squares trilateration | 2.931 | 2.461 | 4.920 | 1.218 | 0 | 0.0 | 0.098 | 15 |
| 0.25 | LSQ + Gauss-Newton | 1.935 | 1.225 | 2.624 | 1.218 | 8 | 7.8 | 0.602 | 15 |
| 0.25 | AMCMPSO (interpretation) | 5.365 | 2.648 | 8.482 | 1.256 | 1220 | 60.0 | 13.881 | 186 |
| 0.25 | Standard PSO + tolerance stop | 6.300 | 3.372 | 12.159 | 1.256 | 871 | 42.5 | 7.032 | 183 |
| 0.25 | Standard PSO + LSQ/GN warm start | 1.336 | 0.986 | 2.362 | 1.256 | 1220 | 60.0 | 5.120 | 183 |
| 0.25 | AMCMPSO (interpretation) + warm start | 1.335 | 0.985 | 2.361 | 1.256 | 1220 | 60.0 | 12.595 | 186 |
| 0.5 | Simplified PSO | 8.379 | 6.253 | 14.651 | 2.513 | 1200 | 60.0 | 14.471 | 120 |
| 0.5 | Standard PSO | 6.533 | 3.983 | 12.931 | 2.513 | 1220 | 60.0 | 6.130 | 183 |
| 0.5 | Least-squares trilateration | 5.860 | 4.920 | 9.850 | 2.435 | 0 | 0.0 | 0.074 | 15 |
| 0.5 | LSQ + Gauss-Newton | 4.174 | 2.678 | 6.526 | 2.435 | 8 | 8.3 | 0.350 | 15 |
| 0.5 | AMCMPSO (interpretation) | 5.468 | 3.202 | 8.414 | 2.513 | 1220 | 60.0 | 7.378 | 186 |
| 0.5 | Standard PSO + tolerance stop | 6.736 | 4.157 | 13.415 | 2.513 | 781 | 38.1 | 7.613 | 183 |
| 0.5 | Standard PSO + LSQ/GN warm start | 3.192 | 2.112 | 4.655 | 2.513 | 1220 | 60.0 | 4.345 | 183 |
| 0.5 | AMCMPSO (interpretation) + warm start | 3.192 | 2.112 | 4.655 | 2.513 | 1220 | 60.0 | 4.663 | 186 |
| 1 | Simplified PSO | 8.555 | 6.705 | 14.687 | 5.026 | 1200 | 60.0 | 1.396 | 120 |
| 1 | Standard PSO | 6.978 | 4.921 | 13.205 | 5.026 | 1220 | 60.0 | 1.867 | 183 |
| 1 | Least-squares trilateration | 11.717 | 9.834 | 19.716 | 4.870 | 0 | 0.0 | 0.031 | 15 |
| 1 | LSQ + Gauss-Newton | 7.366 | 4.971 | 10.590 | 4.870 | 9 | 8.8 | 0.292 | 15 |
| 1 | AMCMPSO (interpretation) | 5.858 | 4.080 | 10.528 | 5.026 | 1220 | 60.0 | 3.474 | 186 |
| 1 | Standard PSO + tolerance stop | 7.136 | 5.041 | 13.431 | 5.026 | 723 | 35.2 | 1.244 | 183 |
| 1 | Standard PSO + LSQ/GN warm start | 6.177 | 4.370 | 11.980 | 5.026 | 1220 | 60.0 | 2.364 | 183 |
| 1 | AMCMPSO (interpretation) + warm start | 6.219 | 4.390 | 11.979 | 5.026 | 1220 | 60.0 | 3.939 | 186 |
| 2 | Simplified PSO | 9.324 | 7.750 | 15.231 | 10.051 | 1200 | 60.0 | 2.273 | 120 |
| 2 | Standard PSO | 7.772 | 6.126 | 14.203 | 10.051 | 1220 | 60.0 | 2.012 | 183 |
| 2 | Least-squares trilateration | 23.424 | 19.648 | 39.043 | 9.740 | 0 | 0.0 | 0.031 | 15 |
| 2 | LSQ + Gauss-Newton | 11.187 | 8.374 | 18.656 | 9.740 | 9 | 9.1 | 0.286 | 15 |
| 2 | AMCMPSO (interpretation) | 7.246 | 5.780 | 12.873 | 10.051 | 1220 | 60.0 | 3.553 | 186 |
| 2 | Standard PSO + tolerance stop | 7.857 | 6.185 | 14.278 | 10.051 | 630 | 30.5 | 1.044 | 183 |
| 2 | Standard PSO + LSQ/GN warm start | 7.502 | 5.956 | 13.863 | 10.051 | 1220 | 60.0 | 2.315 | 183 |
| 2 | AMCMPSO (interpretation) + warm start | 7.598 | 5.989 | 14.120 | 10.051 | 1220 | 60.0 | 3.733 | 186 |

## Number of anchors

| value | method | RMSE [m] | mean err [m] | p90 err [m] | CRLB [m] | fitness evals | iterations | runtime [ms] | memory [floats] |
|---|---|---|---|---|---|---|---|---|---|
| 4 | Simplified PSO | 8.379 | 6.253 | 14.651 | 2.513 | 1200 | 60.0 | 1.430 | 120 |
| 4 | Standard PSO | 6.533 | 3.983 | 12.931 | 2.513 | 1220 | 60.0 | 1.905 | 183 |
| 4 | Least-squares trilateration | 5.860 | 4.920 | 9.850 | 2.435 | 0 | 0.0 | 0.033 | 15 |
| 4 | LSQ + Gauss-Newton | 4.174 | 2.678 | 6.526 | 2.435 | 8 | 8.3 | 0.259 | 15 |
| 4 | AMCMPSO (interpretation) | 5.468 | 3.202 | 8.414 | 2.513 | 1220 | 60.0 | 5.175 | 186 |
| 4 | Standard PSO + tolerance stop | 6.736 | 4.157 | 13.415 | 2.513 | 781 | 38.1 | 1.797 | 183 |
| 4 | Standard PSO + LSQ/GN warm start | 3.192 | 2.112 | 4.655 | 2.513 | 1220 | 60.0 | 3.421 | 183 |
| 4 | AMCMPSO (interpretation) + warm start | 3.192 | 2.112 | 4.655 | 2.513 | 1220 | 60.0 | 8.423 | 186 |
| 5 | Simplified PSO | 7.693 | 5.282 | 14.577 | 1.646 | 1200 | 60.0 | 1.589 | 120 |
| 5 | Standard PSO | 5.683 | 3.110 | 8.000 | 1.646 | 1220 | 60.0 | 2.107 | 183 |
| 5 | Least-squares trilateration | 4.204 | 3.264 | 7.038 | 1.627 | 0 | 0.0 | 0.032 | 18 |
| 5 | LSQ + Gauss-Newton | 2.828 | 1.730 | 4.014 | 1.627 | 8 | 8.1 | 0.258 | 18 |
| 5 | AMCMPSO (interpretation) | 4.063 | 2.250 | 4.675 | 1.646 | 1220 | 60.0 | 4.522 | 186 |
| 5 | Standard PSO + tolerance stop | 5.720 | 3.137 | 8.474 | 1.646 | 751 | 36.5 | 1.286 | 183 |
| 5 | Standard PSO + LSQ/GN warm start | 2.696 | 1.673 | 3.666 | 1.646 | 1220 | 60.0 | 2.284 | 183 |
| 5 | AMCMPSO (interpretation) + warm start | 2.696 | 1.673 | 3.666 | 1.646 | 1220 | 60.0 | 8.653 | 186 |
| 6 | Simplified PSO | 6.424 | 4.247 | 12.296 | 1.349 | 1200 | 60.0 | 1.456 | 120 |
| 6 | Standard PSO | 3.051 | 1.558 | 2.609 | 1.349 | 1220 | 60.0 | 1.992 | 183 |
| 6 | Least-squares trilateration | 3.066 | 2.333 | 4.869 | 1.391 | 0 | 0.0 | 0.032 | 21 |
| 6 | LSQ + Gauss-Newton | 2.124 | 1.436 | 2.825 | 1.391 | 8 | 8.3 | 0.264 | 21 |
| 6 | AMCMPSO (interpretation) | 2.397 | 1.329 | 2.394 | 1.349 | 1220 | 60.0 | 3.418 | 186 |
| 6 | Standard PSO + tolerance stop | 3.245 | 1.699 | 2.919 | 1.349 | 731 | 35.5 | 1.251 | 183 |
| 6 | Standard PSO + LSQ/GN warm start | 1.765 | 1.153 | 2.081 | 1.349 | 1220 | 60.0 | 2.745 | 183 |
| 6 | AMCMPSO (interpretation) + warm start | 1.756 | 1.140 | 2.082 | 1.349 | 1220 | 60.0 | 3.748 | 186 |
| 8 | Simplified PSO | 5.804 | 3.806 | 10.445 | 1.004 | 1200 | 60.0 | 1.560 | 120 |
| 8 | Standard PSO | 2.698 | 1.276 | 1.544 | 1.004 | 1220 | 60.0 | 2.035 | 183 |
| 8 | Least-squares trilateration | 2.136 | 1.672 | 3.140 | 0.956 | 0 | 0.0 | 0.035 | 27 |
| 8 | LSQ + Gauss-Newton | 1.147 | 0.892 | 1.876 | 0.956 | 8 | 8.1 | 0.259 | 27 |
| 8 | AMCMPSO (interpretation) | 2.565 | 1.241 | 1.676 | 1.004 | 1220 | 60.0 | 3.540 | 186 |
| 8 | Standard PSO + tolerance stop | 2.746 | 1.329 | 1.946 | 1.004 | 725 | 35.2 | 1.293 | 183 |
| 8 | Standard PSO + LSQ/GN warm start | 1.043 | 0.809 | 1.319 | 1.004 | 1220 | 60.0 | 2.534 | 183 |
| 8 | AMCMPSO (interpretation) + warm start | 0.995 | 0.793 | 1.290 | 1.004 | 1220 | 60.0 | 3.867 | 186 |
| 10 | Simplified PSO | 4.947 | 3.026 | 8.056 | 0.788 | 1200 | 60.0 | 1.671 | 120 |
| 10 | Standard PSO | 2.372 | 1.055 | 1.695 | 0.788 | 1220 | 60.0 | 2.189 | 183 |
| 10 | Least-squares trilateration | 1.609 | 1.369 | 2.627 | 0.827 | 0 | 0.0 | 0.032 | 33 |
| 10 | LSQ + Gauss-Newton | 0.835 | 0.705 | 1.302 | 0.827 | 8 | 7.9 | 0.346 | 33 |
| 10 | AMCMPSO (interpretation) | 1.605 | 0.865 | 1.439 | 0.788 | 1220 | 60.0 | 3.631 | 186 |
| 10 | Standard PSO + tolerance stop | 2.381 | 1.073 | 1.778 | 0.788 | 703 | 34.2 | 1.313 | 183 |
| 10 | Standard PSO + LSQ/GN warm start | 0.794 | 0.674 | 1.286 | 0.788 | 1220 | 60.0 | 7.973 | 183 |
| 10 | AMCMPSO (interpretation) + warm start | 0.794 | 0.674 | 1.286 | 0.788 | 1220 | 60.0 | 3.940 | 186 |

## Swarm size N

| value | method | RMSE [m] | mean err [m] | p90 err [m] | CRLB [m] | fitness evals | iterations | runtime [ms] | memory [floats] |
|---|---|---|---|---|---|---|---|---|---|
| 5 | Simplified PSO | 10.205 | 8.179 | 17.581 | 2.513 | 300 | 60.0 | 1.179 | 30 |
| 5 | Standard PSO | 7.081 | 4.974 | 14.115 | 2.513 | 305 | 60.0 | 1.673 | 48 |
| 5 | AMCMPSO (interpretation) | 6.718 | 4.683 | 11.935 | 2.513 | 305 | 60.0 | 2.951 | 51 |
| 5 | Standard PSO + tolerance stop | 7.244 | 5.147 | 14.115 | 2.513 | 204 | 39.8 | 1.196 | 48 |
| 5 | Standard PSO + LSQ/GN warm start | 3.217 | 2.141 | 4.747 | 2.513 | 305 | 60.0 | 1.969 | 48 |
| 5 | AMCMPSO (interpretation) + warm start | 3.204 | 2.129 | 4.655 | 2.513 | 305 | 60.0 | 3.321 | 51 |
| 10 | Simplified PSO | 9.136 | 7.091 | 16.950 | 2.513 | 600 | 60.0 | 1.241 | 60 |
| 10 | Standard PSO | 6.785 | 4.317 | 13.589 | 2.513 | 610 | 60.0 | 1.781 | 93 |
| 10 | AMCMPSO (interpretation) | 5.971 | 3.581 | 10.186 | 2.513 | 610 | 60.0 | 3.134 | 96 |
| 10 | Standard PSO + tolerance stop | 6.818 | 4.394 | 13.556 | 2.513 | 413 | 40.3 | 1.247 | 93 |
| 10 | Standard PSO + LSQ/GN warm start | 3.199 | 2.117 | 4.655 | 2.513 | 610 | 60.0 | 2.143 | 93 |
| 10 | AMCMPSO (interpretation) + warm start | 3.195 | 2.122 | 4.655 | 2.513 | 610 | 60.0 | 3.485 | 96 |
| 20 | Simplified PSO | 8.379 | 6.253 | 14.651 | 2.513 | 1200 | 60.0 | 1.415 | 120 |
| 20 | Standard PSO | 6.533 | 3.983 | 12.931 | 2.513 | 1220 | 60.0 | 1.894 | 183 |
| 20 | AMCMPSO (interpretation) | 5.468 | 3.202 | 8.414 | 2.513 | 1220 | 60.0 | 3.384 | 186 |
| 20 | Standard PSO + tolerance stop | 6.736 | 4.157 | 13.415 | 2.513 | 781 | 38.1 | 1.267 | 183 |
| 20 | Standard PSO + LSQ/GN warm start | 3.192 | 2.112 | 4.655 | 2.513 | 1220 | 60.0 | 2.265 | 183 |
| 20 | AMCMPSO (interpretation) + warm start | 3.192 | 2.112 | 4.655 | 2.513 | 1220 | 60.0 | 3.664 | 186 |
| 40 | Simplified PSO | 7.027 | 4.794 | 14.312 | 2.513 | 2400 | 60.0 | 1.599 | 240 |
| 40 | Standard PSO | 6.133 | 3.833 | 11.786 | 2.513 | 2440 | 60.0 | 2.168 | 363 |
| 40 | AMCMPSO (interpretation) | 5.946 | 3.567 | 11.971 | 2.513 | 2440 | 60.0 | 3.679 | 366 |
| 40 | Standard PSO + tolerance stop | 6.201 | 3.880 | 11.784 | 2.513 | 1471 | 35.8 | 3.153 | 363 |
| 40 | Standard PSO + LSQ/GN warm start | 3.192 | 2.112 | 4.655 | 2.513 | 2440 | 60.0 | 2.793 | 363 |
| 40 | AMCMPSO (interpretation) + warm start | 3.209 | 2.132 | 4.750 | 2.513 | 2440 | 60.0 | 4.162 | 366 |
| 80 | Simplified PSO | 6.370 | 4.282 | 12.064 | 2.513 | 4800 | 60.0 | 2.100 | 480 |
| 80 | Standard PSO | 5.160 | 3.044 | 7.228 | 2.513 | 4880 | 60.0 | 2.838 | 723 |
| 80 | AMCMPSO (interpretation) | 4.651 | 2.782 | 6.566 | 2.513 | 4880 | 60.0 | 4.606 | 726 |
| 80 | Standard PSO + tolerance stop | 5.243 | 3.120 | 7.443 | 2.513 | 2830 | 34.4 | 1.679 | 723 |
| 80 | Standard PSO + LSQ/GN warm start | 3.192 | 2.112 | 4.655 | 2.513 | 4880 | 60.0 | 10.839 | 723 |
| 80 | AMCMPSO (interpretation) + warm start | 3.192 | 2.112 | 4.655 | 2.513 | 4880 | 60.0 | 11.737 | 726 |

## Iterations T

| value | method | RMSE [m] | mean err [m] | p90 err [m] | CRLB [m] | fitness evals | iterations | runtime [ms] | memory [floats] |
|---|---|---|---|---|---|---|---|---|---|
| 10 | Simplified PSO | 8.755 | 7.088 | 14.441 | 2.513 | 200 | 10.0 | 0.574 | 120 |
| 10 | Standard PSO | 7.064 | 4.948 | 11.984 | 2.513 | 220 | 10.0 | 1.047 | 183 |
| 10 | AMCMPSO (interpretation) | 6.411 | 4.589 | 10.701 | 2.513 | 220 | 10.0 | 1.543 | 186 |
| 10 | Standard PSO + tolerance stop | 7.064 | 4.948 | 11.984 | 2.513 | 220 | 10.0 | 1.903 | 183 |
| 10 | Standard PSO + LSQ/GN warm start | 3.255 | 2.186 | 4.771 | 2.513 | 220 | 10.0 | 6.802 | 183 |
| 10 | AMCMPSO (interpretation) + warm start | 3.229 | 2.160 | 4.766 | 2.513 | 220 | 10.0 | 6.628 | 186 |
| 20 | Simplified PSO | 8.532 | 6.574 | 14.698 | 2.513 | 400 | 20.0 | 1.057 | 120 |
| 20 | Standard PSO | 6.790 | 4.391 | 12.320 | 2.513 | 420 | 20.0 | 1.251 | 183 |
| 20 | AMCMPSO (interpretation) | 5.886 | 3.763 | 8.959 | 2.513 | 420 | 20.0 | 2.903 | 186 |
| 20 | Standard PSO + tolerance stop | 6.810 | 4.419 | 12.320 | 2.513 | 413 | 19.7 | 1.630 | 183 |
| 20 | Standard PSO + LSQ/GN warm start | 3.187 | 2.108 | 4.655 | 2.513 | 420 | 20.0 | 7.030 | 183 |
| 20 | AMCMPSO (interpretation) + warm start | 3.197 | 2.120 | 4.655 | 2.513 | 420 | 20.0 | 3.363 | 186 |
| 30 | Simplified PSO | 8.383 | 6.275 | 14.653 | 2.513 | 600 | 30.0 | 1.651 | 120 |
| 30 | Standard PSO | 6.616 | 4.104 | 13.046 | 2.513 | 620 | 30.0 | 2.506 | 183 |
| 30 | AMCMPSO (interpretation) | 6.102 | 3.663 | 11.671 | 2.513 | 620 | 30.0 | 4.801 | 186 |
| 30 | Standard PSO + tolerance stop | 6.785 | 4.227 | 13.437 | 2.513 | 581 | 28.0 | 11.319 | 183 |
| 30 | Standard PSO + LSQ/GN warm start | 3.189 | 2.110 | 4.655 | 2.513 | 620 | 30.0 | 5.055 | 183 |
| 30 | AMCMPSO (interpretation) + warm start | 3.179 | 2.092 | 4.655 | 2.513 | 620 | 30.0 | 5.225 | 186 |
| 60 | Simplified PSO | 8.379 | 6.253 | 14.651 | 2.513 | 1200 | 60.0 | 2.006 | 120 |
| 60 | Standard PSO | 6.533 | 3.983 | 12.931 | 2.513 | 1220 | 60.0 | 2.903 | 183 |
| 60 | AMCMPSO (interpretation) | 5.468 | 3.202 | 8.414 | 2.513 | 1220 | 60.0 | 4.788 | 186 |
| 60 | Standard PSO + tolerance stop | 6.736 | 4.157 | 13.415 | 2.513 | 781 | 38.1 | 2.732 | 183 |
| 60 | Standard PSO + LSQ/GN warm start | 3.192 | 2.112 | 4.655 | 2.513 | 1220 | 60.0 | 2.749 | 183 |
| 60 | AMCMPSO (interpretation) + warm start | 3.192 | 2.112 | 4.655 | 2.513 | 1220 | 60.0 | 3.848 | 186 |
| 120 | Simplified PSO | 8.381 | 6.254 | 14.653 | 2.513 | 2400 | 120.0 | 2.695 | 120 |
| 120 | Standard PSO | 6.505 | 3.951 | 12.934 | 2.513 | 2420 | 120.0 | 3.735 | 183 |
| 120 | AMCMPSO (interpretation) | 5.410 | 3.202 | 9.510 | 2.513 | 2420 | 120.0 | 9.396 | 186 |
| 120 | Standard PSO + tolerance stop | 6.736 | 4.158 | 13.415 | 2.513 | 793 | 38.6 | 10.786 | 183 |
| 120 | Standard PSO + LSQ/GN warm start | 3.192 | 2.112 | 4.655 | 2.513 | 2420 | 120.0 | 4.925 | 183 |
| 120 | AMCMPSO (interpretation) + warm start | 3.192 | 2.112 | 4.655 | 2.513 | 2420 | 120.0 | 9.797 | 186 |

## Error CDF (sigma = 0.5 m, 4 anchors)

| method | RMSE [m] | median [m] | p90 [m] | fitness evals | memory [floats] |
|---|---|---|---|---|---|
| Simplified PSO (20 x 60) | 8.379 | 4.660 | 14.651 | 1200 | 120 |
| Standard PSO (20 x 60) | 6.533 | 1.674 | 12.931 | 1220 | 183 |
| Least-squares trilateration | 5.860 | 4.478 | 9.850 | 0 | 15 |
| LSQ + Gauss-Newton | 4.174 | 1.479 | 6.526 | 8 | 15 |
| AMCMPSO (interpretation) (20 x 60) | 5.468 | 1.545 | 8.414 | 1220 | 186 |
| Standard PSO + tolerance stop (20 x 60) | 6.736 | 1.756 | 13.415 | 781 | 183 |
| Standard PSO + LSQ/GN warm start (20 x 60) | 3.192 | 1.317 | 4.655 | 1220 | 183 |
| AMCMPSO (interpretation) + warm start (20 x 60) | 3.192 | 1.317 | 4.655 | 1220 | 186 |

## Extended methods (sigma = 0.5 m, 4 anchors, N = 20, T = 60)

| method | RMSE [m] | median [m] | p90 [m] | CRLB [m] | fitness evals | iterations | memory [floats] |
|---|---|---|---|---|---|---|---|
| Simplified PSO | 8.379 | 4.660 | 14.651 | 2.513 | 1200 | 60.0 | 120 |
| Standard PSO | 6.533 | 1.674 | 12.931 | 2.513 | 1220 | 60.0 | 183 |
| Least-squares trilateration | 5.860 | 4.478 | 9.850 | 2.435 | 0 | 0.0 | 15 |
| LSQ + Gauss-Newton | 4.174 | 1.479 | 6.526 | 2.435 | 8 | 8.3 | 15 |
| AMCMPSO (interpretation) | 5.468 | 1.545 | 8.414 | 2.513 | 1220 | 60.0 | 186 |
| Standard PSO + tolerance stop | 6.736 | 1.756 | 13.415 | 2.513 | 781 | 38.1 | 183 |
| Standard PSO + LSQ/GN warm start | 3.192 | 1.317 | 4.655 | 2.513 | 1220 | 60.0 | 183 |
| AMCMPSO (interpretation) + warm start | 3.192 | 1.317 | 4.655 | 2.513 | 1220 | 60.0 | 186 |

Notes: fitness evaluations count swarm evaluations only (the warm start adds one least-squares solve and <= 10 Gauss-Newton steps per node); iterations are the optimiser iterations actually run (the tolerance stop ends a run early, closed-form rows report Gauss-Newton steps). AMCMPSO numbers are those of this repository's interpretation and must not be quoted as results of the base paper.
