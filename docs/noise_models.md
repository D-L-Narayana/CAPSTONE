# Noise models and robust fitness

Modules: `pso3d/noise.py` (ranging-noise models, `measure`, `from_spec`) and `pso3d/robust.py`
(Huber and weighted range-error fitness). Tests: `tests/test_noise.py`, `tests/test_robust.py`;
shared fixture: `tests/fixtures/noise_cases.json`.

## 1. Contract

```python
class NoiseModel(Protocol):
    name: str
    def sample(self, true_ranges: np.ndarray, rng: np.random.Generator) -> np.ndarray   # same shape
```

* Noise is **additive on the distance domain**: `measured = true_ranges + model.sample(true_ranges, rng)`.
* `true_ranges` may be `(M,)` or `(K, M)` (last axis = anchors); the output has the same shape, dtype float64.
* Sampling is deterministic for a given generator state (`np.random.default_rng(seed)`); models hold no state
  and never mutate their input.
* Models are frozen dataclasses: value equality (`Gaussian(0.5) == Gaussian(0.5)`), hashable, readable `repr`,
  and `model.describe()` returns a spec string with `from_spec(model.describe()) == model`.
* Every model consumes random draws in a fixed, documented order, so streams stay aligned when parameters
  change (e.g. `Gaussian(0.0)` still draws once per range).

## 2. Models

| model | noise on anchor *j* | spec string | use it for |
|---|---|---|---|
| `FixedOffsets(offsets)` | `n_j = offsets[j]` | `fixed:0.4,-0.3,0.5,-0.4` | reproducing fixed numbers (Review-1 baseline), regression tests |
| `Gaussian(sigma)` | `n_j ~ N(0, σ²)` i.i.d., metres | `gaussian:0.5` or `0.5` | line-of-sight ranging whose error does not grow with distance; Monte-Carlo default; the CRLB assumes it |
| `UniformPercent(pn)` | `n_j = d_j · U(−1, 1) · Pn/100` | `percent:2` | range-proportional, bounded error of Kulkarni et al. (2009): `d ± d·Pn/100`; network simulator |
| `LogNormalShadowing(sigma_db, n=2.8, P_ref=−45)` | RSSI round trip, see below | `shadowing:4`, `shadowing:4:3.0:-40` | RSSI-derived ranges: multiplicative, skewed, distance-dependent error |
| `NLOS(p, bias_mean, bias_sigma, base=Gaussian(0.5))` | `base_j + B_j · \|N(μ, s²)\|`, `B_j ~ Bernoulli(p)` | `nlos:0.2`, `nlos:0.2:3.0:1.0`, `nlos:0.2:3.0:1.0:0.7` | obstructed paths: a fraction *p* of anchors get a positive bias; robustness studies |

### FixedOffsets

Returns the stored offsets (broadcast over a `(K, M)` input); a length mismatch raises `ValueError`.
`FixedOffsets((0.4, -0.3, 0.5, -0.4))` reproduces `Scenario().measured()` exactly (first range
40.21519810323691 m), which is what the bit-exact baseline uses.

### Gaussian

Symmetric and unbiased; `sigma` in metres, `sigma >= 0`. Over 20 000 samples the empirical standard
deviation is within 10 % of `sigma` (tested).

### UniformPercent

`Pn` is a percentage. The error is bounded by `d_j · Pn/100`, has zero mean and standard deviation
`d_j · Pn / (100·√3)`; doubling the distance doubles the noise for the same draw.

### LogNormalShadowing (RSSI round trip)

Log-distance path loss with log-normal shadowing, converted back to a distance:

```
P(d)     = P_ref − 10·n·log10(d)          received power in dBm, P_ref measured at 1 m
P_noisy  = P(d) + X,   X ~ N(0, σ_dB²)     shadowing in dB
d_hat    = 10^((P_ref − P_noisy) / (10·n)) = d · 10^(−X / (10·n))
n_j      = d_hat − d
```

The implementation evaluates the closed multiplicative form, which is algebraically identical to the
round trip and gives exactly zero error when `X = 0` (`sigma_db = 0`). The factor `10^(−X/(10n))` is
log-normal, so the error is multiplicative (its spread grows linearly with the distance) and skewed:
a stronger-than-expected signal (`X > 0`) shortens the estimate, a weaker one lengthens it, and the mean
error is positive. Helpers for the literal round trip are exposed: `rssi_from_distance(d, n, P_ref)`,
`distance_from_rssi(rssi, n, P_ref)` and `rssi_round_trip(true_ranges, noise_db, n, P_ref)` (estimated
distances for *given* dB offsets).

Worked example (the repository's RSSI example code, `n = 2.8`, `P_ref = −45 dBm`):

| d (m) | X (dB) | P(d) (dBm) | P_noisy (dBm) | d_hat (m) | error (m) |
|---|---|---|---|---|---|
| 10 | +2.0 | −73.00 | −71.00 | 8.48 | −1.52 |
| 25 | −3.0 | −84.14 | −87.14 | 32.00 | +7.00 |
| 40 | +1.5 | −89.86 | −88.36 | 35.36 | −4.64 |

`sigma_db`, `n` and `P_ref` are not calibrated to any hardware; the defaults are those of the example.

### NLOS (non-line-of-sight mixture)

Draw order per call: the base model's sample, then one uniform per anchor (`< p` selects the anchor),
then one normal per anchor whose absolute value is the bias. Obstructed paths only ever *lengthen* a
range (longer path or reflection), hence the positive half-normal bias. `p = 0` reproduces the base
model exactly for the same seed; `p = 1` biases every anchor; for `0 < p < 1` the biased fraction
over many anchors matches `p`. The default `bias_mean = 2 m`, `bias_sigma = 1 m` are illustrative,
not taken from any paper. `base` may be any noise model (spec form `nlos:p:mean:sigma@percent:2`);
in the CLI form the optional fourth number is the base Gaussian sigma (`nlos:0.2:3.0:1.0:0.7`).

## 3. Helpers

* `measure(anchors, true_position, model, rng=None)` → `‖p − a_j‖ + model.sample(true_ranges, rng)`.
  `model` may be a `NoiseModel`, a float (Gaussian sigma), a spec string or a sequence of fixed offsets;
  `rng` may be a `numpy.random.Generator`, an integer seed or `None` (fresh entropy — pass a seed for
  reproducible runs). Shape errors raise `ValueError`.
* `as_noise_model(obj)` performs the same coercion and raises `TypeError` for anything else.
* `from_spec(spec)` parses the CLI grammar (case-insensitive, whitespace ignored); unknown kinds,
  wrong arity, non-numeric fields and invalid parameter values raise `ValueError`. `SPEC_HELP` is a
  one-line summary for `--help` texts.

```
gaussian:SIGMA | percent:PN | nlos:P[:BIAS_MEAN[:BIAS_SIGMA[:BASE_SIGMA]]] | nlos:P[:BIAS_MEAN[:BIAS_SIGMA]]@BASE_SPEC
shadowing:SIGMA_DB[:EXPONENT[:P_REF_DBM]] | fixed:N1,N2,... | SIGMA (bare number = gaussian) | none (= gaussian:0)
```

## 4. Robust fitness (`pso3d/robust.py`)

With residuals `r_j = ‖p − a_j‖ − d_hat_j`:

```
RangeErrorFitness      f(p) = Σ_j r_j²                                  (Review-1 baseline, unchanged)
HuberRangeFitness      f(p) = Σ_j ρ_δ(r_j),  ρ_δ(r) = r²            if |r| ≤ δ
                                                     = 2δ|r| − δ²    otherwise
WeightedRangeFitness   f(p) = Σ_j w_j r_j²                              (w_j ≥ 0; w_j = 1/σ_j² via from_sigmas)
```

Both classes subclass `RangeErrorFitness`, so they drop into `SimplifiedPSO`, `StandardPSO`, `AMCMPSO`
and the experiments unchanged; they accept a single point `(3,) → float` or a batch `(N, 3) → (N,)`
(Python lists included), provide `evaluate_one(p) -> float`, and keep the two counters with the baseline
semantics: one *evaluation* per candidate position, `M` *distance computations* per evaluation.
`huber_loss(residuals, delta)` is the element-wise loss. The baseline numbers are untouched — the robust
classes are opt-in: `SimplifiedPSO(...).run(HuberRangeFitness(anchors, measured, delta=1.0), lower, upper)`.

### Why Huber

* **Bounded influence.** Under the squared loss the pull of a residual on the gradient grows like `r`,
  so a single 10 m NLOS range dominates the sum and drags the minimiser by metres. Beyond `δ` the Huber
  loss grows linearly with slope `2δ`, so an outlier's pull is capped. In the cube-anchor test
  (`tests/test_robust.py`) the squared-loss argmin is more than 2 m off while the Huber argmin (`δ = 0.5`)
  stays within 0.75 m of the true position.
* **Identical to the baseline for small residuals.** For `|r| ≤ δ` the loss is exactly `r²`, so with
  well-behaved LOS ranges (and in the noise-free case) nothing changes.
* **Smooth.** Both branches have value `δ²` and slope `2δ` at `|r| = δ`: the loss and its first derivative
  are continuous, so neither PSO nor gradient-based refinement sees a kink.
* **Convex** in each residual, unlike redescending losses (e.g. Tukey's biweight), so it does not create
  extra local minima in the search landscape.
* **Choosing δ.** Use `δ ≈ 1–2 ×` the LOS noise standard deviation: ordinary residuals stay in the
  quadratic branch, outliers in the linear one. A residual bias of order `δ` divided by the number of
  well-placed anchors remains (the outlier still pulls with a constant force), which is why Huber
  complements, rather than replaces, good anchor geometry.

### Why weighted

`Σ w_j r_j²` with `w_j = 1/σ_j²` is (up to constants) the negative log-likelihood under independent
Gaussian ranging noise with per-anchor standard deviations `σ_j` — the maximum-likelihood fitness when
anchors differ in quality, e.g. RSSI ranging where `σ_j` grows with the distance. Unit weights reproduce
`RangeErrorFitness` exactly; a zero weight drops an anchor; negative weights and length mismatches raise
`ValueError`.

## 5. Fixture

`tests/fixtures/noise_cases.json` stores, for the slide-16 anchors and true position, one seeded sample
per model (`numpy.random.default_rng(2026)`, model given as a spec string), the fixed-offset measurement
and the RSSI worked example. `tests/test_noise.py` regenerates every entry from its spec and seed and
compares at `1e-12`; the file can also serve cross-language checks.

## 6. Browser counterpart

`web/pso.js` exposes `sampleNoise(model, trueRanges, rand)` for the simulator with the same formulas for
four of the families: `{type:'fixed', offsets}`, `{type:'gaussian', sigma}`, `{type:'uniformPercent', pn}`
and `{type:'nlos', p, biasMean, biasSigma, sigma}` (Gaussian base of `sigma` plus, with probability `p`,
`|N(biasMean, biasSigma²)|`). The browser draws from its own seeded generator (`mulberry32`) in a different
order, so stochastic samples are **not** bit-comparable between Python and JavaScript — they agree in
distribution only; cross-language parity fixtures therefore use fixed offsets. Log-normal shadowing is
Python-only.

## 7. Limitations

* These are textbook noise forms with illustrative defaults; nothing is calibrated to a specific radio.
* Noise is independent across anchors and across calls (no temporal correlation, no multipath model).
* The NLOS bias is a half-normal mixture; real NLOS error distributions vary with the environment.
* `Gaussian` has a constant `σ`; distance-dependent spreads are available through `UniformPercent`,
  `LogNormalShadowing` or per-anchor weights in `WeightedRangeFitness`.
