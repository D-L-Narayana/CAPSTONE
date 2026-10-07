/* Geometry diagnostics for range-based 3D localization — browser twin of pso3d/geometry.py.
 *
 * Plain browser script (no module syntax): the public functions are attached to the global object in the
 * browser and exported through module.exports in Node. No DOM access. Everything here is deterministic
 * (no random numbers), so the values match the Python package up to floating-point rounding.
 */
(function (root) {
  'use strict';

  const MIN_DIST = 1e-12;        // clamp for zero anchor-to-point distances (same guard as the Gauss-Newton solver)
  const SINGULAR_REL = 1e-12;    // J^T J is treated as singular when its smallest eigenvalue <= SINGULAR_REL * largest

  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = a => Math.hypot(a[0], a[1], a[2]);

  function centroid(points) {
    const c = [0, 0, 0];
    for (const p of points) for (let k = 0; k < 3; k++) c[k] += p[k];
    return c.map(v => v / points.length);
  }

  /** Scatter matrix sum_j (a_j - c)(a_j - c)^T of the anchors about their centroid c (3x3, symmetric). */
  function scatterMatrix(anchors) {
    const c = centroid(anchors);
    const S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (const a of anchors) {
      const d = sub(a, c);
      for (let i = 0; i < 3; i++) for (let k = 0; k < 3; k++) S[i][k] += d[i] * d[k];
    }
    return S;
  }

  /**
   * Eigen-decomposition of a symmetric 3x3 matrix by cyclic Jacobi rotations.
   * Returns {values: [l0 <= l1 <= l2], vectors: [v0, v1, v2]} with unit eigenvectors matching the values.
   */
  function symEig3(S) {
    const A = [[S[0][0], S[0][1], S[0][2]], [S[1][0], S[1][1], S[1][2]], [S[2][0], S[2][1], S[2][2]]];
    const V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    for (let sweep = 0; sweep < 100; sweep++) {
      if (Math.abs(A[0][1]) + Math.abs(A[0][2]) + Math.abs(A[1][2]) === 0) break;
      for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
        const apq = A[p][q];
        if (apq === 0) continue;
        const g = 100 * Math.abs(apq);
        if (sweep > 3 && Math.abs(A[p][p]) + g === Math.abs(A[p][p]) && Math.abs(A[q][q]) + g === Math.abs(A[q][q])) {
          A[p][q] = A[q][p] = 0;          // below rounding level of both diagonal entries: cannot change the eigenvalues
          continue;
        }
        const theta = (A[q][q] - A[p][p]) / (2 * apq);
        const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        // A <- G^T A G and V <- V G with the Givens rotation G (G_pp = G_qq = c, G_pq = s, G_qp = -s)
        for (let k = 0; k < 3; k++) {
          const akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - s * akq;
          A[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < 3; k++) {
          const apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - s * aqk;
          A[q][k] = s * apk + c * aqk;
        }
        A[p][q] = A[q][p] = 0;
        for (let k = 0; k < 3; k++) {
          const vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
    const order = [0, 1, 2].sort((i, j) => A[i][i] - A[j][j]);
    return { values: order.map(i => A[i][i]), vectors: order.map(i => [V[0][i], V[1][i], V[2][i]]) };
  }

  /** Unsigned volume |det[b - a, c - a, d - a]| / 6 of the tetrahedron spanned by four points. */
  function tetrahedronVolume(a, b, c, d) {
    return Math.abs(dot(sub(b, a), cross(sub(c, a), sub(d, a)))) / 6;
  }

  /**
   * Singular values of the centred anchor matrix C = anchors - centroid (descending), computed as the
   * root-sum-square projections ||C v_i|| onto the scatter-matrix eigenvectors v_i. Mathematically
   * ||C v_i||^2 = v_i^T S v_i = lambda_i, but the projections keep a tiny out-of-plane spread accurate where the
   * eigenvalue itself would be lost in the rounding of the large diagonal entries.
   */
  function anchorSingularValues(anchors) {
    const c = centroid(anchors);
    const { vectors } = symEig3(scatterMatrix(anchors));
    return vectors.map(v => {
      let s = 0;
      for (const a of anchors) { const h = dot(sub(a, c), v); s += h * h; }
      return Math.sqrt(s);
    }).sort((x, y) => y - x);
  }

  /**
   * Numerical rank of the centred anchor matrix anchors - mean(anchors): the number of singular values above
   * tol * largest singular value (the numpy.linalg.matrix_rank rule used by pso3d.geometry.anchor_rank).
   * 3 = the anchors span a volume, 2 = coplanar, 1 = collinear, 0 = a single point (or no anchors).
   */
  function anchorRank(anchors, tol = 1e-9) {
    if (!anchors || anchors.length < 2) return 0;
    const sv = anchorSingularValues(anchors);
    if (!(sv[0] > 0)) return 0;
    return sv.filter(s => s > tol * sv[0]).length;
  }

  /** True when some four anchors span a tetrahedron with |volume| > minVolume (needed for a unique 3D fix). */
  function isNonCoplanar(anchors, minVolume = 1e-6) {
    const M = anchors.length;
    for (let i = 0; i < M; i++) for (let j = i + 1; j < M; j++) for (let k = j + 1; k < M; k++) for (let l = k + 1; l < M; l++) {
      if (tetrahedronVolume(anchors[i], anchors[j], anchors[k], anchors[l]) > minVolume) return true;
    }
    return false;
  }

  /** Jacobian of the range model at p: M x 3 rows (p - a_j) / ||p - a_j|| (distances below 1e-12 are clamped). */
  function rangeJacobian(anchors, p) {
    return anchors.map(a => {
      const diff = sub(p, a);
      let r = norm(diff);
      if (r < MIN_DIST) r = MIN_DIST;
      return [diff[0] / r, diff[1] / r, diff[2] / r];
    });
  }

  function fisherEig(anchors, p) {
    const J = rangeJacobian(anchors, p);
    const G = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (const row of J) for (let i = 0; i < 3; i++) for (let k = 0; k < 3; k++) G[i][k] += row[i] * row[k];
    const eig = symEig3(G);
    eig.singular = !(eig.values[2] > 0) || eig.values[0] <= SINGULAR_REL * eig.values[2];
    return eig;
  }

  /** Geometric dilution of precision sqrt(trace((J^T J)^-1)); Infinity when J^T J is singular. */
  function gdop(anchors, p) {
    const { values, singular } = fisherEig(anchors, p);
    if (singular) return Infinity;
    return Math.sqrt(1 / values[0] + 1 / values[1] + 1 / values[2]);
  }

  /**
   * Cramer-Rao lower bound for i.i.d. Gaussian ranging noise of standard deviation sigma (m, must be > 0):
   * FIM = J^T J / sigma^2, cov = FIM^-1, rmseBound = sqrt(trace(cov)) (= sigma * GDOP), perAxis = sqrt(diag(cov)).
   * A singular geometry gives every entry Infinity (cov is a 3x3 of Infinity), like the Python CRLB.
   */
  function crlb(anchors, p, sigma) {
    if (typeof sigma !== 'number' || !(sigma > 0) || !Number.isFinite(sigma)) {
      throw new RangeError(`sigma must be a positive ranging noise level in metres, got ${sigma}`);
    }
    const { values, vectors, singular } = fisherEig(anchors, p);
    if (singular) {
      const inf = [Infinity, Infinity, Infinity];
      return { cov: [inf.slice(), inf.slice(), inf.slice()], rmseBound: Infinity, perAxis: inf };
    }
    const s2 = sigma * sigma;
    const cov = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let n = 0; n < 3; n++) {
      const v = vectors[n], scale = s2 / values[n];
      for (let i = 0; i < 3; i++) for (let k = 0; k < 3; k++) cov[i][k] += scale * v[i] * v[k];
    }
    return {
      cov,
      rmseBound: Math.sqrt(cov[0][0] + cov[1][1] + cov[2][2]),
      perAxis: [Math.sqrt(cov[0][0]), Math.sqrt(cov[1][1]), Math.sqrt(cov[2][2])],
    };
  }

  /**
   * Reflection of p across the least-squares plane of the anchors (through their centroid, normal = eigenvector of the
   * smallest scatter eigenvalue). Returns null when the anchors do not span a plane (rank < 2).
   */
  function mirrorPoint(anchors, p) {
    if (!anchors || anchors.length < 3 || anchorRank(anchors) < 2) return null;
    const c = centroid(anchors);
    const n = symEig3(scatterMatrix(anchors)).vectors[0];
    const h = dot(sub(p, c), n);
    return [p[0] - 2 * h * n[0], p[1] - 2 * h * n[1], p[2] - 2 * h * n[2]];
  }

  function rangeFitness(anchors, measured, q) {
    let s = 0;
    for (let j = 0; j < anchors.length; j++) { const e = norm(sub(q, anchors[j])) - measured[j]; s += e * e; }
    return s;
  }

  /**
   * Flip-ambiguity risk = clip(f(p) / max(f(mirror), 1e-12), 0, 1) with the range-error fitness
   * f(q) = sum_j (||q - a_j|| - d_j)^2 and mirror = mirrorPoint(anchors, p); 1 when no mirror plane exists.
   * About 1 means the mirrored position fits the measured ranges as well as p (indistinguishable);
   * near 0 means the mirror fits much worse.
   */
  function flipAmbiguityRisk(anchors, p, measured) {
    const m = mirrorPoint(anchors, p);
    if (m === null) return 1;
    const ratio = rangeFitness(anchors, measured, p) / Math.max(rangeFitness(anchors, measured, m), 1e-12);
    return Math.min(1, Math.max(0, ratio));
  }

  const api = { tetrahedronVolume, anchorRank, isNonCoplanar, rangeJacobian, gdop, crlb, mirrorPoint, flipAmbiguityRisk };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else Object.assign(root, api);
})(typeof globalThis !== 'undefined' ? globalThis : this);
