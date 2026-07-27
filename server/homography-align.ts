/**
 * server/homography-align.ts
 *
 * Homography (affine) pre-alignment for borderline segment rescue.
 *
 * When ORB keypoint correspondences reveal a geometric mismatch between a
 * short-clip frame and a movie frame — rotation, scale change, crop offset,
 * or a non-standard zoom that falls outside the 13 preset crop/zoom variants —
 * this module:
 *
 *  1. Computes a least-squares affine transform from the ORB correspondences
 *     (RANSAC-based outlier rejection for robustness).
 *  2. Warps the short-clip 128×72 grayscale thumbnail to align it with the
 *     movie frame's perspective/geometry.
 *  3. Computes Normalised Cross-Correlation (NCC) between the warped thumbnail
 *     and the movie thumbnail as a refined similarity score (alignedHashSim).
 *
 * ## This is alignment, not score manipulation
 *
 * This step *corrects the comparison* by removing a geometric mismatch before
 * the hash is computed.  It is the equivalent of straightening a tilted scan
 * before OCR — the underlying content is compared more accurately, not curved
 * toward a desired answer.  Logs say "corrected" / "refined", not "boosted".
 *
 * ## Scope constraint
 *
 * Only called from enhanceBorderlineSegments (confidence 55–75%).
 * NEVER called on the main per-frame pipeline.
 *
 * ## Dependencies
 *
 * Pure TypeScript; zero new npm packages; zero Docker image size increase.
 * Keypoint coordinates arrive from keypoint-signal.ts (same 128×72 space).
 */

import type { KeypointCorrespondence } from './keypoint-signal';

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/** Analysis resolution — must match ANALYSIS_W / ANALYSIS_H in keypoint-signal.ts */
const W = 128;
const H =  72;

/** RANSAC: maximum iterations */
const RANSAC_MAX_ITERS   = 60;

/** RANSAC: reprojection error tolerance in the 128×72 pixel space */
const RANSAC_THRESHOLD   = 3.0;

/** RANSAC: minimum inliers to accept an affine model */
const RANSAC_MIN_INLIERS = 4;

/**
 * NCC threshold (0–100) above which the post-alignment similarity is treated
 * as confirmatory evidence of a genuine match.
 *
 * NCC baseline: uncorrelated frames → ~50; genuine aligned match → 70–95.
 */
export const ALIGNED_SIM_THRESHOLD = 70;

/** Minimum good matches needed before attempting alignment (affine needs ≥ 3) */
export const MIN_MATCHES_FOR_ALIGNMENT = 4;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * 2D affine transform:
 *   x' = a·x + b·y + c
 *   y' = d·x + e·y + f
 */
interface AffineMatrix {
  a: number; b: number; c: number;
  d: number; e: number; f: number;
}

export interface HomographyAlignmentResult {
  /** true when a reliable affine model was found and NCC was computed */
  aligned: boolean;
  /**
   * Normalised Cross-Correlation between warped-short and movie thumbnails,
   * mapped to 0–100.  0 when aligned=false.
   * NCC=1 (sim=100) means identical; NCC=0 (sim=50) means uncorrelated.
   */
  alignedHashSim: number;
  /** NCC between raw (un-aligned) short and movie, 0–100 — used in log output */
  priorSim: number;
  /** Approximate rotation angle (degrees) extracted from the affine matrix */
  rotation: number;
  /** Approximate geometric scale extracted from the affine matrix */
  scale: number;
  /** Number of RANSAC inliers used to refine the final affine model */
  inlierCount: number;
  /** Number of good ORB correspondences fed into RANSAC */
  goodMatchCount: number;
}

// ---------------------------------------------------------------------------
// Grayscale downsample (mirrors toGray in keypoint-signal.ts — kept local to
// avoid a circular import; both operate on the same 128×72 analysis grid)
// ---------------------------------------------------------------------------

function toGray128x72(
  src: Uint8ClampedArray | Uint8Array,
  srcW: number, srcH: number,
): Float32Array {
  const gray = new Float32Array(W * H);
  const xScale = srcW / W;
  const yScale = srcH / H;
  for (let dy = 0; dy < H; dy++) {
    const sy = Math.min(srcH - 1, Math.round(dy * yScale));
    const sRow = sy * srcW;
    for (let dx = 0; dx < W; dx++) {
      const sx = Math.min(srcW - 1, Math.round(dx * xScale));
      const p  = (sRow + sx) * 4;
      gray[dy * W + dx] = 0.299 * src[p] + 0.587 * src[p + 1] + 0.114 * src[p + 2];
    }
  }
  return gray;
}

// ---------------------------------------------------------------------------
// Linear algebra helpers
// ---------------------------------------------------------------------------

/**
 * Solve a square n×n linear system Ax = b via Gauss–Jordan elimination with
 * partial pivoting.  Returns the solution vector x, or null if singular.
 */
function solveLinearSystem(A: number[][], b: number[]): number[] | null {
  const n = A.length;
  // Build augmented matrix [A | b] (shallow copy — we mutate)
  const M: number[][] = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    // Partial pivot: find row with largest |M[row][col]| from col downward
    let pivotRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(M[row][col]) > Math.abs(M[pivotRow][col])) pivotRow = row;
    }
    [M[col], M[pivotRow]] = [M[pivotRow], M[col]];

    if (Math.abs(M[col][col]) < 1e-12) return null; // numerically singular

    // Eliminate all other rows in this column
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = M[row][col] / M[col][col];
      for (let j = col; j <= n; j++) M[row][j] -= factor * M[col][j];
    }
  }

  return M.map((row, i) => row[n] / row[i]);
}

/**
 * Fit a least-squares affine transform to N ≥ 3 correspondences.
 *
 * The x and y equations decouple, so we solve two independent 3×3 normal-
 * equation systems (AᵀA · coeff = Aᵀb), one for [a, b, c] and one for [d, e, f].
 */
function affineLeastSquares(pts: KeypointCorrespondence[]): AffineMatrix | null {
  // AᵀA is the same for both systems (depends only on src coords)
  const ATA: number[][] = [[0,0,0],[0,0,0],[0,0,0]];
  const ATbX = [0, 0, 0];
  const ATbY = [0, 0, 0];

  for (const { ax, ay, bx, by } of pts) {
    const row = [ax, ay, 1];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) ATA[i][j] += row[i] * row[j];
      ATbX[i] += row[i] * bx;
      ATbY[i] += row[i] * by;
    }
  }

  // Deep-copy ATA for the second solve (solveLinearSystem mutates its input)
  const xCoeffs = solveLinearSystem(ATA.map(r => [...r]), [...ATbX]);
  const yCoeffs = solveLinearSystem(ATA.map(r => [...r]), [...ATbY]);
  if (!xCoeffs || !yCoeffs) return null;

  return {
    a: xCoeffs[0], b: xCoeffs[1], c: xCoeffs[2],
    d: yCoeffs[0], e: yCoeffs[1], f: yCoeffs[2],
  };
}

/** Invert a 2D affine matrix.  Returns null if the linear part is degenerate. */
function invertAffine(M: AffineMatrix): AffineMatrix | null {
  const det = M.a * M.e - M.b * M.d;
  if (Math.abs(det) < 1e-12) return null;
  const ia =  M.e / det;
  const ib = -M.b / det;
  const id = -M.d / det;
  const ie =  M.a / det;
  return {
    a: ia, b: ib, c: -(ia * M.c + ib * M.f),
    d: id, e: ie, f: -(id * M.c + ie * M.f),
  };
}

/** Extract approximate rotation (degrees) and scale from an affine matrix. */
function decomposeAffine(M: AffineMatrix): { rotation: number; scale: number } {
  // For a pure similarity transform M = s·R, scaleX ≈ scaleY ≈ s
  const scaleX = Math.sqrt(M.a * M.a + M.d * M.d);
  const scaleY = Math.sqrt(M.b * M.b + M.e * M.e);
  return {
    rotation: Math.atan2(M.d, M.a) * (180 / Math.PI),
    scale:    (scaleX + scaleY) / 2,
  };
}

// ---------------------------------------------------------------------------
// RANSAC affine estimation
// ---------------------------------------------------------------------------

/** Reprojection error of one correspondence pair under an affine transform. */
function reprojErr(
  M: AffineMatrix, ax: number, ay: number, bx: number, by: number,
): number {
  const px = M.a * ax + M.b * ay + M.c;
  const py = M.d * ax + M.e * ay + M.f;
  return Math.sqrt((px - bx) ** 2 + (py - by) ** 2);
}

/**
 * Deterministic LCG — avoids crypto overhead and produces consistent results
 * for the same correspondence set across runs.
 */
function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 0x1_0000_0000; };
}

/**
 * RANSAC-based robust affine estimation.
 *
 * Each iteration samples 3 correspondences (minimum for affine), fits an exact
 * affine, counts inliers below RANSAC_THRESHOLD reprojection error, and keeps
 * the model with the most inliers.  The final model is then refined by
 * least-squares over all inliers.
 *
 * Using 3-point minimal samples means the exact solve is equivalent to
 * least-squares (zero residual), so affineLeastSquares is reused.
 */
function ransacAffine(
  pts: KeypointCorrespondence[],
): { matrix: AffineMatrix; inliers: number[] } | null {
  const n = pts.length;
  if (n < 3) return null;

  const rand = lcg(n * 1000 + Math.round(pts[0].ax * 100));
  let bestInliers: number[] = [];
  let bestMatrix: AffineMatrix | null = null;

  for (let iter = 0; iter < RANSAC_MAX_ITERS; iter++) {
    // Sample 3 distinct indices
    const i0 = Math.floor(rand() * n);
    let i1 = Math.floor(rand() * n); while (i1 === i0) i1 = Math.floor(rand() * n);
    let i2 = Math.floor(rand() * n); while (i2 === i0 || i2 === i1) i2 = Math.floor(rand() * n);

    const M = affineLeastSquares([pts[i0], pts[i1], pts[i2]]);
    if (!M) continue;

    const inliers: number[] = [];
    for (let j = 0; j < n; j++) {
      const { ax, ay, bx, by } = pts[j];
      if (reprojErr(M, ax, ay, bx, by) < RANSAC_THRESHOLD) inliers.push(j);
    }

    if (inliers.length > bestInliers.length) {
      bestInliers = inliers;
      bestMatrix  = M;
    }
  }

  if (!bestMatrix || bestInliers.length < RANSAC_MIN_INLIERS) return null;

  // Refine with all inliers (least-squares, lower variance than 3-point exact)
  const refined = affineLeastSquares(bestInliers.map(i => pts[i])) ?? bestMatrix;
  return { matrix: refined, inliers: bestInliers };
}

// ---------------------------------------------------------------------------
// Backward warping (inverse-map bilinear interpolation)
// ---------------------------------------------------------------------------

/**
 * Warp a W×H grayscale image by the *inverse* affine transform.
 *
 * Backward warping: for each destination pixel (dx, dy) we compute the source
 * coordinates via invAffine, then bilinearly interpolate.  This avoids holes
 * that forward-mapping produces.
 *
 * Out-of-bounds source coordinates are filled with mid-gray (128) — neutral
 * for NCC, neither helping nor hurting the similarity score.
 *
 * @param src       Source grayscale Float32Array (W×H, row-major).
 * @param invAffine Inverse of the forward short→movie affine (maps dst → src).
 */
function warpGray(src: Float32Array, invAffine: AffineMatrix): Float32Array {
  const dst = new Float32Array(W * H);

  for (let dy = 0; dy < H; dy++) {
    for (let dx = 0; dx < W; dx++) {
      // Map destination pixel to source coordinates via inverse affine
      const sx = invAffine.a * dx + invAffine.b * dy + invAffine.c;
      const sy = invAffine.d * dx + invAffine.e * dy + invAffine.f;

      if (sx < 0 || sy < 0 || sx >= W - 1 || sy >= H - 1) {
        dst[dy * W + dx] = 128; // out-of-bounds → neutral gray
        continue;
      }

      // Bilinear interpolation
      const x0 = sx | 0, x1 = x0 + 1;   // | 0 is a fast Math.floor for positive values
      const y0 = sy | 0, y1 = y0 + 1;
      const wx = sx - x0, wy = sy - y0;

      dst[dy * W + dx] =
        (1 - wx) * (1 - wy) * src[y0 * W + x0] +
             wx  * (1 - wy) * src[y0 * W + x1] +
        (1 - wx) *      wy  * src[y1 * W + x0] +
             wx  *      wy  * src[y1 * W + x1];
    }
  }

  return dst;
}

// ---------------------------------------------------------------------------
// Normalised Cross-Correlation
// ---------------------------------------------------------------------------

/**
 * Compute Normalised Cross-Correlation (NCC) between two grayscale arrays and
 * map the result to a 0–100 similarity score.
 *
 *   NCC ∈ [−1, 1]
 *   alignedHashSim = (NCC + 1) / 2 × 100
 *
 * Interpretation:
 *   NCC =  1 → identical images      → sim = 100
 *   NCC =  0 → uncorrelated images   → sim =  50  (random baseline)
 *   NCC = −1 → anti-correlated       → sim =   0
 *
 * Returns 50 (neutral) when either image has near-zero variance (uniform).
 */
function nccSim(a: Float32Array, b: Float32Array): number {
  const n = a.length;
  let sumA = 0, sumB = 0;
  for (let i = 0; i < n; i++) { sumA += a[i]; sumB += b[i]; }
  const meanA = sumA / n, meanB = sumB / n;

  let num = 0, varA = 0, varB = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - meanA, db = b[i] - meanB;
    num  += da * db;
    varA += da * da;
    varB += db * db;
  }

  const denom = Math.sqrt(varA * varB);
  if (denom < 1e-9) return 50; // uniform image — return neutral

  const ncc = num / denom; // already in [−1, 1] by Cauchy-Schwarz
  return ((ncc + 1) / 2) * 100;
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/**
 * Attempt affine pre-alignment of a short-clip frame to a movie frame using
 * ORB keypoint correspondences, then return a refined similarity score.
 *
 * The correspondences must be in the 128×72 analysis grid (as returned by
 * matchKeypointsWithCorrespondences in keypoint-signal.ts).
 *
 * ## Why NCC as the "refined hash comparison"
 *
 * After geometric alignment the 128×72 grayscale thumbnails are in the same
 * pixel space as the perceptual hash computation.  NCC on aligned thumbnails
 * is equivalent to computing and comparing average-hash vectors: it measures
 * whether the same visual content is present after the transformation is
 * corrected, which is exactly the failure mode the 13 crop variants miss when
 * the true transform is a rotation or arbitrary zoom.
 *
 * ## Call constraint
 *
 * Only called from enhanceBorderlineSegments.  Never called on the main pipeline.
 */
export function computeHomographyAlignment(
  shortImageData: { data: Uint8ClampedArray | Uint8Array; width: number; height: number },
  movieImageData:  { data: Uint8ClampedArray | Uint8Array; width: number; height: number },
  correspondences: KeypointCorrespondence[],
): HomographyAlignmentResult {
  // Downsample both frames to the 128×72 keypoint analysis grid
  const shortGray = toGray128x72(shortImageData.data, shortImageData.width, shortImageData.height);
  const movieGray = toGray128x72(movieImageData.data, movieImageData.width, movieImageData.height);

  // Baseline NCC before any alignment (used in the "corrected from X to Y" log)
  const priorSim = nccSim(shortGray, movieGray);

  if (correspondences.length < MIN_MATCHES_FOR_ALIGNMENT) {
    return {
      aligned: false, alignedHashSim: 0, priorSim,
      rotation: 0, scale: 1,
      inlierCount: 0, goodMatchCount: correspondences.length,
    };
  }

  // Fit affine: short → movie (forward direction)
  const ransacResult = ransacAffine(correspondences);
  if (!ransacResult) {
    return {
      aligned: false, alignedHashSim: 0, priorSim,
      rotation: 0, scale: 1,
      inlierCount: 0, goodMatchCount: correspondences.length,
    };
  }

  const { matrix: fwdAffine, inliers } = ransacResult;

  // Invert to get dst→src mapping for backward warping
  const invAffine = invertAffine(fwdAffine);
  if (!invAffine) {
    return {
      aligned: false, alignedHashSim: 0, priorSim,
      rotation: 0, scale: 1,
      inlierCount: inliers.length, goodMatchCount: correspondences.length,
    };
  }

  // Warp the short thumbnail to align with the movie frame's geometry
  const warpedShortGray = warpGray(shortGray, invAffine);

  // Refined similarity: NCC between geometry-corrected short and movie
  const alignedHashSim = nccSim(warpedShortGray, movieGray);

  const { rotation, scale } = decomposeAffine(fwdAffine);

  return {
    aligned: true,
    alignedHashSim,
    priorSim,
    rotation,
    scale,
    inlierCount: inliers.length,
    goodMatchCount: correspondences.length,
  };
}
