/**
 * DTW (Dynamic Time Warping) alignment refinement — server/dtw-align.ts
 *
 * Used as a post-processing step after the scene-chunk matching passes.
 * Candidates are segments where the primary walk-based alignment produces an
 * implausible speedRatio (outside [MIN_SPEED_RATIO, MAX_SPEED_RATIO]) or a
 * borderline confidence score.
 *
 * DTW finds the optimal warping path between the short-clip frame sequence
 * and a narrowed candidate movie region, naturally handling cases where the
 * short clip's speed relative to the movie changes mid-segment (slow-mo
 * transitions, variable-speed edits, etc.).
 *
 * Computational cost is O(n × m) and is hard-capped at DTW_MAX_CELLS to
 * prevent blow-up.  DTW is NEVER run against the full movie — only the
 * already-narrowed candidate movie region (± a small buffer).
 */

import { MatchedSegment, FPData } from './matching-engine';

// ---------------------------------------------------------------------------
// Tuning constants — must match the values in matching-engine.ts
// ---------------------------------------------------------------------------

/** Minimum plausible speedRatio (mirrors matching-engine MIN_SPEED_RATIO) */
const MIN_SPEED_RATIO = 0.4;
/** Maximum plausible speedRatio (mirrors matching-engine MAX_SPEED_RATIO) */
const MAX_SPEED_RATIO = 2.5;

/** Linear regression slope clamps (mirrors SLOPE_MIN / SLOPE_MAX) */
const SLOPE_MIN = 0.1;
const SLOPE_MAX = 8.0;

/**
 * Hard cap on n_short × m_movie matrix cells.
 * Limits DTW to at most ~80 000 frameSim calls per segment, keeping each
 * refinement pass well under 200 ms in practice.
 */
const DTW_MAX_CELLS = 80_000;

/**
 * Maximum extra movie frames added as a buffer on each side of the candidate
 * movie region identified by the existing walk.  Generous enough to allow
 * re-alignment across timing shifts; small enough to keep the matrix tight.
 */
const DTW_BUFFER_FRAMES = 50;

/**
 * Confidence range that qualifies a segment for DTW even when its speedRatio
 * is already plausible.  Avoids wasting compute on clearly strong matches.
 */
const DTW_BORDERLINE_MIN = 55;
const DTW_BORDERLINE_MAX = 82;

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

interface DtwAlignment {
  /** Optimal warping path: one entry per short-frame step */
  path: Array<{ si: number; mi: number; sim: number }>;
  /** Linear-regression speedRatio along the path (Δmi / Δsi) */
  speedRatio: number;
  /** Average frame similarity across the entire path (0–100) */
  avgSimilarity: number;
  /** Normalised total path cost: totalCost / pathLength (lower = better) */
  normalizedCost: number;
}

// ---------------------------------------------------------------------------
// Linear regression helper
// ---------------------------------------------------------------------------

function regressionSlope(path: Array<{ si: number; mi: number }>): number {
  const n = path.length;
  if (n < 2) return 1.0;
  if (n === 2) {
    const dsi = path[1].si - path[0].si;
    if (dsi === 0) return 1.0;
    return Math.max(SLOPE_MIN, Math.min(SLOPE_MAX, (path[1].mi - path[0].mi) / dsi));
  }
  let sumX = 0, sumY = 0, sumXX = 0, sumXY = 0;
  for (const { si, mi } of path) {
    sumX += si; sumY += mi;
    sumXX += si * si; sumXY += si * mi;
  }
  const denom = n * sumXX - sumX * sumX;
  if (Math.abs(denom) < 1e-9) return 1.0;
  const slope = (n * sumXY - sumX * sumY) / denom;
  return Math.max(SLOPE_MIN, Math.min(SLOPE_MAX, slope));
}

// ---------------------------------------------------------------------------
// Core DTW algorithm
// ---------------------------------------------------------------------------

/**
 * Run DTW between `siIndices` (short-clip frame indices) and the contiguous
 * movie-frame range [miStart, miEnd].
 *
 * Local cost at cell (i, j):
 *   cost = (100 − simFn(siIndices[i], miStart+j)) / 100
 *   (0 = perfect match, 1 = no similarity)
 *
 * Recurrence (standard accumulated-cost DTW):
 *   dtw[0][0] = cost[0][0]
 *   dtw[i][j] = cost[i][j] + min(dtw[i−1][j−1], dtw[i−1][j], dtw[i][j−1])
 *
 * Returns null when the matrix would exceed DTW_MAX_CELLS or either sequence
 * is empty.
 */
function runDtw(
  siIndices: number[],
  miStart: number,
  miEnd: number,
  simFn: (si: number, mi: number) => number,
): DtwAlignment | null {
  const n = siIndices.length;
  const m = miEnd - miStart + 1;

  if (n === 0 || m === 0) return null;
  if (n * m > DTW_MAX_CELLS) return null;

  // ── Build cost matrix and similarity cache in a single pass ──────────────
  // Using two flat Float32Arrays (row-major) avoids GC pressure from nested arrays.
  const simMat = new Float32Array(n * m);
  const cost   = new Float32Array(n * m);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      const raw   = simFn(siIndices[i], miStart + j);
      const sim   = Math.max(0, Math.min(100, raw));
      simMat[i * m + j] = sim;
      cost  [i * m + j] = (100 - sim) / 100;
    }
  }

  // ── Accumulated-cost DTW ─────────────────────────────────────────────────
  const dtw = new Float32Array(n * m);

  dtw[0] = cost[0];
  for (let j = 1; j < m; j++) dtw[j] = dtw[j - 1] + cost[j];
  for (let i = 1; i < n; i++) dtw[i * m] = dtw[(i - 1) * m] + cost[i * m];

  for (let i = 1; i < n; i++) {
    for (let j = 1; j < m; j++) {
      const diag = dtw[(i - 1) * m + (j - 1)];
      const up   = dtw[(i - 1) * m + j];
      const left = dtw[i * m + (j - 1)];
      dtw[i * m + j] = cost[i * m + j] + Math.min(diag, up, left);
    }
  }

  const totalCost = dtw[(n - 1) * m + (m - 1)];

  // ── Backtrace ─────────────────────────────────────────────────────────────
  // Walk from (n−1, m−1) back to (0, 0), choosing the predecessor with the
  // lowest accumulated cost.
  const rawPath: Array<{ i: number; j: number }> = [];
  let ci = n - 1, cj = m - 1;

  while (ci > 0 || cj > 0) {
    rawPath.push({ i: ci, j: cj });
    if (ci === 0)      { cj--; }
    else if (cj === 0) { ci--; }
    else {
      const diag = dtw[(ci - 1) * m + (cj - 1)];
      const up   = dtw[(ci - 1) * m + cj];
      const left = dtw[ci * m + (cj - 1)];
      const mn   = Math.min(diag, up, left);
      if (mn === diag)      { ci--; cj--; }
      else if (mn === up)   { ci--; }
      else                  { cj--; }
    }
  }
  rawPath.push({ i: 0, j: 0 });
  rawPath.reverse();

  // ── Build output path ─────────────────────────────────────────────────────
  // Deduplicate consecutive identical (i,j) pairs produced by horizontal/
  // vertical steps — we only want one entry per short-frame index.
  const path: Array<{ si: number; mi: number; sim: number }> = [];
  let lastI = -1, lastJ = -1;
  let totalSim = 0;

  for (const { i, j } of rawPath) {
    if (i === lastI && j === lastJ) continue;
    const sim = simMat[i * m + j];
    path.push({ si: siIndices[i], mi: miStart + j, sim });
    totalSim += sim;
    lastI = i; lastJ = j;
  }

  if (path.length === 0) return null;

  const speedRatio     = regressionSlope(path.map(p => ({ si: p.si, mi: p.mi })));
  const avgSimilarity  = totalSim / path.length;
  const normalizedCost = totalCost / path.length;

  return { path, speedRatio, avgSimilarity, normalizedCost };
}

// ---------------------------------------------------------------------------
// Build a MatchedSegment from a DTW alignment result
// ---------------------------------------------------------------------------

function segmentFromDtwPath(
  align: DtwAlignment,
  shortFps: FPData[],
  movieFps: FPData[],
): MatchedSegment {
  const { path, speedRatio } = align;

  const firstSi  = path[0].si;
  const lastSi   = path[path.length - 1].si;
  const firstMi  = path[0].mi;
  const rawMiEnd = path[path.length - 1].mi;

  // Use regression to predict the correct movie endpoint (same logic as
  // acceptSegment in matching-engine.ts — corrects walk-endpoint noise).
  const siSpan   = lastSi - firstSi;
  const regMiEnd = firstMi + Math.round(speedRatio * siSpan);
  const clampedMiEnd = Math.max(0, Math.min(movieFps.length - 1, regMiEnd));
  const miEnd = Math.abs(regMiEnd - rawMiEnd) > 1 ? clampedMiEnd : rawMiEnd;

  // Count short-frame gaps within the matched range
  const inSeq = new Set(path.map(p => p.si));
  let gapCount = 0;
  for (let g = firstSi + 1; g < lastSi; g++) {
    if (!inSeq.has(g)) gapCount++;
  }

  return {
    shortStart:    shortFps[firstSi].timestamp,
    shortEnd:      shortFps[lastSi].timestamp,
    movieStart:    movieFps[firstMi].timestamp,
    movieEnd:      movieFps[miEnd].timestamp,
    confidence:    align.avgSimilarity,
    frameCount:    path.length,
    isApproximate: true,   // DTW-refined segments are always approximate
    gapCount,
    speedRatio,
    matchSequence: path.map(p => ({
      shortTime:  shortFps[p.si].timestamp,
      movieTime:  movieFps[p.mi].timestamp,
      similarity: p.sim,
    })),
    // bestFrameDetail intentionally omitted — DTW is a refinement pass,
    // not a full re-match with heavy signal extraction.
  };
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/**
 * Attempt DTW-based re-alignment for segments that are:
 *
 *  (a) Implausible speedRatio — speedRatio outside [MIN_SPEED_RATIO, MAX_SPEED_RATIO].
 *      The walk-based approach can produce compressed movie spans (speedRatio≈0.1)
 *      when the video has uniform/low-detail frames that anchor the walk at the wrong
 *      movie position.  DTW re-aligns the full short-frame sequence against the
 *      candidate movie region and may recover a plausible speedRatio.
 *
 *  (b) Borderline confidence — confidence in [DTW_BORDERLINE_MIN, DTW_BORDERLINE_MAX).
 *      The rigid frame-step walk can under-perform on speed-varying clips.  DTW
 *      finds the globally optimal frame correspondence and may yield higher
 *      average similarity when the clip's speed relative to the movie is uneven.
 *
 * For each candidate the function:
 *  1. Resolves the short-clip and movie frame-index ranges from timestamps.
 *  2. Expands the movie region by ±DTW_BUFFER_FRAMES (or 20 % of span) to
 *     allow re-alignment beyond the original walk boundaries.
 *  3. Caps the matrix at DTW_MAX_CELLS to prevent O(n×m) blow-up.
 *  4. Runs the DTW accumulation + backtrace.
 *  5. Replaces the segment with the DTW-refined result only when DTW either:
 *       • corrects an implausible speedRatio to a plausible one, OR
 *       • improves average similarity for a borderline segment.
 *     Otherwise the original segment is preserved unchanged.
 *
 * @param segments  Segments to evaluate (output of contextValidateSegments).
 * @param simFn     Frame-pair similarity function — pass `(si, mi) => frameSim(sSet, si, mSet, mi)`.
 * @param shortFps  Short-clip frame data (timestamp, index).
 * @param movieFps  Movie frame data (timestamp, index).
 */
export function refineWithDTW(
  segments: MatchedSegment[],
  simFn: (si: number, mi: number) => number,
  shortFps: FPData[],
  movieFps: FPData[],
): MatchedSegment[] {
  if (segments.length === 0) return segments;

  // Build timestamp → frame-index lookup maps (O(1) per resolve)
  const shortTsToSi = new Map<string, number>();
  shortFps.forEach((fp, si) => shortTsToSi.set(fp.timestamp.toFixed(4), si));

  const movieTsToMi = new Map<string, number>();
  movieFps.forEach((fp, mi) => movieTsToMi.set(fp.timestamp.toFixed(4), mi));

  let dtwCount = 0;
  const result: MatchedSegment[] = [];

  for (const seg of segments) {
    const isImplausibleSpeed =
      seg.speedRatio < MIN_SPEED_RATIO || seg.speedRatio > MAX_SPEED_RATIO;
    const isBorderline =
      seg.confidence >= DTW_BORDERLINE_MIN && seg.confidence < DTW_BORDERLINE_MAX;

    if (!isImplausibleSpeed && !isBorderline) {
      result.push(seg);
      continue;
    }

    const reason = isImplausibleSpeed
      ? `speedRatio=${seg.speedRatio.toFixed(3)} (implausible)`
      : `confidence=${seg.confidence.toFixed(1)}% (borderline)`;

    // ── Resolve short-clip frame indices ─────────────────────────────────
    let siStart = shortTsToSi.get(seg.shortStart.toFixed(4));
    let siEnd   = shortTsToSi.get(seg.shortEnd.toFixed(4));

    // Fallback: linear search when exact timestamp key misses
    if (siStart === undefined) {
      siStart = shortFps.findIndex(fp => fp.timestamp >= seg.shortStart - 0.001);
      if (siStart < 0) siStart = 0;
    }
    if (siEnd === undefined) {
      siEnd = shortFps.findIndex(fp => fp.timestamp >= seg.shortEnd - 0.001);
      if (siEnd < 0) siEnd = shortFps.length - 1;
    }

    if (siEnd <= siStart) {
      result.push(seg);
      continue;
    }

    const siIndices: number[] = [];
    for (let si = siStart; si <= siEnd; si++) siIndices.push(si);

    // ── Resolve movie frame-index range ───────────────────────────────────
    let miStart = movieTsToMi.get(seg.movieStart.toFixed(4));
    let miEnd   = movieTsToMi.get(seg.movieEnd.toFixed(4));

    if (miStart === undefined) {
      miStart = movieFps.findIndex(fp => fp.timestamp >= seg.movieStart - 0.001);
      if (miStart < 0) miStart = 0;
    }
    if (miEnd === undefined) {
      miEnd = movieFps.findIndex(fp => fp.timestamp >= seg.movieEnd - 0.001);
      if (miEnd < 0) miEnd = movieFps.length - 1;
    }

    if (miEnd < miStart) [miStart, miEnd] = [miEnd, miStart];

    // ── Expand movie region with buffer ───────────────────────────────────
    // For implausible-speed segments the walk likely stagnated: the movie span
    // can be nearly 0 frames (speedRatio≈0.1 → the walk compressed a 5-second
    // short against <1 second of movie).  Using 20% of that tiny span as the
    // buffer (≈0 frames) means DTW searches the wrong region entirely.
    // Instead, use the short-clip frame count as the expected span when the
    // segment's movie span is implausibly compressed.
    const baseSpan    = Math.max(1, miEnd - miStart);
    const effectiveSpan = isImplausibleSpeed
      ? Math.max(baseSpan, siIndices.length)   // assume ~1:1 speed as fallback
      : baseSpan;
    const bufSize  = Math.min(DTW_BUFFER_FRAMES, Math.round(effectiveSpan * 0.20));
    let   adjLo    = Math.max(0,                    miStart - bufSize);
    let   adjHi    = Math.min(movieFps.length - 1,  miEnd   + bufSize);
    // Guarantee the window covers at least effectiveSpan frames for implausible speed
    if (isImplausibleSpeed && adjHi - adjLo < effectiveSpan) {
      const centre = Math.round((adjLo + adjHi) / 2);
      adjLo = Math.max(0, centre - Math.floor(effectiveSpan / 2));
      adjHi = Math.min(movieFps.length - 1, adjLo + effectiveSpan);
    }

    // ── Apply cell cap: shrink movie region toward its centre if needed ───
    const n = siIndices.length;
    if (n * (adjHi - adjLo + 1) > DTW_MAX_CELLS) {
      const maxM  = Math.floor(DTW_MAX_CELLS / Math.max(1, n));
      const centre = Math.round((adjLo + adjHi) / 2);
      adjLo = Math.max(0,                    centre - Math.floor(maxM / 2));
      adjHi = Math.min(movieFps.length - 1,  adjLo + maxM - 1);
    }

    // ── Run DTW ───────────────────────────────────────────────────────────
    const align = runDtw(siIndices, adjLo, adjHi, simFn);

    if (!align) {
      console.log(
        `[DTW] Segment [short ${seg.shortStart.toFixed(2)}–${seg.shortEnd.toFixed(2)}s]` +
        ` — ${reason} — DTW skipped (matrix too large or empty), keeping original result.`
      );
      result.push(seg);
      continue;
    }

    dtwCount++;

    const dtwPlausible =
      align.speedRatio >= MIN_SPEED_RATIO && align.speedRatio <= MAX_SPEED_RATIO;

    const dtwImproves =
      (isImplausibleSpeed && dtwPlausible) ||
      (!isImplausibleSpeed && isBorderline && align.avgSimilarity > seg.confidence);

    if (dtwImproves) {
      const refined = segmentFromDtwPath(align, shortFps, movieFps);
      console.log(
        `[DTW] Re-aligned segment [short ${seg.shortStart.toFixed(2)}–${seg.shortEnd.toFixed(2)}s]` +
        ` — original ${reason},` +
        ` DTW-refined alignment found speedRatio=${refined.speedRatio.toFixed(3)}` +
        ` avgSim=${refined.confidence.toFixed(1)}%` +
        ` (normalizedCost=${align.normalizedCost.toFixed(4)})` +
        ` — using DTW result.`
      );
      result.push(refined);
    } else {
      console.log(
        `[DTW] Segment [short ${seg.shortStart.toFixed(2)}–${seg.shortEnd.toFixed(2)}s]` +
        ` — ${reason}` +
        ` — DTW did not find a better alignment` +
        ` (dtwSpeedRatio=${align.speedRatio.toFixed(3)} dtwSim=${align.avgSimilarity.toFixed(1)}%` +
        ` normalizedCost=${align.normalizedCost.toFixed(4)})` +
        `, keeping original result.`
      );
      result.push(seg);
    }
  }

  if (dtwCount > 0) {
    console.log(`[DTW] Ran DTW on ${dtwCount} candidate segment(s).`);
  }

  return result;
}
