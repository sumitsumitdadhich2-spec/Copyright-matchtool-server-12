/**
 * server/keypoint-signal.ts
 *
 * ORB-lite keypoint detector + binary-descriptor matcher for borderline-segment rescue.
 *
 * Pure TypeScript — zero native bindings, zero WASM, zero new npm dependencies,
 * zero Docker image size increase, zero recurring cost.  Satisfies the same
 * self-hosted / no-external-API constraints as the existing MediaPipe / CLIP signals.
 *
 * Algorithm
 * ─────────
 *  1. Nearest-neighbour grayscale downsample to 128×72 (fast, preserves geometry).
 *  2. FAST-9 corner detection (Bresenham circle of radius 3; T=20 brightness threshold).
 *  3. Sort by FAST score → non-maximum suppression (radius 7 px) → top 250 corners.
 *  4. Intensity-centroid orientation in a 9-px radius window (enables rotation-invariant
 *     BRIEF → "rBRIEF", the core of ORB).
 *  5. 256-bit rBRIEF descriptor per keypoint (Gaussian-sampled pair pattern rotated to
 *     keypoint orientation) → 32 bytes per descriptor.
 *  6. Brute-force Hamming matching + Lowe ratio test (threshold 0.75).
 *  7. keypointSim = clamp(goodMatches / min(nA, nB) × 200, 0, 100).
 *
 * Typical second-pass timing: ~2–8 ms per segment pair on a 4-core server
 * (62 500 descriptor pairs × 32-byte Hamming each).
 */

// ---------------------------------------------------------------------------
// Config — all tunable constants in one place
// ---------------------------------------------------------------------------

const ANALYSIS_W       = 128;  // downscaled analysis width (px)
const ANALYSIS_H       =  72;  // downscaled analysis height (px)
const FAST_THRESHOLD   =  20;  // brightness difference threshold for FAST
const FAST_N           =   9;  // consecutive bright/dark pixels required (FAST-9)
const MAX_KEYPOINTS    = 250;  // maximum keypoints per frame after NMS
const NMS_RADIUS       =   7;  // non-maximum suppression radius (px)
const DESCRIPTOR_BITS  = 256;  // rBRIEF descriptor size in bits
const DESCRIPTOR_BYTES = DESCRIPTOR_BITS >>> 3;  // 32
const ORIENTATION_R    =   9;  // intensity-centroid radius (px)
const RATIO_THRESHOLD  = 0.75; // Lowe's ratio test

/** keypointSim scaling: a 50% keypoint match rate → 100% sim */
const SIM_SCALE = 200;

// ---------------------------------------------------------------------------
// FAST-9: 16 Bresenham circle offsets at radius 3
// ---------------------------------------------------------------------------
const CIRCLE16 = [
  [ 0,  3], [ 1,  3], [ 2,  2], [ 3,  1],
  [ 3,  0], [ 3, -1], [ 2, -2], [ 1, -3],
  [ 0, -3], [-1, -3], [-2, -2], [-3, -1],
  [-3,  0], [-3,  1], [-2,  2], [-1,  3],
] as const;

// Quick-reject compass indices (cardinal + diagonal)
const FAST_COMPASS = [0, 4, 8, 12] as const;

// ---------------------------------------------------------------------------
// BRIEF sampling pattern — 256 pairs of (ax, ay, bx, by)
// Generated once with a Gaussian distribution (σ=8) and a fixed seed.
// The same pattern is applied to every frame, rotated to each keypoint's orientation.
// ---------------------------------------------------------------------------
function buildBriefPattern(pairs: number, seed: number): Int8Array {
  // LCG pseudo-random (deterministic)
  let s = seed >>> 0;
  const rand = (): number => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 0x1_0000_0000;
  };
  const gauss = (sigma: number): number => {
    const u = Math.max(1e-9, rand());
    const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
    return Math.max(-15, Math.min(15, Math.round(z * sigma)));
  };
  // Flat layout: [ax0, ay0, bx0, by0, ax1, ay1, bx1, by1, ...]
  const out = new Int8Array(pairs * 4);
  for (let i = 0; i < pairs; i++) {
    out[i * 4    ] = gauss(8);
    out[i * 4 + 1] = gauss(8);
    out[i * 4 + 2] = gauss(8);
    out[i * 4 + 3] = gauss(8);
  }
  return out;
}

const BRIEF_PATTERN = buildBriefPattern(DESCRIPTOR_BITS, 12345);

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

export interface KeypointSignalResult {
  hasKeypoints: boolean;
  /** Keypoint coordinates in the downscaled 128×72 analysis grid */
  keypoints: Array<{ x: number; y: number; angle: number; score: number }>;
  /** Flat: N × DESCRIPTOR_BYTES bytes, row-major (descriptor i = bytes [i*32, (i+1)*32)) */
  descriptors: Uint8Array;
  count: number;
}

export interface KeypointMatchResult {
  goodMatches: number;
  totalA: number;
  totalB: number;
  keypointSim: number;  // 0–100
}

/**
 * Pixel-space coordinate pair for one matched keypoint, in the 128×72
 * downscaled analysis grid that getKeypointSignal() operates in.
 */
export interface KeypointCorrespondence {
  ax: number; ay: number;  // keypoint in frame A (short clip)
  bx: number; by: number;  // keypoint in frame B (movie)
}

export interface KeypointMatchResultWithCorrespondences extends KeypointMatchResult {
  /** Coordinate pairs for all good matches — used by homography-align.ts. */
  correspondences: KeypointCorrespondence[];
}

// ---------------------------------------------------------------------------
// init — pure-TS implementation has no async startup cost; exported for
//        API symmetry with the other signal modules.
// ---------------------------------------------------------------------------
export function initKeypointSignal(): void {
  // Nothing to initialise — no model files, no WASM, no native bindings.
  console.log('[KeypointSignal] ORB-lite ready (pure TypeScript, no external deps).');
}

// ---------------------------------------------------------------------------
// Grayscale helpers
// ---------------------------------------------------------------------------

function toGray(
  src: Uint8ClampedArray | Uint8Array,
  srcW: number, srcH: number,
  dstW: number, dstH: number,
): Float32Array {
  const gray = new Float32Array(dstW * dstH);
  const xScale = srcW / dstW;
  const yScale = srcH / dstH;
  for (let dy = 0; dy < dstH; dy++) {
    const sy = Math.min(srcH - 1, Math.round(dy * yScale));
    const sRow = sy * srcW;
    const dRow = dy * dstW;
    for (let dx = 0; dx < dstW; dx++) {
      const sx = Math.min(srcW - 1, Math.round(dx * xScale));
      const p  = (sRow + sx) * 4;
      gray[dRow + dx] = 0.299 * src[p] + 0.587 * src[p + 1] + 0.114 * src[p + 2];
    }
  }
  return gray;
}

function grayAt(gray: Float32Array, w: number, h: number, x: number, y: number): number {
  const cx = x < 0 ? 0 : x >= w ? w - 1 : x;
  const cy = y < 0 ? 0 : y >= h ? h - 1 : y;
  return gray[cy * w + cx];
}

// ---------------------------------------------------------------------------
// FAST-9 detector
// ---------------------------------------------------------------------------
function detectFAST9(
  gray: Float32Array, w: number, h: number, threshold: number,
): Array<{ x: number; y: number; score: number }> {
  const margin = 4;
  const candidates: Array<{ x: number; y: number; score: number }> = [];

  for (let y = margin; y < h - margin; y++) {
    for (let x = margin; x < w - margin; x++) {
      const center = gray[y * w + x];
      const lo = center - threshold;
      const hi = center + threshold;

      // Quick reject: at least 3 of 4 compass pixels must be bright or dark
      let brightCompass = 0, darkCompass = 0;
      for (const qi of FAST_COMPASS) {
        const v = gray[(y + CIRCLE16[qi][1]) * w + (x + CIRCLE16[qi][0])];
        if (v > hi) brightCompass++;
        else if (v < lo) darkCompass++;
      }
      if (brightCompass < 3 && darkCompass < 3) continue;

      // Full circle: classify all 16 pixels
      const flags = new Int8Array(16);
      for (let i = 0; i < 16; i++) {
        const v = gray[(y + CIRCLE16[i][1]) * w + (x + CIRCLE16[i][0])];
        flags[i] = v > hi ? 1 : v < lo ? -1 : 0;
      }

      // Longest consecutive run (doubled array trick to handle wrap-around)
      let maxB = 0, maxD = 0, cB = 0, cD = 0;
      for (let i = 0; i < 32; i++) {
        const f = flags[i & 15];
        if (f === 1)  { cB++; cD = 0; } else if (f === -1) { cD++; cB = 0; } else { cB = 0; cD = 0; }
        if (cB > maxB) maxB = cB;
        if (cD > maxD) maxD = cD;
      }
      if (Math.max(maxB, maxD) < FAST_N) continue;

      // FAST score: sum of |circle_pixel − center| for all 16 pixels
      let score = 0;
      for (const [dx, dy] of CIRCLE16) {
        score += Math.abs(gray[(y + dy) * w + (x + dx)] - center);
      }
      candidates.push({ x, y, score });
    }
  }
  return candidates;
}

// ---------------------------------------------------------------------------
// Non-maximum suppression
// ---------------------------------------------------------------------------
function nms(
  candidates: Array<{ x: number; y: number; score: number }>,
  radius: number, maxPoints: number,
): Array<{ x: number; y: number; score: number }> {
  candidates.sort((a, b) => b.score - a.score);
  const kept: Array<{ x: number; y: number; score: number }> = [];
  const rSq = radius * radius;
  for (const c of candidates) {
    if (kept.length >= maxPoints) break;
    let ok = true;
    for (const k of kept) {
      const dx = c.x - k.x;
      const dy = c.y - k.y;
      if (dx * dx + dy * dy <= rSq) { ok = false; break; }
    }
    if (ok) kept.push(c);
  }
  return kept;
}

// ---------------------------------------------------------------------------
// Orientation via intensity centroid (IC)
// ---------------------------------------------------------------------------
function computeOrientation(
  gray: Float32Array, w: number, h: number, x: number, y: number,
): number {
  let m10 = 0, m01 = 0;
  const r = ORIENTATION_R;
  const rSq = r * r;
  for (let dy = -r; dy <= r; dy++) {
    for (let dx = -r; dx <= r; dx++) {
      if (dx * dx + dy * dy > rSq) continue;
      const v = grayAt(gray, w, h, x + dx, y + dy);
      m10 += dx * v;
      m01 += dy * v;
    }
  }
  return Math.atan2(m01, m10);
}

// ---------------------------------------------------------------------------
// rBRIEF descriptor
// ---------------------------------------------------------------------------
function computeDescriptor(
  gray: Float32Array, w: number, h: number,
  x: number, y: number, angle: number,
): Uint8Array {
  const desc = new Uint8Array(DESCRIPTOR_BYTES);
  const cos  = Math.cos(angle);
  const sin  = Math.sin(angle);

  for (let i = 0; i < DESCRIPTOR_BITS; i++) {
    const base = i * 4;
    const ax = BRIEF_PATTERN[base    ];
    const ay = BRIEF_PATTERN[base + 1];
    const bx = BRIEF_PATTERN[base + 2];
    const by = BRIEF_PATTERN[base + 3];

    // Rotate sample offsets to keypoint orientation
    const rax = Math.round(cos * ax - sin * ay);
    const ray = Math.round(sin * ax + cos * ay);
    const rbx = Math.round(cos * bx - sin * by);
    const rby = Math.round(sin * bx + cos * by);

    const va = grayAt(gray, w, h, x + rax, y + ray);
    const vb = grayAt(gray, w, h, x + rbx, y + rby);

    if (va < vb) desc[i >>> 3] |= (1 << (i & 7));
  }
  return desc;
}

// ---------------------------------------------------------------------------
// Hamming distance between two 32-byte descriptors (plain byte loop)
// ---------------------------------------------------------------------------
function hamming(a: Uint8Array, aOff: number, b: Uint8Array, bOff: number): number {
  let d = 0;
  for (let i = 0; i < DESCRIPTOR_BYTES; i++) {
    let x = (a[aOff + i] ^ b[bOff + i]) & 0xff;
    x -= (x >>> 1) & 0x55;
    x  = (x & 0x33) + ((x >>> 2) & 0x33);
    d += (x + (x >>> 4)) & 0x0f;
  }
  return d;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Detect ORB-lite keypoints and compute rBRIEF descriptors for one frame.
 * Input imageData must be RGBA (4 bytes/pixel).
 */
export function getKeypointSignal(
  imageData: { data: Uint8ClampedArray | Uint8Array; width: number; height: number },
): KeypointSignalResult {
  const { data, width, height } = imageData;

  // 1. Downsample to analysis grid
  const gray = toGray(data, width, height, ANALYSIS_W, ANALYSIS_H);

  // 2. FAST-9 detection
  const candidates = detectFAST9(gray, ANALYSIS_W, ANALYSIS_H, FAST_THRESHOLD);
  if (candidates.length === 0) {
    return { hasKeypoints: false, keypoints: [], descriptors: new Uint8Array(0), count: 0 };
  }

  // 3. NMS → top MAX_KEYPOINTS
  const kps = nms(candidates, NMS_RADIUS, MAX_KEYPOINTS);

  // 4. Orientation + 5. Descriptor
  const allDesc = new Uint8Array(kps.length * DESCRIPTOR_BYTES);
  const keypoints: KeypointSignalResult['keypoints'] = [];

  for (let i = 0; i < kps.length; i++) {
    const { x, y, score } = kps[i];
    const angle = computeOrientation(gray, ANALYSIS_W, ANALYSIS_H, x, y);
    const desc  = computeDescriptor(gray, ANALYSIS_W, ANALYSIS_H, x, y, angle);
    allDesc.set(desc, i * DESCRIPTOR_BYTES);
    keypoints.push({ x, y, angle, score });
  }

  return { hasKeypoints: kps.length > 0, keypoints, descriptors: allDesc, count: kps.length };
}

/**
 * Brute-force Hamming match with Lowe ratio test.
 * Returns { goodMatches, totalA, totalB, keypointSim }.
 *
 *   keypointSim = clamp(goodMatches / min(nA, nB) × 200, 0, 100)
 *
 * A 50% keypoint match rate maps to keypointSim=100.
 * Typical genuine matches (with moderate crop/angle): 20–50% → sim 40–100.
 * Typical false positives: <10% → sim <20.
 */
export function compareKeypointSignals(
  a: KeypointSignalResult,
  b: KeypointSignalResult,
): KeypointMatchResult {
  const nA = a.count;
  const nB = b.count;

  if (!a.hasKeypoints || !b.hasKeypoints || nA === 0 || nB === 0) {
    return { goodMatches: 0, totalA: nA, totalB: nB, keypointSim: 0 };
  }

  let goodMatches = 0;

  for (let i = 0; i < nA; i++) {
    const aOff = i * DESCRIPTOR_BYTES;
    let d1 = Infinity, d2 = Infinity;

    for (let j = 0; j < nB; j++) {
      const d = hamming(a.descriptors, aOff, b.descriptors, j * DESCRIPTOR_BYTES);
      if (d < d1) { d2 = d1; d1 = d; }
      else if (d < d2) { d2 = d; }
    }

    // Lowe's ratio test: nearest neighbour must be clearly better than second-nearest
    if (d2 > 0 && d1 / d2 < RATIO_THRESHOLD) goodMatches++;
  }

  const keypointSim = Math.min(100, (goodMatches / Math.min(nA, nB)) * SIM_SCALE);
  return { goodMatches, totalA: nA, totalB: nB, keypointSim };
}

/**
 * Same as compareKeypointSignals but also returns the matched coordinate pairs
 * in the 128×72 analysis grid.  The correspondences are consumed by
 * homography-align.ts to compute a geometric pre-alignment for borderline segments.
 *
 * Cost: identical to compareKeypointSignals (same O(nA × nB) Hamming loop) —
 * no extra work beyond recording the matched indices.
 */
export function matchKeypointsWithCorrespondences(
  a: KeypointSignalResult,
  b: KeypointSignalResult,
): KeypointMatchResultWithCorrespondences {
  const nA = a.count;
  const nB = b.count;

  if (!a.hasKeypoints || !b.hasKeypoints || nA === 0 || nB === 0) {
    return { goodMatches: 0, totalA: nA, totalB: nB, keypointSim: 0, correspondences: [] };
  }

  let goodMatches = 0;
  const correspondences: KeypointCorrespondence[] = [];

  for (let i = 0; i < nA; i++) {
    const aOff = i * DESCRIPTOR_BYTES;
    let d1 = Infinity, d2 = Infinity;
    let bestJ = -1;

    for (let j = 0; j < nB; j++) {
      const d = hamming(a.descriptors, aOff, b.descriptors, j * DESCRIPTOR_BYTES);
      if (d < d1) { d2 = d1; d1 = d; bestJ = j; }
      else if (d < d2) { d2 = d; }
    }

    if (d2 > 0 && d1 / d2 < RATIO_THRESHOLD && bestJ >= 0) {
      goodMatches++;
      correspondences.push({
        ax: a.keypoints[i].x,
        ay: a.keypoints[i].y,
        bx: b.keypoints[bestJ].x,
        by: b.keypoints[bestJ].y,
      });
    }
  }

  const keypointSim = Math.min(100, (goodMatches / Math.min(nA, nB)) * SIM_SCALE);
  return { goodMatches, totalA: nA, totalB: nB, keypointSim, correspondences };
}
