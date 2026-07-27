/**
 * Server-side video matching engine — v5 (scene-chunk-first matching)
 *
 * v5 additions over v4:
 *  1. Multi-signal scene cut detection: aHash + dHash + temporal color magnitude.
 *     Catches cuts that v4 missed in outdoor/beach content with similar global color.
 *  2. Scene-chunk-first matching: short clip is pre-split at detected cuts, then
 *     each chunk is matched independently against the full movie. The bidirectional
 *     walk is bounded to [chunkStart, chunkEnd] so it can never cross a scene boundary.
 *  3. Guaranteed full-clip coverage: every chunk that passes Passes 1+2 without a
 *     match gets a forced best-match segment in Pass 3, so no scene is ever skipped.
 */

import * as fs from 'fs';
import * as os from 'os';
import * as readline from 'readline';
import { FrameSignature, VariantHashes } from '../src/shared/fingerprint';
import { loadEmbeddingsFile, cosineSimilarity } from './embedding';
import { loadOrBuildHnswIndex, findNearestMovieFrames, hnswDistToSim100, MovieVectorIndex } from './vector-index';
import { refineWithDTW } from './dtw-align';
import { detectShotBoundaries, isShotBoundaryEnabled } from './shot-boundary';

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface FrameDetail {
  /** Human-readable name of the crop region that produced the best match */
  cropRegion: string;
  /** Hash-based (structural) similarity 0–100 — 84 % weight in final score */
  structureSim: number;
  /** Normalized colorGrid similarity 0–100 */
  colorSim: number;
  /** SkinScoreGrid similarity 0–100 (human / character presence) */
  skinSim: number;
  /** DetailGrid (edge / texture) similarity 0–100 */
  detailSim: number;
  /** 256-bit aHash of the best-matching movie frame (full variant), binary string */
  movieHash: string;
  /** 256-bit aHash of the best-matching short frame (full variant), binary string */
  shortHash: string;
}

export interface FPData {
  frameIndex: number;
  timestamp: number;
  variants: Record<string, VariantHashes>;
  signature?: FrameSignature;
  /** Fraction of frame pixels that were subtitle-masked during extraction (0–1). */
  maskCoverage?: number;
}

export interface MatchedSegment {
  shortStart: number;
  shortEnd: number;
  movieStart: number;
  movieEnd: number;
  confidence: number;
  frameCount: number;
  isApproximate: boolean;
  /** Short-clip frames skipped due to low confidence within this segment */
  gapCount: number;
  /**
   * Effective speed ratio of the short clip relative to the reference movie.
   * Computed via linear regression over the full match sequence.
   *   1.0  = normal speed
   *   0.5  = 0.5× slow-mo (editor slowed clip → clip is longer than movie section)
   *   2.0  = 2× fast-forward (editor sped up clip → clip is shorter than movie section)
   * movieEnd is corrected using this ratio so it always reflects the actual
   * reference-movie span, not just the raw last-matched frame.
   */
  speedRatio: number;
  matchSequence: Array<{
    shortTime: number;
    movieTime: number;
    similarity: number;
  }>;
  /** Per-channel similarity breakdown for the best-matching frame in this segment */
  bestFrameDetail?: FrameDetail;
}

export interface MatchResult {
  segments: MatchedSegment[];
  /** Short-clip time ranges that no segment covers */
  unmatchedRanges: Array<{ shortStart: number; shortEnd: number }>;
}

/** Progress update emitted during matchVideosFromFiles — optional callback, zero algorithm impact. */
export interface MatchProgressInfo {
  phase: 'loading_short' | 'indexing' | 'loading_movie' | 'scanning' | 'matching' | 'finalizing';
  pct: number;            // 0–100 overall progress
  chunkIdx?: number;      // 0-based index of the short-clip scene chunk being processed
  totalChunks?: number;   // total scene chunks in the short clip
  shortStart?: number;    // timestamp (s) of chunk start in the short clip
  shortEnd?: number;      // timestamp (s) of chunk end in the short clip
  segmentsFound?: number; // confirmed matched segments found so far
}

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------

/** Movie-frame search window for Pass 3 grouping and seed verification */
const LOOK_AHEAD = 25;

/**
 * Base half-width of the movie-frame search window during the walk.
 * Extended by frameDrift (user param) + missCount (per missed frame),
 * capped at WALK_LOOK_AHEAD_MAX.
 */
const WALK_LOOK_AHEAD     = 7;
const WALK_LOOK_AHEAD_MAX = 18;

/** Base minimum similarity (%) for a frame to extend the segment walk */
const WALK_MIN_SIM = 50;

/**
 * How many consecutive low-confidence short frames we tolerate before ending
 * a segment. Lower than v4 (12 vs 25) so the walk stops sooner when it
 * drifts past the chunk boundary due to a missed cut.
 */
const GAP_LOOKAHEAD = 12;

/** Relax WALK_MIN_SIM by this many % per 25 matched frames */
const ADAPTIVE_DROP_PER_STEP = 1;
const ADAPTIVE_STEP_FRAMES   = 25;
const ADAPTIVE_FLOOR         = 40;

/** Multi-candidate seeding */
const MAX_SEED_CANDIDATES = 8;

/**
 * SpeedRatio validity window.
 * Segments whose regression-computed speedRatio (movie-frames-per-short-frame)
 * falls outside [MIN_SPEED_RATIO, MAX_SPEED_RATIO] are implausible duration
 * relationships — e.g. a 2 s short-clip window matching a 0.2 s movie window
 * (speedRatio ≈ 0.1) — and are rejected as false positives.
 *
 * Bounds are generous enough to accept genuine slow-motion (0.5×) and
 * timelapse/sped-up (2×) edits while reliably rejecting the ~0.1 cluster.
 */
const MIN_SPEED_RATIO = 0.75; // FIX-3: tightened from 0.4 — rejects stuck-frame artifacts
const MAX_SPEED_RATIO = 1.25; // FIX-3: tightened from 2.5 — real edits stay in this range
/** Candidates closer than this many movie frames are merged (2 s @ 25 fps) */
const SEED_SEPARATION = 50;

/**
 * Slope (speed-ratio) clamps for the speed-tolerant walk.
 * Range covers CapCut's 0.1x super-slow-mo up to 8x fast-forward.
 */
const SLOPE_MIN = 0.1;
const SLOPE_MAX = 8.0;

// ---------------------------------------------------------------------------
// Low-detail / degenerate frame gate
// ---------------------------------------------------------------------------
/**
 * Frames whose detailGrid mean-MAD falls below LOW_DETAIL_SOFT_THRESHOLD *and*
 * whose cell-luminance standard deviation falls below LOW_DETAIL_COLOR_THRESHOLD
 * are considered "low-information" (near-blank, fade-to-black, solid-colour fill).
 *
 * Two severity bands:
 *   hard-low: truly degenerate (near-blank, white-flash).  The movie-side frame
 *             alone is enough to trigger the gate.
 *   soft-low: somewhat flat.  Both the movie AND the short frame must be flagged
 *             before the gate fires, so a simple-but-real frame (sky, clean BG)
 *             only gates when it also matches a similarly flat short frame.
 *
 * Genuine logo-card or solid-colour-scene matches are unaffected because their
 * per-cell luminance *varies* across the 4×4 grid (logo vs. background), giving
 * a colorVar well above LOW_DETAIL_COLOR_THRESHOLD.
 *
 * When the gate fires the required similarity is raised by LOW_DETAIL_SIM_BOOST %.
 * Truly identical (real) matches easily clear this bar; spurious hash collisions
 * on near-blank content (the false-positive root cause) do not.
 */
const LOW_DETAIL_HARD_THRESHOLD  =  5.0;  // detailGrid mean MAD (0–255): definitely degenerate
const LOW_DETAIL_SOFT_THRESHOLD  = 12.0;  // detailGrid mean MAD (0–255): suspicious
const LOW_DETAIL_COLOR_THRESHOLD = 12.0;  // cell-luminance std-dev (0–255): structurally flat
const LOW_DETAIL_SIM_BOOST       = 15;    // extra similarity % required when gate triggers

/** aHash weight vs dHash weight when both are available (no pHash) */
const A_WEIGHT = 0.55;
const D_WEIGHT = 0.45;
/** Weights when all three hashes (a + d + p) are available */
const A_WEIGHT3 = 0.25;
const D_WEIGHT3 = 0.35;
const P_WEIGHT3 = 0.40;

// ---------------------------------------------------------------------------
// Fast Hamming distance using Uint32Array XOR + popcount32
// ---------------------------------------------------------------------------

function popcount32(x: number): number {
  x = x >>> 0;
  x -= (x >>> 1) & 0x55555555;
  x  = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  return (((x + (x >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

function hashToU32(hash: string, words: number): Uint32Array {
  const arr = new Uint32Array(words);
  const len = Math.min(hash.length, words * 32);
  for (let i = 0; i < len; i++) {
    if (hash.charCodeAt(i) === 49 /* '1' */) {
      arr[i >>> 5] |= (1 << (i & 31));
    }
  }
  return arr;
}

function hammingN(
  a: Uint32Array, offsetA: number,
  b: Uint32Array, offsetB: number,
  words: number
): number {
  let d = 0;
  for (let k = 0; k < words; k++) d += popcount32(a[offsetA + k] ^ b[offsetB + k]);
  return d;
}

// ---------------------------------------------------------------------------
// Pre-computed per-set structure for O(1) per-frame lookups
// ---------------------------------------------------------------------------

export interface PreSet {
  fps: FPData[];
  variantNames: string[];
  numVariants: number;
  /** aHash flat: stride A_WORDS u32 per (frame, variant) */
  aFlat: Uint32Array;
  /** Flipped aHash flat (null if fingerprints predate flip support) */
  faFlat: Uint32Array | null;
  /** dHash flat: stride dWords u32 per (frame, variant); null if unavailable */
  dFlat: Uint32Array | null;
  /** Flipped dHash flat */
  fdFlat: Uint32Array | null;
  /** pHash flat: stride pWords u32 per (frame, variant); null if unavailable */
  pFlat: Uint32Array | null;
  aBits: number;
  aWords: number;
  dBits: number;
  dWords: number;
  pBits: number;
  pWords: number;
  variantIdx: Map<string, number>;
  /** Temporal color deltas: 48 floats per frame (frame i minus frame i-1); null if signatures missing */
  tDelta: Float32Array | null;
  /** L2 magnitude of each frame's tDelta */
  tMag: Float32Array | null;
  /**
   * CLIP embedding flat array: stride embDim floats per frame.
   * Null when ENABLE_CLIP_MATCHING is disabled or embeddings were not computed.
   * Frames without an embedding (zero vector) are automatically skipped in frameSim.
   */
  embFlat: Float32Array | null;
  /** Embedding dimension (e.g. 768 for CLIP ViT-B/32). 0 when embFlat is null. */
  embDim: number;
}

function precompute(fps: FPData[]): PreSet {
  const empty: PreSet = {
    fps, variantNames: [], numVariants: 0,
    aFlat: new Uint32Array(0), faFlat: null, dFlat: null, fdFlat: null, pFlat: null,
    aBits: 256, aWords: 8, dBits: 0, dWords: 0, pBits: 64, pWords: 2,
    variantIdx: new Map(), tDelta: null, tMag: null,
    embFlat: null, embDim: 0,
  };
  if (fps.length === 0) return empty;

  const variantNames = Object.keys(fps[0].variants);
  const numVariants  = variantNames.length;
  const variantIdx   = new Map<string, number>();
  variantNames.forEach((n, i) => variantIdx.set(n, i));

  const firstVar = fps[0].variants[variantNames[0]];
  const aBits  = firstVar?.hash?.length || 256;
  const aWords = Math.max(1, Math.ceil(aBits / 32));
  const hasD    = typeof firstVar?.dhash === 'string' && firstVar.dhash.length > 0;
  const hasFlip = typeof firstVar?.fhash === 'string' && firstVar.fhash.length > 0;
  const dBits  = hasD ? firstVar.dhash!.length : 0;
  const dWords = hasD ? Math.max(1, Math.ceil(dBits / 32)) : 0;
  const hasP    = typeof firstVar?.phash === 'string' && firstVar.phash.length > 0;
  const pBits   = hasP ? firstVar.phash!.length : 0;
  const pWords  = hasP ? Math.max(1, Math.ceil(pBits / 32)) : 0;

  const aFlat  = new Uint32Array(fps.length * numVariants * aWords);
  const faFlat = hasFlip ? new Uint32Array(fps.length * numVariants * aWords) : null;
  const dFlat  = hasD ? new Uint32Array(fps.length * numVariants * dWords) : null;
  const fdFlat = hasD && hasFlip ? new Uint32Array(fps.length * numVariants * dWords) : null;
  const pFlat  = hasP ? new Uint32Array(fps.length * numVariants * pWords) : null;

  for (let fi = 0; fi < fps.length; fi++) {
    for (let vi = 0; vi < numVariants; vi++) {
      const v = fps[fi].variants[variantNames[vi]];
      const aOff = (fi * numVariants + vi) * aWords;
      aFlat.set(hashToU32(v?.hash ?? '', aWords), aOff);
      if (faFlat) faFlat.set(hashToU32(v?.fhash ?? '', aWords), aOff);
      if (dFlat) {
        const dOff = (fi * numVariants + vi) * dWords;
        dFlat.set(hashToU32(v?.dhash ?? '', dWords), dOff);
        if (fdFlat) fdFlat.set(hashToU32(v?.fdhash ?? '', dWords), dOff);
      }
      if (pFlat) {
        const pOff = (fi * numVariants + vi) * pWords;
        pFlat.set(hashToU32(v?.phash ?? '', pWords), pOff);
      }
    }
  }

  // Temporal motion deltas from signatures (color grid frame-to-frame change)
  let tDelta: Float32Array | null = null;
  let tMag: Float32Array | null = null;
  const allHaveSig = fps.every(f => f.signature && f.signature.colorGrid.length === 48);
  if (allHaveSig && fps.length > 1) {
    tDelta = new Float32Array(fps.length * 48);
    tMag   = new Float32Array(fps.length);
    for (let fi = 1; fi < fps.length; fi++) {
      const cur  = fps[fi].signature!.colorGrid;
      const prev = fps[fi - 1].signature!.colorGrid;
      let mag = 0;
      for (let k = 0; k < 48; k++) {
        const d = cur[k] - prev[k];
        tDelta[fi * 48 + k] = d;
        mag += d * d;
      }
      tMag[fi] = Math.sqrt(mag);
    }
  }

  return {
    fps, variantNames, numVariants,
    aFlat, faFlat, dFlat, fdFlat, pFlat,
    aBits, aWords, dBits, dWords, pBits, pWords,
    variantIdx, tDelta, tMag,
    embFlat: null, embDim: 0,
  };
}

// ---------------------------------------------------------------------------
// Per-pair similarity helpers
// ---------------------------------------------------------------------------

function pairSim(
  sSet: PreSet, si: number, svi: number,
  mSet: PreSet, mi: number, mvi: number
): number {
  const aWords = sSet.aWords;
  const sAOff = (si * sSet.numVariants + svi) * aWords;
  const mAOff = (mi * mSet.numVariants + mvi) * aWords;

  const aSim = 1 - hammingN(sSet.aFlat, sAOff, mSet.aFlat, mAOff, aWords) / sSet.aBits;

  const useD = sSet.dFlat !== null && mSet.dFlat !== null && sSet.dBits === mSet.dBits && sSet.dBits > 0;
  const useP = sSet.pFlat !== null && mSet.pFlat !== null && sSet.pBits === mSet.pBits && sSet.pBits > 0;

  let normal = aSim;
  if (useD && useP) {
    // All three hash signals available — give pHash the highest weight (most robust)
    const dWords = sSet.dWords;
    const sDOff = (si * sSet.numVariants + svi) * dWords;
    const mDOff = (mi * mSet.numVariants + mvi) * dWords;
    const dSim = 1 - hammingN(sSet.dFlat!, sDOff, mSet.dFlat!, mDOff, dWords) / sSet.dBits;
    const pWords = sSet.pWords;
    const sPOff = (si * sSet.numVariants + svi) * pWords;
    const mPOff = (mi * mSet.numVariants + mvi) * pWords;
    const pSim = 1 - hammingN(sSet.pFlat!, sPOff, mSet.pFlat!, mPOff, pWords) / sSet.pBits;
    normal = A_WEIGHT3 * aSim + D_WEIGHT3 * dSim + P_WEIGHT3 * pSim;
  } else if (useD) {
    // Legacy path: d+a only (old fingerprints without pHash)
    const dWords = sSet.dWords;
    const sDOff = (si * sSet.numVariants + svi) * dWords;
    const mDOff = (mi * mSet.numVariants + mvi) * dWords;
    const dSim = 1 - hammingN(sSet.dFlat!, sDOff, mSet.dFlat!, mDOff, dWords) / sSet.dBits;
    normal = A_WEIGHT * aSim + D_WEIGHT * dSim;
  } else if (useP) {
    // Unlikely (pHash without dHash) — equal split as graceful fallback
    const pWords = sSet.pWords;
    const sPOff = (si * sSet.numVariants + svi) * pWords;
    const mPOff = (mi * mSet.numVariants + mvi) * pWords;
    const pSim = 1 - hammingN(sSet.pFlat!, sPOff, mSet.pFlat!, mPOff, pWords) / sSet.pBits;
    normal = 0.50 * aSim + 0.50 * pSim;
  }

  let best = normal;

  // Flip detection uses aHash/dHash flipped variants (no pHash flip variant)
  if (mSet.faFlat !== null && sSet.aBits === mSet.aBits) {
    const faSim = 1 - hammingN(sSet.aFlat, sAOff, mSet.faFlat, mAOff, aWords) / sSet.aBits;
    let flip = faSim;
    if (useD && mSet.fdFlat !== null) {
      const dWords = sSet.dWords;
      const sDOff = (si * sSet.numVariants + svi) * dWords;
      const mDOff = (mi * mSet.numVariants + mvi) * dWords;
      const fdSim = 1 - hammingN(sSet.dFlat!, sDOff, mSet.fdFlat!, mDOff, dWords) / sSet.dBits;
      flip = A_WEIGHT * faSim + D_WEIGHT * fdSim;
    }
    if (flip > best) best = flip;
  }

  return best * 100;
}

function hashSimFastCross(
  sSet: PreSet, si: number,
  mSet: PreSet, mi: number
): number {
  const svIdx = sSet.variantIdx.get('full') ?? 0;
  let best = 0;
  for (let mvi = 0; mvi < mSet.numVariants; mvi++) {
    const sim = pairSim(sSet, si, svIdx, mSet, mi, mvi);
    if (sim > best) best = sim;
  }
  return best;
}

function hashSimBestCross(
  sSet: PreSet, si: number,
  mSet: PreSet, mi: number
): { sim: number; is9x16: boolean } {
  let best = 0, bestSvi = 0, bestMvi = 0;
  for (let svi = 0; svi < sSet.numVariants; svi++) {
    for (let mvi = 0; mvi < mSet.numVariants; mvi++) {
      const sim = pairSim(sSet, si, svi, mSet, mi, mvi);
      if (sim > best) { best = sim; bestSvi = svi; bestMvi = mvi; }
    }
  }
  // FIX-5: detect if the winning variant pair involves a 9:16 crop — gSim is
  // unreliable in this case because 50 % of the frame area is absent.
  const sName  = sSet.variantNames[bestSvi] ?? '';
  const mName  = mSet.variantNames[bestMvi] ?? '';
  const is9x16 = sName.startsWith('crop_9_16_') || mName.startsWith('crop_9_16_');
  return { sim: best, is9x16 };
}

/** Z-score normalize a 48-value colorGrid per channel (R, G, B) */
function normalizeColorGrid(cg: number[]): Float32Array {
  const out = new Float32Array(48);
  for (let c = 0; c < 3; c++) {
    let mean = 0;
    for (let cell = 0; cell < 16; cell++) mean += cg[cell * 3 + c];
    mean /= 16;
    let variance = 0;
    for (let cell = 0; cell < 16; cell++) {
      const d = cg[cell * 3 + c] - mean;
      variance += d * d;
    }
    const std = Math.max(8, Math.sqrt(variance / 16));
    for (let cell = 0; cell < 16; cell++) {
      out[cell * 3 + c] = (cg[cell * 3 + c] - mean) / std;
    }
  }
  return out;
}

/**
 * Horizontally flip a 4×4 spatial grid stored as a flat array.
 * Used so signatureSim can compare a normal frame against a mirrored one.
 * @param grid       Flat array: 16 cells × valuesPerCell
 * @param vpc        Values per cell (3 for colorGrid RGB, 1 for skin/detail)
 */
function flipGrid4x4(grid: number[], vpc: number): number[] {
  const out = grid.slice();
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 2; col++) {
      const mirrorCol = 3 - col;
      const i1 = (row * 4 + col)       * vpc;
      const i2 = (row * 4 + mirrorCol) * vpc;
      for (let k = 0; k < vpc; k++) {
        const tmp  = out[i1 + k];
        out[i1 + k] = out[i2 + k];
        out[i2 + k] = tmp;
      }
    }
  }
  return out;
}

/** Raw (non-mirror-aware) signature similarity — used internally. */
function _signatureSimRaw(sig1: FrameSignature, sig2: FrameSignature): number {
  let total = 0, count = 0;

  if (sig1.colorGrid.length === 48 && sig2.colorGrid.length === 48) {
    const z1 = normalizeColorGrid(sig1.colorGrid);
    const z2 = normalizeColorGrid(sig2.colorGrid);
    let diff = 0;
    for (let i = 0; i < 48; i++) diff += Math.abs(z1[i] - z2[i]);
    const meanZDiff = diff / 48;
    total += Math.max(0, 1 - meanZDiff / 2);
    count++;
  } else if (sig1.colorGrid.length > 0 && sig1.colorGrid.length === sig2.colorGrid.length) {
    let diff = 0;
    for (let i = 0; i < sig1.colorGrid.length; i++) diff += Math.abs(sig1.colorGrid[i] - sig2.colorGrid[i]);
    total += 1 - diff / (sig1.colorGrid.length * 255);
    count++;
  }
  if (sig1.skinScoreGrid.length > 0 && sig1.skinScoreGrid.length === sig2.skinScoreGrid.length) {
    let diff = 0;
    for (let i = 0; i < sig1.skinScoreGrid.length; i++) diff += Math.abs(sig1.skinScoreGrid[i] - sig2.skinScoreGrid[i]);
    total += 1 - diff / sig1.skinScoreGrid.length;
    count++;
  }
  if (sig1.detailGrid.length > 0 && sig1.detailGrid.length === sig2.detailGrid.length) {
    let maxVal = 1;
    for (let i = 0; i < sig1.detailGrid.length; i++) {
      if (sig1.detailGrid[i] > maxVal) maxVal = sig1.detailGrid[i];
      if (sig2.detailGrid[i] > maxVal) maxVal = sig2.detailGrid[i];
    }
    let diff = 0;
    for (let i = 0; i < sig1.detailGrid.length; i++) diff += Math.abs(sig1.detailGrid[i] - sig2.detailGrid[i]) / maxVal;
    total += 1 - diff / sig1.detailGrid.length;
    count++;
  }
  return count > 0 ? (total / count) * 100 : 50;
}

/**
 * Mirror-aware signature similarity.
 *
 * The hash layer (84 % weight in frameSim) already detects horizontal flips via
 * fhash/fdhash.  But the signature's colorGrid / skinScoreGrid / detailGrid are
 * spatial 4×4 grids — a mirrored clip has its columns reversed, causing a false
 * mismatch at this layer.  We compare both the normal and the horizontally-
 * flipped version of sig2's grids and keep whichever is higher.
 */
function signatureSim(sig1: FrameSignature, sig2: FrameSignature): number {
  const normal = _signatureSimRaw(sig1, sig2);

  // Only bother flipping if the colorGrid is the expected 4×4×3 = 48 values
  if (sig2.colorGrid.length !== 48) return normal;

  const sig2Flipped: FrameSignature = {
    colorGrid:     flipGrid4x4(sig2.colorGrid,     3),
    skinScoreGrid: flipGrid4x4(sig2.skinScoreGrid, 1),
    detailGrid:    flipGrid4x4(sig2.detailGrid,    1),
  };
  const mirrored = _signatureSimRaw(sig1, sig2Flipped);
  return Math.max(normal, mirrored);
}

function temporalSim(sSet: PreSet, si: number, mSet: PreSet, mi: number): number {
  if (!sSet.tDelta || !mSet.tDelta || si === 0 || mi === 0) return -1;
  const magS = sSet.tMag![si];
  const magM = mSet.tMag![mi];
  const STATIC = 6;
  if (magS < STATIC && magM < STATIC) return 78;
  if (magS < STATIC || magM < STATIC) return 38;
  let dot = 0;
  const so = si * 48, mo = mi * 48;
  for (let k = 0; k < 48; k++) dot += sSet.tDelta[so + k] * mSet.tDelta[mo + k];
  const cos = dot / (magS * magM);
  return ((cos + 1) / 2) * 100;
}

/**
 * CLIP embedding cosine similarity, 0–100.
 *
 * Returns –1 when:
 *  – either PreSet has no embeddings (embFlat is null / embDim is 0)
 *  – embDim mismatch between the two sets
 *  – either frame's embedding is an all-zero vector (not computed)
 *
 * NOTE: this function is intentionally NOT called from the brute-force scan
 * (hashSimFastCross).  It is only called from frameSim(), which is used for
 * seed verification and the directional walk — i.e., only on candidate frames
 * already shortlisted by the fast hash pass.
 */
function embeddingSim(sSet: PreSet, si: number, mSet: PreSet, mi: number): number {
  if (!sSet.embFlat || !mSet.embFlat) return -1;
  if (sSet.embDim <= 0 || sSet.embDim !== mSet.embDim) return -1;

  const dim  = sSet.embDim;
  const sOff = si * dim;
  const mOff = mi * dim;

  if (sOff + dim > sSet.embFlat.length || mOff + dim > mSet.embFlat.length) return -1;

  // Skip zero vectors (frames whose CLIP inference failed)
  let sMag = 0, mMag = 0;
  for (let k = 0; k < dim; k++) {
    sMag += sSet.embFlat[sOff + k] * sSet.embFlat[sOff + k];
    mMag += mSet.embFlat[mOff + k] * mSet.embFlat[mOff + k];
  }
  if (sMag < 0.01 || mMag < 0.01) return -1;

  // Embeddings are L2-normalised at write time so a dot product = cosine sim
  let dot = 0;
  for (let k = 0; k < dim; k++) {
    dot += sSet.embFlat[sOff + k] * mSet.embFlat[mOff + k];
  }
  // Map [-1, 1] → [0, 100]
  return ((Math.max(-1, Math.min(1, dot)) + 1) / 2) * 100;
}

export function frameSim(sSet: PreSet, si: number, mSet: PreSet, mi: number): number {
  const { sim: rawHSim, is9x16 } = hashSimBestCross(sSet, si, mSet, mi);
  const sSig = sSet.fps[si].signature;
  const mSig = mSet.fps[mi].signature;
  const tSim = temporalSim(sSet, si, mSet, mi);
  const eSim = embeddingSim(sSet, si, mSet, mi); // -1 if CLIP unavailable

  // Change 3: subtitle-mask correction — if the short clip frame was extracted
  // with subtitle pixels inpainted, scale hSim upward to compensate for the
  // fraction of bits that came from inpainting rather than real content.
  // maskCoverage is 0 for old fingerprints (no change) and gracefully optional.
  const sMask = sSet.fps[si].maskCoverage ?? 0;
  const hSim  = sMask > 0.02
    ? Math.min(100, rawHSim / (1 - sMask * 0.8))
    : rawHSim;

  const hasEmb = eSim >= 0;

  if (sSig && mSig) {
    // FIX-5 + Change 2: 9:16 crop match — gSim (color grid) is unreliable
    // because ~50 % of the frame area is absent.  Zero its weight and boost
    // CLIP to 60 % so semantic understanding dominates over raw pixel diff.
    if (is9x16) {
      if (hasEmb) return hSim * 0.40 + eSim * 0.60;
      return hSim; // no CLIP and no reliable gSim — pure hash score
    }
    const gSim = signatureSim(sSig, mSig);
    if (hasEmb) {
      // All four signals: hash + spatial signature + temporal motion + CLIP
      if (tSim >= 0) return hSim * 0.60 + gSim * 0.12 + tSim * 0.14 + eSim * 0.14;
      return hSim * 0.72 + gSim * 0.14 + eSim * 0.14;
    }
    // Original behaviour (no CLIP) — identical to pre-CLIP code
    if (tSim >= 0) return hSim * 0.70 + gSim * 0.14 + tSim * 0.16;
    return hSim * 0.84 + gSim * 0.16;
  }
  if (hasEmb) return hSim * 0.84 + eSim * 0.16;
  return hSim;
}

// ---------------------------------------------------------------------------
// Change 1: Non-linear perception boost
// ---------------------------------------------------------------------------
/**
 * Applies an exponential perception curve to raw similarity scores.
 *
 * Human perception of "same video" is non-linear: once structural similarity
 * passes ~82 %, the remaining differences (crops, colour grades, text overlays)
 * are largely invisible to a viewer.  This function maps the [82, 100] range
 * through a power curve (exponent 0.2) so that strong structural matches
 * produce confidence values in the 90–98 % range expected by human reviewers.
 *
 * Calibration:  rawScore 85 % → ~94 %,  rawScore 88 % → ~97 %,  100 % → 100 %.
 * Scores below 82 % are returned unchanged to preserve the existing thresholds
 * (minSimilarity, BORDERLINE checks, etc.).
 */
function boostScore(rawScore: number): number {
  const THRESHOLD = 82;
  if (rawScore < THRESHOLD) return rawScore;
  const t       = (rawScore - THRESHOLD) / (100 - THRESHOLD); // normalise [0,1]
  const boosted = Math.pow(t, 0.20);                           // perception curve
  return Math.min(100, THRESHOLD + boosted * (100 - THRESHOLD));
}

// ---------------------------------------------------------------------------
// Frame detail helpers (per-channel breakdown for the best-match frame)
// ---------------------------------------------------------------------------

function formatCropName(name: string): string {
  if (name === 'full')               return 'Full Frame';
  if (name === 'zoom_2_0_center')    return 'Zoom 2.0x Center';
  if (name === 'zoom_1_5_center')    return 'Zoom 1.5x Center';
  if (name === 'zoom_1_5_left')      return 'Zoom 1.5x Left';
  if (name === 'zoom_1_5_right')     return 'Zoom 1.5x Right';
  if (name === 'zoom_1_25_center')   return 'Zoom 1.25x Center';
  if (name === 'zoom_1_25_left')     return 'Zoom 1.25x Left';
  if (name === 'zoom_1_25_right')    return 'Zoom 1.25x Right';
  if (name.startsWith('crop_9_16_')) {
    const idx = parseInt(name.split('_').pop() ?? '0', 10);
    return `9:16 Crop ${idx + 1}`;
  }
  return name;
}

/**
 * Compute per-channel similarity breakdown for the given frame pair
 * plus identify which crop region produced the best structural match.
 */
function getFrameDetail(sSet: PreSet, si: number, mSet: PreSet, mi: number): FrameDetail {
  // ── Best crop region ──
  let bestVariantSim = 0;
  let bestMovieVariant = 'full';
  for (let svi = 0; svi < sSet.numVariants; svi++) {
    for (let mvi = 0; mvi < mSet.numVariants; mvi++) {
      const sim = pairSim(sSet, si, svi, mSet, mi, mvi);
      if (sim > bestVariantSim) {
        bestVariantSim = sim;
        bestMovieVariant = mSet.variantNames[mvi];
      }
    }
  }

  // ── Structure sim (hash-based, 0–100) ──
  const structureSim = hashSimBestCross(sSet, si, mSet, mi).sim;

  // ── Signature-based breakdown ──
  const sSig = sSet.fps[si].signature;
  const mSig = mSet.fps[mi].signature;
  let colorSim = 50, skinSim = 50, detailSim = 50;

  if (sSig && mSig) {
    // Color grid — normalized
    if (sSig.colorGrid.length === 48 && mSig.colorGrid.length === 48) {
      const z1 = normalizeColorGrid(sSig.colorGrid);
      const z2 = normalizeColorGrid(mSig.colorGrid);
      let diff = 0;
      for (let i = 0; i < 48; i++) diff += Math.abs(z1[i] - z2[i]);
      colorSim = Math.max(0, Math.min(100, (1 - diff / 48 / 2) * 100));
    }

    // Skin grid
    if (sSig.skinScoreGrid.length > 0 && sSig.skinScoreGrid.length === mSig.skinScoreGrid.length) {
      let diff = 0;
      for (let i = 0; i < sSig.skinScoreGrid.length; i++)
        diff += Math.abs(sSig.skinScoreGrid[i] - mSig.skinScoreGrid[i]);
      skinSim = Math.max(0, Math.min(100, (1 - diff / sSig.skinScoreGrid.length) * 100));
    }

    // Detail grid
    if (sSig.detailGrid.length > 0 && sSig.detailGrid.length === mSig.detailGrid.length) {
      let maxVal = 1;
      for (let i = 0; i < sSig.detailGrid.length; i++) {
        if (sSig.detailGrid[i] > maxVal) maxVal = sSig.detailGrid[i];
        if (mSig.detailGrid[i] > maxVal) maxVal = mSig.detailGrid[i];
      }
      let diff = 0;
      for (let i = 0; i < sSig.detailGrid.length; i++)
        diff += Math.abs(sSig.detailGrid[i] - mSig.detailGrid[i]) / maxVal;
      detailSim = Math.max(0, Math.min(100, (1 - diff / sSig.detailGrid.length) * 100));
    }
  }

  const movieHash = mSet.fps[mi].variants['full']?.hash ?? mSet.fps[mi].variants[mSet.variantNames[0]]?.hash ?? '';
  const shortHash = sSet.fps[si].variants['full']?.hash ?? sSet.fps[si].variants[sSet.variantNames[0]]?.hash ?? '';

  return {
    cropRegion: formatCropName(bestMovieVariant),
    structureSim,
    colorSim,
    skinSim,
    detailSim,
    movieHash,
    shortHash,
  };
}

// ---------------------------------------------------------------------------
// Yield helper
// ---------------------------------------------------------------------------
function yieldIfNeeded(iter: number, every = 400): Promise<void> | null {
  return iter % every === 0 ? new Promise<void>(r => setImmediate(r)) : null;
}

// ---------------------------------------------------------------------------
// Scene-cut detection — multi-signal (v5)
// ---------------------------------------------------------------------------

function shortConsecutiveSim(sSet: PreSet, si: number): number {
  const svIdx = sSet.variantIdx.get('full') ?? 0;
  const aWords = sSet.aWords;
  const off1 = ((si - 1) * sSet.numVariants + svIdx) * aWords;
  const off2 = (si       * sSet.numVariants + svIdx) * aWords;
  return (1 - hammingN(sSet.aFlat, off1, sSet.aFlat, off2, aWords) / sSet.aBits) * 100;
}

/**
 * Classify a single frame's intrinsic information content using its signature.
 *
 * Returns:
 *   'hard-low' — truly degenerate: near-blank, fade-to-black, white-flash.
 *                detailGrid mean < LOW_DETAIL_HARD_THRESHOLD AND colorVar < LOW_DETAIL_COLOR_THRESHOLD.
 *   'soft-low' — partially flat: low texture + uniform colour.
 *                detailGrid mean < LOW_DETAIL_SOFT_THRESHOLD AND colorVar < LOW_DETAIL_COLOR_THRESHOLD.
 *   'normal'   — has meaningful visual content (returned when signature is absent too).
 *
 * NOTE: frames without a signature always return 'normal' so the gate is never
 * triggered on fingerprints that pre-date signature computation.
 */
function frameInfoLevel(fp: FPData): 'normal' | 'soft-low' | 'hard-low' {
  const sig = fp.signature;
  if (!sig || sig.detailGrid.length === 0) return 'normal';

  // detailGrid: 16 values, each = mean-absolute-deviation of grayscale in a 4×4 cell (0–255)
  const detailMean = sig.detailGrid.reduce((a, v) => a + v, 0) / sig.detailGrid.length;

  // colorVar: std-dev of per-cell luminance across the 4×4 spatial grid.
  // A solid near-black or near-white frame has colorVar ≈ 0.
  // A logo card (logo vs background) has distinct cells → high colorVar → passes as 'normal'.
  let colorVar = 128; // default: assume structured when colorGrid unavailable
  if (sig.colorGrid.length === 48) {
    let lumSum = 0;
    const lums = new Array<number>(16);
    for (let i = 0; i < 16; i++) {
      const l = 0.299 * sig.colorGrid[i * 3]
              + 0.587 * sig.colorGrid[i * 3 + 1]
              + 0.114 * sig.colorGrid[i * 3 + 2];
      lums[i] = l;
      lumSum += l;
    }
    const lumMean = lumSum / 16;
    colorVar = Math.sqrt(lums.reduce((a, v) => a + (v - lumMean) ** 2, 0) / 16);
  }

  if (detailMean < LOW_DETAIL_HARD_THRESHOLD && colorVar < LOW_DETAIL_COLOR_THRESHOLD) return 'hard-low';
  if (detailMean < LOW_DETAIL_SOFT_THRESHOLD && colorVar < LOW_DETAIL_COLOR_THRESHOLD) return 'soft-low';
  return 'normal';
}

/**
 * Detect scene cuts in the short clip using three independent signals.
 * A frame is a cut if ANY signal exceeds its threshold:
 *
 *  Signal 1 — aHash consecutive similarity < aThreshold (25)
 *              aHash is brightness-distribution based; only catches very dramatic
 *              global lighting changes (real hard cuts), not camera motion.
 *
 *  Signal 2 — dHash consecutive similarity < dThreshold (28)
 *              dHash is gradient/edge based; only catches very dramatic edge pattern
 *              changes, not pans/zooms within the same scene.
 *
 *  Signal 3 — Temporal color-grid magnitude > colorMagThreshold (100)
 *              L2 norm of frame-to-frame color grid delta (colorGrid is 0-255 per
 *              channel, 48 values).  A real hard cut between visually distinct scenes
 *              produces tMag >> 100.  Camera motion within a scene produces tMag < 50.
 *              Old value (28) ≈ 4 per cell change = 1.6 % of full range — far too
 *              sensitive; triggered on any pan/zoom and caused 49 false cuts.
 */
function detectSceneCuts(
  sSet: PreSet,
  aThreshold        = 32,  // FIX-1: tightened from 25 — catches hard cuts reliably
  dThreshold        = 34,  // FIX-1: tightened from 28
  colorMagThreshold = 85   // FIX-1: lowered from 100 — more sensitive to color jumps
): Uint8Array {
  const isCut = new Uint8Array(sSet.fps.length);

  for (let si = 1; si < sSet.fps.length; si++) {
    // Signal 1: aHash
    const aSim = shortConsecutiveSim(sSet, si);
    if (aSim < aThreshold) { isCut[si] = 1; continue; }

    // Signal 2: dHash (if available)
    if (sSet.dFlat && sSet.dBits > 0) {
      const svIdx  = sSet.variantIdx.get('full') ?? 0;
      const dWords = sSet.dWords;
      const off1   = ((si - 1) * sSet.numVariants + svIdx) * dWords;
      const off2   = (si       * sSet.numVariants + svIdx) * dWords;
      const dSim   = (1 - hammingN(sSet.dFlat, off1, sSet.dFlat, off2, dWords) / sSet.dBits) * 100;
      if (dSim < dThreshold) { isCut[si] = 1; continue; }
    }

    // Signal 3: temporal color magnitude
    if (sSet.tMag) {
      if (sSet.tMag[si] > colorMagThreshold) { isCut[si] = 1; continue; }
    }
  }

  return isCut;
}

/**
 * Split FPData array into scene chunks at detected cut positions.
 * Returns array of {start, end} frame index pairs (inclusive).
 */
function splitBySceneCuts(
  fps: FPData[],
  isCut: Uint8Array
): Array<{ start: number; end: number }> {
  const chunks: Array<{ start: number; end: number }> = [];
  let start = 0;
  for (let i = 1; i <= fps.length; i++) {
    if (i === fps.length || isCut[i]) {
      chunks.push({ start, end: i - 1 });
      start = i;
    }
  }
  return chunks;
}

// ---------------------------------------------------------------------------
// Directional walk — bounded to a scene chunk [siMin, siMax]
// ---------------------------------------------------------------------------

interface RawSeq { si: number; mi: number; sim: number }

function estimateSlope(seq: RawSeq[], seedSi: number, seedMi: number): number {
  const pts: Array<{ si: number; mi: number }> = [{ si: seedSi, mi: seedMi }];
  const start = Math.max(0, seq.length - 25);
  for (let i = start; i < seq.length; i++) pts.push({ si: seq[i].si, mi: seq[i].mi });
  // Lowered from 8 → 4 so slope converges faster for short slow-mo sequences
  if (pts.length < 4) return 1;

  let minSi = Infinity, maxSi = -Infinity, minPt = pts[0], maxPt = pts[0];
  for (const p of pts) {
    if (p.si < minSi) { minSi = p.si; minPt = p; }
    if (p.si > maxSi) { maxSi = p.si; maxPt = p; }
  }
  const siSpan = maxPt.si - minPt.si;
  // Lowered from 6 → 3 to allow early detection of duplicate frames (slow-mo)
  if (siSpan < 3) return 1;
  const slope = (maxPt.mi - minPt.mi) / siSpan;
  if (!isFinite(slope)) return 1;
  return Math.min(SLOPE_MAX, Math.max(SLOPE_MIN, slope));
}

/**
 * Linear-regression slope of movie-frame index vs short-frame index over a
 * complete walk sequence.  Returns Δmi / Δsi — the effective speed ratio:
 *   1.0 = normal speed
 *   0.5 = 0.5× slow-mo (editor slowed clip; clip is longer than movie section)
 *   2.0 = 2× fast-forward (editor sped clip up; clip is shorter)
 *
 * Regression over ALL matched frames is more robust than just using the first
 * and last points, which are sensitive to noise in the walk endpoints.
 */
function computeRegressionSlope(seq: RawSeq[]): number {
  if (seq.length < 2) return 1.0;
  if (seq.length === 2) {
    const span = seq[1].si - seq[0].si;
    if (span === 0) return 1.0;
    return Math.max(SLOPE_MIN, Math.min(SLOPE_MAX, (seq[1].mi - seq[0].mi) / span));
  }
  let n = 0, sumX = 0, sumY = 0, sumXX = 0, sumXY = 0;
  for (const p of seq) {
    sumX  += p.si;        sumY  += p.mi;
    sumXX += p.si * p.si; sumXY += p.si * p.mi;
    n++;
  }
  const denom = n * sumXX - sumX * sumX;
  if (Math.abs(denom) < 1e-9) return 1.0;
  const slope = (n * sumXY - sumX * sumY) / denom;
  return Math.max(SLOPE_MIN, Math.min(SLOPE_MAX, slope));
}

/**
 * Walk away from (startSi, startMi) in one direction, bounded to [siMin, siMax].
 *
 * v5: siMin/siMax enforce chunk boundaries so the walk cannot cross scene cuts.
 *     isCut is kept as an additional safety check.
 */
function walkOneDir(
  sSet: PreSet,
  mSet: PreSet,
  startSi: number,
  startMi: number,
  usedShort: Uint8Array,
  direction: 1 | -1,
  isCut: Uint8Array,
  frameDrift: number,
  siMin: number = 0,
  siMax: number = sSet.fps.length - 1
): RawSeq[] {
  const seq: RawSeq[] = [];
  let lastGoodSi = startSi;
  let lastGoodMi = startMi;
  let missCount  = 0;
  let slope      = 1;

  for (
    let nextSi = startSi + direction;
    direction === 1 ? nextSi <= siMax : nextSi >= siMin;
    nextSi += direction
  ) {
    if (usedShort[nextSi]) break;

    // Respect scene cuts (shouldn't trigger inside a chunk, but kept as safety)
    if (direction === 1  && isCut[nextSi])     break;
    if (direction === -1 && isCut[nextSi + 1]) break;

    const adaptiveMin = Math.max(
      ADAPTIVE_FLOOR,
      WALK_MIN_SIM - Math.floor(seq.length / ADAPTIVE_STEP_FRAMES) * ADAPTIVE_DROP_PER_STEP
    );

    const expectedMi = lastGoodMi + Math.round(slope * (nextSi - lastGoodSi));
    const baseHalf = Math.min(WALK_LOOK_AHEAD_MAX, WALK_LOOK_AHEAD + frameDrift + missCount);
    const half = Math.min(WALK_LOOK_AHEAD_MAX, baseHalf);
    const lo = Math.max(0, expectedMi - half);
    const hi = Math.min(mSet.fps.length - 1, expectedMi + half);
    if (lo > hi) break;

    let best = 0, bestMi = -1;
    for (let mi = lo; mi <= hi; mi++) {
      const s = frameSim(sSet, nextSi, mSet, mi);
      if (s > best) { best = s; bestMi = mi; }
    }

    if (best >= adaptiveMin && bestMi >= 0) {
      // ── Low-detail gate (primary false-positive filter) ──────────────────
      // Transitions, fades, and near-blank movie frames have low visual
      // information.  Their hashes match *any* similarly flat short-clip
      // frame, producing the frozen-movieTime / speedRatio≈0.1 pattern.
      //
      // Gate logic:
      //   • hard-low movie frame alone → gate fires (no useful anchor).
      //   • soft-low movie + soft-low short → gate fires (both sides flat).
      //   • soft-low on one side only → normal signals decide (logo-card etc.).
      //
      // When the gate fires we require extra similarity (LOW_DETAIL_SIM_BOOST)
      // before accepting the match.  Genuinely-identical frames (real content)
      // easily clear this bar; spurious collisions on near-blank data do not.
      // Frames without a signature (no detailGrid) are always treated as normal.
      const sLevel = frameInfoLevel(sSet.fps[nextSi]);
      const mLevel = frameInfoLevel(mSet.fps[bestMi]);
      const gateActive =
        mLevel === 'hard-low' ||
        (mLevel === 'soft-low' && sLevel !== 'normal') ||
        (sLevel === 'soft-low' && mLevel !== 'normal');

      if (gateActive) {
        const boostedMin = Math.min(96, adaptiveMin + LOW_DETAIL_SIM_BOOST);
        if (best < boostedMin) {
          const mSig = mSet.fps[bestMi].signature;
          const sSig = sSet.fps[nextSi].signature;
          const mDetailStr = mSig
            ? (mSig.detailGrid.reduce((a, v) => a + v, 0) / mSig.detailGrid.length).toFixed(1)
            : 'n/a';
          const sDetailStr = sSig
            ? (sSig.detailGrid.reduce((a, v) => a + v, 0) / sSig.detailGrid.length).toFixed(1)
            : 'n/a';
          console.log(
            `[Matcher] Low-detail skip: movie frame at ${mSet.fps[bestMi].timestamp.toFixed(2)}s` +
            ` (detailScore=${mDetailStr}, level=${mLevel})` +
            ` short at ${sSet.fps[nextSi].timestamp.toFixed(2)}s` +
            ` (detailScore=${sDetailStr}, level=${sLevel})` +
            ` — sim=${best.toFixed(1)}% < boosted min ${boostedMin}%`
          );
          missCount++;
          if (missCount >= GAP_LOOKAHEAD) break;
          continue;
        }
      }
      // ─────────────────────────────────────────────────────────────────────

      seq.push({ si: nextSi, mi: bestMi, sim: best });
      lastGoodSi = nextSi;
      lastGoodMi = bestMi;
      missCount  = 0;
      slope = estimateSlope(seq, startSi, startMi);
    } else {
      missCount++;
      if (missCount >= GAP_LOOKAHEAD) break;
    }
  }

  return seq;
}

// ---------------------------------------------------------------------------
// Gap interpolation — bridge 1-2 skipped short frames within a sequence
// ---------------------------------------------------------------------------

/**
 * When the directional walk encounters 1–2 consecutive short frames that
 * fall below the similarity threshold (motion blur, compression artifact,
 * on-screen text, brief brightness spike) it skips them and keeps going.
 * Those skipped frames leave a gap in the sequence (si jumps by 2 or 3).
 *
 * This pass fills those gaps with linearly-interpolated movie indices and
 * the averaged confidence of their immediate neighbours — implementing the
 * "interpolate / bridge 1-2 dropped frames" behaviour described in the
 * reference algorithm:
 *
 *   "Agar pichla frame match hai aur agla frame match hai, toh algorithm
 *    ko us 1 frame ke error ko ignore karke block ko continue rakhna
 *    chahiye."
 *
 * The interpolated frames are flagged via their sim value being the
 * average of neighbours — they do not inflate the real confidence score.
 */
function fillSequenceGaps(seq: RawSeq[], maxFillGap = 2): RawSeq[] {
  if (seq.length < 2) return seq;
  const out: RawSeq[] = [seq[0]];
  for (let i = 1; i < seq.length; i++) {
    const prev = seq[i - 1];
    const curr = seq[i];
    const siGap = curr.si - prev.si; // how many short-clip frames were skipped
    if (siGap > 1 && siGap <= maxFillGap + 1) {
      // Fill each skipped frame with a linearly interpolated movie position
      for (let g = 1; g < siGap; g++) {
        const t = g / siGap;
        out.push({
          si:  prev.si + g,
          mi:  Math.round(prev.mi + t * (curr.mi - prev.mi)),
          sim: (prev.sim + curr.sim) / 2, // average of neighbours
        });
      }
    }
    out.push(curr);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Build one segment bidirectionally from a seed, bounded to [siMin, siMax]
// ---------------------------------------------------------------------------

function buildSegment(
  sSet: PreSet,
  mSet: PreSet,
  seedSi: number,
  seedMi: number,
  seedSim: number,
  usedShort: Uint8Array,
  isCut: Uint8Array,
  frameDrift: number,
  siMin: number = 0,
  siMax: number = sSet.fps.length - 1
): RawSeq[] {
  const backwardSeq = walkOneDir(sSet, mSet, seedSi, seedMi, usedShort, -1, isCut, frameDrift, siMin, siMax);
  const forwardSeq  = walkOneDir(sSet, mSet, seedSi, seedMi, usedShort,  1, isCut, frameDrift, siMin, siMax);

  backwardSeq.reverse();
  const raw = [...backwardSeq, { si: seedSi, mi: seedMi, sim: seedSim }, ...forwardSeq];
  // Bridge 1-2 frame gaps: fills skipped frames (motion blur / compression artifact)
  return fillSequenceGaps(raw);
}

// ---------------------------------------------------------------------------
// Compute unmatched short-clip ranges
// ---------------------------------------------------------------------------

function computeUnmatched(
  shortFps: FPData[],
  usedShort: Uint8Array
): Array<{ shortStart: number; shortEnd: number }> {
  const ranges: Array<{ shortStart: number; shortEnd: number }> = [];
  let rangeStart = -1;

  for (let i = 0; i <= shortFps.length; i++) {
    const free = i < shortFps.length && !usedShort[i];
    if (free) {
      if (rangeStart < 0) rangeStart = i;
    } else {
      if (rangeStart >= 0) {
        ranges.push({
          shortStart: shortFps[rangeStart].timestamp,
          shortEnd:   shortFps[i - 1].timestamp
        });
        rangeStart = -1;
      }
    }
  }
  return ranges;
}

// ---------------------------------------------------------------------------
// Accept a raw sequence as a MatchedSegment (with optional frame detail)
// ---------------------------------------------------------------------------

function acceptSegment(
  seq: RawSeq[],
  shortFps: FPData[],
  movieFps: FPData[],
  isApproximate: boolean,
  sSet?: PreSet,
  mSet?: PreSet
): MatchedSegment {
  const avgConf = boostScore(seq.reduce((s, f) => s + f.sim, 0) / seq.length);

  const firstSi = seq[0].si;
  const lastSi  = seq[seq.length - 1].si;
  const inSeq   = new Set(seq.map(f => f.si));
  let gapCount  = 0;
  for (let g = firstSi + 1; g < lastSi; g++) {
    if (!inSeq.has(g)) gapCount++;
  }

  // ── Speed-ratio correction ────────────────────────────────────────────────
  // Compute Δmi/Δsi via linear regression over the full matched sequence.
  // This is more robust than using just the first/last endpoints, which are
  // sensitive to walk-endpoint noise and optical-flow interpolation artifacts.
  //
  // Examples:
  //   regSlope ≈ 1.0 → normal speed
  //   regSlope ≈ 0.5 → clip was slowed 0.5× (3 s clip from 1.5 s of movie)
  //   regSlope ≈ 2.0 → clip was sped up 2× (1.5 s clip from 3 s of movie)
  const speedRatio = computeRegressionSlope(seq);

  // Use regression to predict the correct movie endpoint rather than trusting
  // the raw last-matched frame, which may be off when the walk slope drifted.
  const siSpan       = lastSi - firstSi;
  const rawMiEnd     = seq[seq.length - 1].mi;
  const regMiEnd     = seq[0].mi + Math.round(speedRatio * siSpan);
  const clampedMiEnd = Math.max(0, Math.min(movieFps.length - 1, regMiEnd));

  // Only adopt the regression-corrected endpoint when it differs meaningfully
  // from the raw walk endpoint (> 1 frame) — avoids unnecessary jitter on
  // normal-speed content where the walk endpoint is already accurate.
  const miEnd = Math.abs(regMiEnd - rawMiEnd) > 1 ? clampedMiEnd : rawMiEnd;

  // Find best frame for detail computation
  let bestFrameDetail: FrameDetail | undefined;
  if (sSet && mSet) {
    let bestSim = -1, bestSi = seq[0].si, bestMi = seq[0].mi;
    for (const f of seq) {
      if (f.sim > bestSim) { bestSim = f.sim; bestSi = f.si; bestMi = f.mi; }
    }
    bestFrameDetail = getFrameDetail(sSet, bestSi, mSet, bestMi);
  }

  return {
    shortStart: shortFps[firstSi].timestamp,
    shortEnd:   shortFps[lastSi].timestamp,
    movieStart: movieFps[seq[0].mi].timestamp,
    movieEnd:   movieFps[miEnd].timestamp,
    confidence: avgConf,
    frameCount: seq.length,
    isApproximate,
    gapCount,
    speedRatio,
    matchSequence: seq.map(f => ({
      shortTime: shortFps[f.si].timestamp,
      movieTime: movieFps[f.mi].timestamp,
      similarity: f.sim
    })),
    bestFrameDetail
  };
}

// ---------------------------------------------------------------------------
// Post-process: merge temporally adjacent segments that belong to the same run
// ---------------------------------------------------------------------------

/**
 * Context-aware validation of low-confidence segments.
 *
 * Segments accepted only at Pass-2 threshold (40–82 %) are kept only when
 * at least one high-confidence neighbour confirms the movie timeline is
 * progressing forward consistently.  This matches the reference algorithm:
 *
 *   "pichle scenes match hone ki wajah se isko validate kar diya gaya"
 *   (Segment 9, 10 frames / 89 % — accepted because prior segments formed
 *   a solid timeline.)
 *
 * Segments that fail this check are dropped; their clip frames become
 * unmatchedRanges (altered / third-party content detection).
 */
function contextValidateSegments(segs: MatchedSegment[]): MatchedSegment[] {
  if (segs.length <= 1) return segs;

  const MIN_NEIGHBOUR_CONF = 85; // neighbour must be at least this confident
  const MAX_MOVIE_JUMP     = 10; // movie time jump > this (s) is suspicious

  const out: MatchedSegment[] = [];
  for (let i = 0; i < segs.length; i++) {
    const seg = segs[i];

    // High-confidence or long segments are always kept
    if (!seg.isApproximate || seg.confidence >= 80 || seg.frameCount >= 20) {
      out.push(seg); continue;
    }

    // Low-confidence short segment — validate against neighbours
    const prev = out.length > 0 ? out[out.length - 1] : null;
    const next = i < segs.length - 1 ? segs[i + 1] : null;

    const prevGood = prev !== null
      && prev.confidence >= MIN_NEIGHBOUR_CONF
      && seg.movieStart  >= prev.movieEnd - 1.0
      && (seg.movieStart - prev.movieEnd) <= MAX_MOVIE_JUMP;

    const nextGood = next !== null
      && next.confidence >= MIN_NEIGHBOUR_CONF
      && next.movieStart >= seg.movieEnd - 1.0
      && (next.movieStart - seg.movieEnd) <= MAX_MOVIE_JUMP;

    if (prevGood || nextGood) {
      out.push(seg);
    } else {
      console.log(
        `[Matcher] Context-drop: seg [${seg.shortStart.toFixed(2)}–` +
        `${seg.shortEnd.toFixed(2)}s] conf ${seg.confidence.toFixed(1)}%` +
        ` frameCount=${seg.frameCount} — no valid context neighbour.`
      );
      // frames left free → appear in unmatchedRanges
    }
  }
  return out;
}

/**
 * SpeedRatio validation — rejects segments whose temporal duration ratio is
 * implausible (too far from 1.0 in either direction).
 *
 * A speedRatio of ~0.1 means the algorithm matched a long short-clip window
 * (e.g. 2 s) against an implausibly short movie window (e.g. 0.2 s).  This
 * is never a real speed edit; it is a false-positive produced by the walk
 * locking onto a high-similarity frame cluster that happens to be temporally
 * compressed in the movie.
 *
 * Segments with speedRatio ∈ [MIN_SPEED_RATIO, MAX_SPEED_RATIO] are kept;
 * all others are dropped and their short-clip frames become unmatchedRanges.
 */
function speedRatioFilterSegments(segs: MatchedSegment[]): MatchedSegment[] {
  const out: MatchedSegment[] = [];
  for (const seg of segs) {
    if (seg.speedRatio >= MIN_SPEED_RATIO && seg.speedRatio <= MAX_SPEED_RATIO) {
      out.push(seg);
    } else {
      console.log(
        `[Matcher] SpeedRatio-drop: seg [${seg.shortStart.toFixed(2)}–` +
        `${seg.shortEnd.toFixed(2)}s] speedRatio=${seg.speedRatio.toFixed(3)}` +
        ` — implausible duration ratio, rejected.`
      );
    }
  }
  return out;
}

/**
 * Sequence-consistency validation — rejects isolated segments whose movie
 * timeline position is a large, one-off jump away from both surrounding
 * neighbours.
 *
 * The short clip always plays linearly, so the movieStart of each segment
 * (sorted by shortStart) should progress monotonically in roughly the same
 * direction.  An isolated segment that jumps far from the previous segment
 * AND from which the next segment also jumps far — but where both neighbours
 * are close to each other — is an outlier: a false-positive match in a
 * distant, unrelated part of the movie.
 *
 * Runs of multiple consecutive segments at a new movie location are NOT
 * dropped — they represent genuine repeated / reused footage.  A run is
 * detected by the fact that the next segment continues the new location
 * (gapAfter ≤ JUMP_THRESHOLD), so the outlier condition (large gapAfter)
 * does not fire.
 *
 * This filter receives the list that has already been through speedRatio
 * filtering, so that already-removed false positives do not distort the
 * neighbour-gap calculations.  All drops are independently logged with
 * [Matcher] Sequence-drop so they can be audited separately from
 * [Matcher] SpeedRatio-drop events.
 */
function sequenceConsistencyFilter(segs: MatchedSegment[]): MatchedSegment[] {
  if (segs.length <= 2) return segs; // need at least 3 to detect isolation

  // A jump larger than this (seconds in movie time) from both neighbours
  // is considered suspicious.  30 s allows legitimate forward scene cuts
  // while reliably catching the 400 s+ outliers we observe as false positives.
  const JUMP_THRESHOLD = 30;

  // ── Pass 1: Mark candidates ──────────────────────────────────────────────
  // Flag every segment that is far (> JUMP_THRESHOLD) from BOTH immediate
  // neighbours as suspicious.  This is only a candidate list — a genuine
  // good segment sandwiched between two bad ones will also be flagged here.
  // Pass 2 resolves the ambiguity by using non-suspicious context instead.
  const suspicious = new Array<boolean>(segs.length).fill(false);
  for (let i = 1; i < segs.length - 1; i++) {
    const prev = segs[i - 1], seg = segs[i], next = segs[i + 1];
    if (
      Math.abs(seg.movieStart  - prev.movieEnd) > JUMP_THRESHOLD &&
      Math.abs(next.movieStart - seg.movieEnd)  > JUMP_THRESHOLD
    ) {
      suspicious[i] = true;
    }
  }

  // ── Pass 2: Confirm drops with trusted (non-suspicious) context ──────────
  // For each suspicious segment, skip over other suspicious neighbours when
  // searching for context.  This ensures a good segment that happens to sit
  // between two bad ones is evaluated against the wider non-suspicious thread
  // and found to be close to it — so it is correctly kept.
  const dropMask = new Array<boolean>(segs.length).fill(false);

  for (let i = 0; i < segs.length; i++) {
    if (!suspicious[i]) continue;

    // Nearest non-suspicious segment to the left.
    let prevIdx = -1;
    for (let j = i - 1; j >= 0; j--) {
      if (!suspicious[j]) { prevIdx = j; break; }
    }
    // Nearest non-suspicious segment to the right.
    let nextIdx = -1;
    for (let j = i + 1; j < segs.length; j++) {
      if (!suspicious[j]) { nextIdx = j; break; }
    }

    // No clean context on one side → cannot confirm isolation → keep.
    if (prevIdx < 0 || nextIdx < 0) continue;

    const prev = segs[prevIdx], seg = segs[i], next = segs[nextIdx];
    const gapBefore = Math.abs(seg.movieStart  - prev.movieEnd);
    const gapAfter  = Math.abs(next.movieStart - seg.movieEnd);

    // Connected to at least one trusted neighbour → part of a legitimate run → keep.
    if (gapBefore <= JUMP_THRESHOLD || gapAfter <= JUMP_THRESHOLD) continue;

    // Both trusted neighbours are far.  Isolation is confirmed when they are
    // substantially closer to each other than to this segment — meaning the
    // sequence flows coherently without it.
    const neighborGap = Math.abs(next.movieStart - prev.movieEnd);
    if (neighborGap < gapBefore * 0.5 && neighborGap < gapAfter * 0.5) {
      dropMask[i] = true;
      console.log(
        `[Matcher] Sequence-drop: seg [${seg.shortStart.toFixed(2)}–` +
        `${seg.shortEnd.toFixed(2)}s] jumped to movie ${seg.movieStart.toFixed(2)}s,` +
        ` inconsistent with surrounding sequence` +
        ` (neighbors at movie ~${prev.movieEnd.toFixed(0)}s` +
        ` and ~${next.movieStart.toFixed(0)}s) — rejected.`
      );
    }
  }

  return segs.filter((_, i) => !dropMask[i]);
}

// FIX-4: drop leading/trailing frames whose similarity is below SIM_CUTOFF.
// Segments that shrink below MIN_FRAMES_AFTER are removed entirely.
function trimLowSimFrames(segs: MatchedSegment[]): MatchedSegment[] {
  const SIM_CUTOFF       = 75; // FIX-4: hard cutoff for individual frame similarity
  const MIN_FRAMES_AFTER = 3;  // FIX-4: drop segment when too few frames survive trim

  const out: MatchedSegment[] = [];
  for (const seg of segs) {
    const seq = seg.matchSequence;
    if (seq.length === 0) continue;

    let lo = 0;
    while (lo < seq.length && seq[lo].similarity < SIM_CUTOFF) lo++;
    let hi = seq.length - 1;
    while (hi > lo && seq[hi].similarity < SIM_CUTOFF) hi--;

    const trimmed = seq.slice(lo, hi + 1);
    if (trimmed.length < MIN_FRAMES_AFTER) {
      console.log(
        `[Matcher] TrimLowSim: seg [${seg.shortStart.toFixed(2)}–${seg.shortEnd.toFixed(2)}s]` +
        ` dropped (${trimmed.length} frame(s) remain after trim from ${seq.length}).`
      );
      continue;
    }

    const avgConf = boostScore(trimmed.reduce((s, f) => s + f.similarity, 0) / trimmed.length);
    out.push({
      ...seg,
      matchSequence: trimmed,
      shortStart:    trimmed[0].shortTime,
      shortEnd:      trimmed[trimmed.length - 1].shortTime,
      movieStart:    trimmed[0].movieTime,
      movieEnd:      trimmed[trimmed.length - 1].movieTime,
      frameCount:    trimmed.length,
      confidence:    avgConf,
    });
  }
  return out;
}

// FIX-3: reject segments where the short-clip advances significantly but the
// matched movie-time barely moves (walk locked onto a static/smoke frame cluster).
function frameStagnationFilter(segs: MatchedSegment[]): MatchedSegment[] {
  const SHORT_ADVANCE_GATE = 1.5;  // s — short-clip window that triggers the check
  const MOVIE_SPREAD_GATE  = 0.08; // s — movie-time must spread more than this

  return segs.filter(seg => {
    const seq = seg.matchSequence;
    if (seq.length < 2) return true;

    for (let i = 0; i < seq.length - 1; i++) {
      let minMT = seq[i].movieTime, maxMT = seq[i].movieTime;
      for (let j = i + 1; j < seq.length; j++) {
        const shortAdvance = seq[j].shortTime - seq[i].shortTime;
        if (seq[j].movieTime < minMT) minMT = seq[j].movieTime;
        if (seq[j].movieTime > maxMT) maxMT = seq[j].movieTime;
        if (shortAdvance >= SHORT_ADVANCE_GATE) {
          if (maxMT - minMT < MOVIE_SPREAD_GATE) {
            console.log(
              `[Matcher] Stagnation-drop: seg [${seg.shortStart.toFixed(2)}–` +
              `${seg.shortEnd.toFixed(2)}s] movie-time spread=${(maxMT - minMT).toFixed(3)}s` +
              ` while short advanced ${shortAdvance.toFixed(2)}s — rejected.`
            );
            return false;
          }
          break;
        }
      }
    }
    return true;
  });
}

// FIX-2: check whether a detected scene-cut boundary falls between two
// adjacent segments in the short clip.  Prevents merging across hard cuts.
function hasSceneCutBetween(
  cur: MatchedSegment,
  nxt: MatchedSegment,
  isCut: Uint8Array,
  shortFps: FPData[]
): boolean {
  let curEndSi = -1, nxtStartSi = -1;
  let curEndDist = Infinity, nxtStartDist = Infinity;
  for (let si = 0; si < shortFps.length; si++) {
    const t = shortFps[si].timestamp;
    const d1 = Math.abs(t - cur.shortEnd);
    if (d1 < curEndDist)   { curEndDist   = d1; curEndSi   = si; }
    const d2 = Math.abs(t - nxt.shortStart);
    if (d2 < nxtStartDist) { nxtStartDist = d2; nxtStartSi = si; }
  }
  if (curEndSi < 0 || nxtStartSi <= curEndSi) return false;
  for (let si = curEndSi + 1; si <= nxtStartSi; si++) {
    if (isCut[si]) return true;
  }
  return false;
}

/**
 * Merge consecutive segments where:
 *  - The gap in the short clip is small (< SHORT_GAP_MAX seconds)
 *  - The movie timeline is progressing forward and proportionally
 *
 * This repairs over-segmentation caused by false scene cuts: two segments
 * that should be one continuous match get re-joined here.
 *
 * Example: seg A ends at clip 9.76s/movie 15.44s, seg B starts at clip
 * 9.80s/movie 14.04s.  Short gap = 0.04 s (1 frame).  Movie gap = -1.4 s
 * (backward — likely a false cut inside a static/slow scene).  These should
 * NOT be merged (movie goes backward too far).
 *
 * Example: seg A ends clip 1.33s/movie 2.83s, seg B starts clip 1.38s/movie
 * 3.42s.  Short gap = 0.05 s, movie gap = 0.59 s.  Merge → single segment.
 */
function mergeAdjacentSegments(
  segs: MatchedSegment[],
  isCut?: Uint8Array,    // FIX-2: scene-cut boundaries from detectSceneCuts
  shortFps?: FPData[]   // FIX-2: short-clip frame timestamps for cut lookup
): MatchedSegment[] {
  if (segs.length <= 1) return segs;

  // Work in short-clip time order
  const sorted = [...segs].sort((a, b) => a.shortStart - b.shortStart);

  // Max allowed gap (seconds) in the short clip between two segments to merge
  const SHORT_GAP_MAX = 0.52; // ~13 frames @ 25 fps

  const result: MatchedSegment[] = [];
  let cur = sorted[0];

  for (let i = 1; i < sorted.length; i++) {
    const nxt = sorted[i];

    const shortGap = nxt.shortStart - cur.shortEnd;   // gap in clip  (s)
    const movieGap = nxt.movieStart - cur.movieEnd;   // gap in movie (s)

    // Allow merge when:
    //  1. Short gap is small (same scene, brief false-cut boundary)
    //  2. Movie time is moving forward (or very slightly backward — 1 frame jitter)
    //  3. Movie gap is proportional to short gap (same speed ratio, ±4 s tolerance)
    //  4. FIX-2: no detected scene-cut boundary falls between the two segments
    const mergeable =
      shortGap >= -0.04 &&
      shortGap <= SHORT_GAP_MAX &&
      movieGap >= -0.08 &&                             // not jumping backward in movie
      movieGap <= shortGap * 5 + 2.5 &&               // movie gap roughly proportional
      !(isCut && shortFps && hasSceneCutBetween(cur, nxt, isCut, shortFps)); // FIX-2

    if (mergeable) {
      const totalFrames = cur.frameCount + nxt.frameCount;
      cur = {
        ...cur,
        shortEnd:   nxt.shortEnd,
        movieEnd:   nxt.movieEnd,
        frameCount: totalFrames,
        confidence: (cur.confidence * cur.frameCount + nxt.confidence * nxt.frameCount) / totalFrames,
        isApproximate: cur.isApproximate || nxt.isApproximate,
        gapCount:   cur.gapCount + nxt.gapCount + Math.round(shortGap * 25),
        // Weighted average speed ratio from both halves
        speedRatio: (cur.speedRatio * cur.frameCount + nxt.speedRatio * nxt.frameCount) / totalFrames,
        matchSequence:  [...cur.matchSequence, ...nxt.matchSequence],
        bestFrameDetail: (cur.bestFrameDetail && nxt.bestFrameDetail)
          ? (cur.confidence >= nxt.confidence ? cur.bestFrameDetail : nxt.bestFrameDetail)
          : (cur.bestFrameDetail ?? nxt.bestFrameDetail),
      };
    } else {
      result.push(cur);
      cur = nxt;
    }
  }
  result.push(cur);
  return result;
}

// ---------------------------------------------------------------------------
// Main engine — v5 scene-chunk-first
// ---------------------------------------------------------------------------

/**
 * Find ALL matched segments of shortFps inside movieFps.
 *
 * v5 strategy — three passes, per scene chunk:
 *
 *  1. Pre-split the short clip at detected scene cuts → N chunks.
 *     Each chunk is one continuous scene from the edited compilation.
 *
 *  2. Pass 1 — for each chunk, seed-search + bounded bidirectional walk
 *     (confidence ≥ minSimilarity).  Walk cannot cross chunk boundaries.
 *
 *  3. Pass 2 — same for still-unmatched chunks, lower threshold (≥ 40 %).
 *
 *  4. Pass 3 — forced best-match: any chunk still unmatched gets assigned
 *     the best-scoring movie region regardless of threshold, so every scene
 *     in the short clip is guaranteed to produce at least one segment.
 *
 * @param frameDrift  Extra frames to add to the base walk search window.
 */
export async function groundMatchedSegments(
  shortFps: FPData[],
  movieFps: FPData[],
  minSimilarity = 82,
  minConsecutiveFrames = 10,
  frameDrift = 3,
  _prebuiltShort?: PreSet,
  _prebuiltMovie?: PreSet,
  onProgress?: (info: MatchProgressInfo) => void,
  externalCuts?: Uint8Array,
  hnswIndex?: MovieVectorIndex | null,
): Promise<MatchResult> {
  if (shortFps.length === 0 || movieFps.length === 0) {
    return { segments: [], unmatchedRanges: [] };
  }

  if (_prebuiltShort && _prebuiltMovie) {
    console.log(`[Matcher] Using pre-built hash arrays: ${shortFps.length} short + ${movieFps.length} movie frames (frameDrift=${frameDrift})`);
  } else {
    console.log(`[Matcher] Precomputing hash arrays: ${shortFps.length} short + ${movieFps.length} movie frames… (frameDrift=${frameDrift})`);
  }
  const sSet = _prebuiltShort ?? precompute(shortFps);
  const mSet = _prebuiltMovie ?? precompute(movieFps);

  const dEnabled    = sSet.dFlat !== null && mSet.dFlat !== null && sSet.dBits === mSet.dBits;
  const flipEnabled = mSet.faFlat !== null && sSet.aBits === mSet.aBits;
  const tEnabled    = sSet.tDelta !== null && mSet.tDelta !== null;
  console.log(`[Matcher] Feature channels: dHash=${dEnabled ? 'on' : 'off'} flipDetect=${flipEnabled ? 'on' : 'off'} temporalMotion=${tEnabled ? 'on' : 'off'}`);

  // Scene cut detection — multi-signal (threshold-based)
  const isCut            = detectSceneCuts(sSet);
  const numThresholdCuts = isCut.reduce((n, v) => n + v, 0);

  // OR in TransNetV2 shot boundaries (strictly additive — never removes existing cuts)
  if (externalCuts) {
    for (let i = 1; i < Math.min(isCut.length, externalCuts.length); i++) {
      if (externalCuts[i] && !isCut[i]) {
        isCut[i] = 1;
        console.log(
          `[ShotBoundary] TransNetV2 detected additional cut at frame ${i}` +
          ` (t=${shortFps[i]?.timestamp.toFixed(2)}s, missed by threshold-based detectSceneCuts).`
        );
      }
    }
  }
  const numCuts          = isCut.reduce((n, v) => n + v, 0);
  const numTransNetCuts  = numCuts - numThresholdCuts;

  // Split short clip into scene chunks
  const chunks = splitBySceneCuts(shortFps, isCut);
  console.log(
    `[Matcher] Scene cuts: ${numThresholdCuts} threshold-based` +
    (numTransNetCuts > 0 ? ` + ${numTransNetCuts} TransNetV2` : '') +
    ` = ${numCuts} total → ${chunks.length} chunk(s).`
  );

  console.log('[Matcher] Precompute done. Starting scene-chunk scan…');

  const usedShort = new Uint8Array(shortFps.length);
  const segments: MatchedSegment[] = [];

  // ------------------------------------------------------------------
  // Passes 1 & 2: per-chunk seeded matching
  // ------------------------------------------------------------------
  for (let pass = 1; pass <= 2; pass++) {
    const passMinSim = pass === 1 ? minSimilarity : 40;
    const isApprox   = pass === 2;
    let   passCount  = 0;

    for (let ci = 0; ci < chunks.length; ci++) {
      const chunk     = chunks[ci];
      const chunkSize = chunk.end - chunk.start + 1;

      // Report progress per chunk in pass 1 (pass 2 is a lighter retry — not double-reported)
      if (pass === 1) {
        onProgress?.({
          phase: 'matching',
          pct: 25 + Math.round((ci / chunks.length) * 55),
          chunkIdx: ci,
          totalChunks: chunks.length,
          shortStart: shortFps[chunk.start]?.timestamp,
          shortEnd:   shortFps[chunk.end]?.timestamp,
          segmentsFound: segments.length,
        });
      }

      // Skip if already fully matched
      let hasUnmatched = false;
      for (let si = chunk.start; si <= chunk.end; si++) {
        if (!usedShort[si]) { hasUnmatched = true; break; }
      }
      if (!hasUnmatched) continue;

      // Minimum frames needed to accept a segment (scales with chunk size)
      const chunkMinFrames = Math.min(minConsecutiveFrames, Math.max(3, Math.floor(chunkSize * 0.4)));

      // Try seeding from 5 strategic positions within the chunk:
      // 0%, 25%, 50%, 75%, 100%
      const seedPositions = new Set<number>();
      for (let p = 0; p <= 4; p++) {
        seedPositions.add(chunk.start + Math.round(p * (chunkSize - 1) / 4));
      }

      let bestSeq: RawSeq[] | null = null;
      let bestSeqConf = 0;

      for (const scanSi of seedPositions) {
        // Use closest unmatched frame if scanSi is already used
        let si = scanSi;
        if (usedShort[si]) {
          let found = false;
          for (let d = 1; d <= chunkSize; d++) {
            if (si + d <= chunk.end && !usedShort[si + d]) { si = si + d; found = true; break; }
            if (si - d >= chunk.start && !usedShort[si - d]) { si = si - d; found = true; break; }
          }
          if (!found) continue;
        }

        const yp = yieldIfNeeded(ci * 5);
        if (yp) await yp;

        const fastFloor = passMinSim - 20;
        const cands: Array<{ mi: number; sim: number }> = [];
        let lastCand: { mi: number; sim: number } | null = null;

        for (let mi = 0; mi < movieFps.length; mi++) {
          const s = hashSimFastCross(sSet, si, mSet, mi);
          if (s < fastFloor) continue;
          if (lastCand && mi - lastCand.mi < SEED_SEPARATION) {
            if (s > lastCand.sim) { lastCand.mi = mi; lastCand.sim = s; }
          } else {
            lastCand = { mi, sim: s };
            cands.push(lastCand);
          }
        }

        // ── HNSW vector-search augmentation ────────────────────────────────
        // After the hash scan, query the HNSW index to surface candidate movie
        // regions the sequential walk might miss (e.g. heavy speed variation
        // placing the true match far outside the hash-drift window).
        // Gated on ENABLE_CLIP_MATCHING — hnswIndex is null when CLIP is off.
        if (hnswIndex && sSet.embFlat && sSet.embDim > 0) {
          const eOff = si * sSet.embDim;
          if (eOff + sSet.embDim <= sSet.embFlat.length) {
            const shortEmb = sSet.embFlat.subarray(eOff, eOff + sSet.embDim);
            const t0Hnsw   = Date.now();
            const hnswHits = findNearestMovieFrames(hnswIndex, shortEmb, 20);
            const queryMs  = Date.now() - t0Hnsw;
            let newCount   = 0;
            for (const hc of hnswHits) {
              const sim100 = hnswDistToSim100(hc.distance);
              if (sim100 < fastFloor) continue;
              const alreadyCovered = cands.some(
                c => Math.abs(c.mi - hc.movieFrameIndex) < SEED_SEPARATION
              );
              if (!alreadyCovered) {
                cands.push({ mi: hc.movieFrameIndex, sim: sim100 });
                newCount++;
                console.log(
                  `[VectorIndex] HNSW found alternate candidate movie region at` +
                  ` ${movieFps[hc.movieFrameIndex]?.timestamp?.toFixed(2)}s` +
                  ` for short-clip chunk [${shortFps[chunk.start]?.timestamp?.toFixed(2)}-${shortFps[chunk.end]?.timestamp?.toFixed(2)}s]` +
                  ` (not reached by sequential walk) — passed to standard matching pipeline for verification.`
                );
              }
            }
            if (newCount > 0) {
              console.log(
                `[VectorIndex] HNSW query (si=${si}): ${newCount} new candidate(s)` +
                ` added in ${queryMs} ms (${hnswHits.length} hits evaluated).`
              );
            }
          }
        }

        if (cands.length === 0) continue;
        cands.sort((a, b) => b.sim - a.sim);
        const topCands = cands.slice(0, MAX_SEED_CANDIDATES);
        if (topCands[0].sim < passMinSim - 18) continue;

        for (const cand of topCands) {
          const seedSim = frameSim(sSet, si, mSet, cand.mi);
          if (seedSim < passMinSim) continue;

          // Low-detail seed gate: skip seeds rooted in near-blank / fade frames
          // using the same logic as the walk gate above.
          {
            const sSeedLevel = frameInfoLevel(sSet.fps[si]);
            const mSeedLevel = frameInfoLevel(mSet.fps[cand.mi]);
            const seedGateActive =
              mSeedLevel === 'hard-low' ||
              (mSeedLevel === 'soft-low' && sSeedLevel !== 'normal') ||
              (sSeedLevel === 'soft-low' && mSeedLevel !== 'normal');
            if (seedGateActive && seedSim < passMinSim + LOW_DETAIL_SIM_BOOST) {
              console.log(
                `[Matcher] Low-detail seed skip: movie ${mSet.fps[cand.mi].timestamp.toFixed(2)}s` +
                ` (level=${mSeedLevel}) short ${sSet.fps[si].timestamp.toFixed(2)}s` +
                ` (level=${sSeedLevel}) — sim=${seedSim.toFixed(1)}% < ${passMinSim + LOW_DETAIL_SIM_BOOST}%`
              );
              continue;
            }
          }

          const seq = buildSegment(
            sSet, mSet, si, cand.mi, seedSim,
            usedShort, isCut, frameDrift,
            chunk.start, chunk.end
          );
          if (seq.length < chunkMinFrames) continue;

          const conf = seq.reduce((a, f) => a + f.sim, 0) / seq.length;
          if (
            bestSeq === null ||
            seq.length > bestSeq.length ||
            (seq.length === bestSeq.length && conf > bestSeqConf)
          ) {
            bestSeq = seq;
            bestSeqConf = conf;
          }
        }
      }

      if (!bestSeq) continue;

      for (const item of bestSeq) usedShort[item.si] = 1;
      segments.push(acceptSegment(bestSeq, shortFps, movieFps, isApprox, sSet, mSet));
      passCount++;
    }

    console.log(`[Matcher] Pass ${pass} (minSim=${passMinSim}%): ${passCount} chunk(s) matched.`);
  }

  onProgress?.({ phase: 'finalizing', pct: 92 });

  // ------------------------------------------------------------------
  // Pass 3: forced best-match — only for chunks large enough to be real scenes
  // ------------------------------------------------------------------
  // Small chunks (< MIN_FORCED_FRAMES) are almost always caused by false scene
  // cuts (e.g. a single-frame brightness spike, motion blur, etc.).  Forcing a
  // segment for them produces spurious 1–4 frame segments with random movie
  // times.  Skip them; they'll become tiny "unmatched" ranges (< 0.2 s) which
  // are invisible to the user.
  // Minimum 10 frames = reference algorithm's stated threshold for a valid segment.
  // Smaller leftovers are almost always false-cut fragments or CGI/altered content.
  const MIN_FORCED_FRAMES = 10;
  let pass3Count = 0;

  for (const chunk of chunks) {
    const remaining: number[] = [];
    for (let si = chunk.start; si <= chunk.end; si++) {
      if (!usedShort[si]) remaining.push(si);
    }
    if (remaining.length === 0) continue;

    if (remaining.length < MIN_FORCED_FRAMES) {
      console.log(`[Matcher] Pass 3 (skip): chunk [${chunk.start}–${chunk.end}], only ${remaining.length} frame(s) — too small to force-match.`);
      continue;
    }

    console.log(`[Matcher] Pass 3 (forced): chunk [${chunk.start}–${chunk.end}], ${remaining.length} unmatched frame(s)…`);

    // For each remaining frame, find the globally best-matching movie frame
    const bestOf: Array<{ si: number; mi: number; sim: number }> = [];
    for (let k = 0; k < remaining.length; k++) {
      const si = remaining[k];

      const yp = yieldIfNeeded(k, 200);
      if (yp) await yp;

      let bestMi = 0, bestSim = 0;
      for (let mi = 0; mi < movieFps.length; mi++) {
        const s = frameSim(sSet, si, mSet, mi);
        if (s > bestSim) { bestSim = s; bestMi = mi; }
      }
      bestOf.push({ si, mi: bestMi, sim: bestSim });
    }

    // Confidence gate: if even the best forced match is very weak, the content
    // likely doesn't exist in the reference movie (e.g. CGI insert, green-screen
    // overlay, third-party clip).  Leave it as an unmatched range rather than
    // fabricating a low-quality segment.
    const UNMATCHED_SIM_GATE = 65; // below this avg % → altered / unmatched
    const avgForcedSim = bestOf.reduce((s, f) => s + f.sim, 0) / bestOf.length;
    if (avgForcedSim < UNMATCHED_SIM_GATE) {
      console.log(
        `[Matcher] Pass 3 (unmatched): chunk [${chunk.start}–${chunk.end}]` +
        ` avg sim ${avgForcedSim.toFixed(1)}% < ${UNMATCHED_SIM_GATE}%` +
        ` — flagged as altered/unmatched content.`
      );
      continue; // frames stay free → reported in unmatchedRanges
    }

    // Group temporally-contiguous frames into sub-segments
    let k = 0;
    while (k < bestOf.length) {
      const group: typeof bestOf = [bestOf[k]];
      let curMi = bestOf[k].mi;

      for (let j = k + 1; j < bestOf.length; j++) {
        const item     = bestOf[j];
        const siGap    = item.si - bestOf[j - 1].si;
        const expected = curMi + siGap;
        if (Math.abs(item.mi - expected) <= LOOK_AHEAD * 2) {
          group.push(item);
          curMi = item.mi;
        } else {
          break;
        }
      }

      for (const item of group) usedShort[item.si] = 1;
      segments.push(acceptSegment(group, shortFps, movieFps, true, sSet, mSet));
      pass3Count++;
      k += Math.max(1, group.length);
    }
  }

  if (pass3Count > 0) {
    console.log(`[Matcher] Pass 3: ${pass3Count} forced segment(s).`);
  }

  // ------------------------------------------------------------------
  // FIX-4: Hard-trim low-similarity leading/trailing frames before merging
  // ------------------------------------------------------------------
  const trimmedSegments = trimLowSimFrames(segments);
  if (trimmedSegments.length !== segments.length) {
    console.log(`[Matcher] TrimLowSim: ${segments.length - trimmedSegments.length} segment(s) dropped.`);
  }

  // ------------------------------------------------------------------
  // Merge adjacent segments that belong to the same continuous run.
  // This repairs over-segmentation from false scene cuts: two segments
  // that should be one get re-joined if their short-clip gap is small
  // and the movie timeline progresses forward proportionally.
  // FIX-2: pass isCut + shortFps so merges never cross a hard scene cut.
  // ------------------------------------------------------------------
  const preDedup = mergeAdjacentSegments(trimmedSegments, isCut, shortFps);
  console.log(`[Matcher] After merge: ${preDedup.length} segment(s) (was ${trimmedSegments.length}).`);

  // ------------------------------------------------------------------
  // Deduplication — keep highest-confidence segment when short-clip
  // ranges overlap by more than 0.15 s
  // ------------------------------------------------------------------
  preDedup.sort((a, b) => b.confidence - a.confidence);

  const final: MatchedSegment[] = [];
  for (const seg of preDedup) {
    const overlaps = final.some(kept => {
      const oStart = Math.max(kept.shortStart, seg.shortStart);
      const oEnd   = Math.min(kept.shortEnd,   seg.shortEnd);
      return oEnd - oStart > 0.15;
    });
    if (!overlaps) final.push(seg);
  }

  final.sort((a, b) => a.shortStart - b.shortStart);

  // Context-aware validation: drop low-confidence segments that have no
  // high-confidence neighbour confirming a consistent movie timeline.
  const contextValidated = contextValidateSegments(final);
  if (contextValidated.length !== final.length) {
    console.log(`[Matcher] Context validation: dropped ${final.length - contextValidated.length} segment(s).`);
  }

  // DTW refinement: attempt re-alignment for borderline-confidence segments and
  // segments with implausible speedRatio before the speed-ratio filter drops them.
  // Only applied to the already-narrowed candidate regions — never the full movie.
  const dtwRefined = refineWithDTW(
    contextValidated,
    (si, mi) => frameSim(sSet, si, mSet, mi),
    shortFps,
    movieFps,
  );
  if (dtwRefined.length !== contextValidated.length) {
    console.log(`[Matcher] DTW refinement: changed ${contextValidated.length - dtwRefined.length} segment(s).`);
  }

  // SpeedRatio validation: drop segments with implausible temporal duration ratios.
  const speedRatioValidated = speedRatioFilterSegments(dtwRefined);
  if (speedRatioValidated.length !== contextValidated.length) {
    console.log(`[Matcher] SpeedRatio validation: dropped ${contextValidated.length - speedRatioValidated.length} segment(s).`);
  }

  // FIX-3: reject segments with frame-stagnation (movie time stuck while short advances).
  const stagnationFiltered = frameStagnationFilter(speedRatioValidated);
  if (stagnationFiltered.length !== speedRatioValidated.length) {
    console.log(`[Matcher] Stagnation filter: dropped ${speedRatioValidated.length - stagnationFiltered.length} segment(s).`);
  }

  // Sequence-consistency validation: reject isolated position outliers.
  const validated = sequenceConsistencyFilter(stagnationFiltered);
  if (validated.length !== speedRatioValidated.length) {
    console.log(`[Matcher] Sequence validation: dropped ${speedRatioValidated.length - validated.length} segment(s).`);
  }

  const tToSi = new Map<string, number>();
  shortFps.forEach((fp, si) => tToSi.set(fp.timestamp.toFixed(4), si));

  const usedFinal = new Uint8Array(shortFps.length);
  for (const seg of validated) {
    for (const frame of seg.matchSequence) {
      const si = tToSi.get(frame.shortTime.toFixed(4));
      if (si !== undefined) usedFinal[si] = 1;
    }
  }

  const unmatchedRanges = computeUnmatched(shortFps, usedFinal);

  console.log(`[Matcher] Final: ${validated.length} segment(s), ${unmatchedRanges.length} unmatched range(s).`);
  return { segments: validated, unmatchedRanges };
}

// ---------------------------------------------------------------------------
// Memory-efficient streaming precompute — reads NDJSON without loading all
// hash strings into memory at once.
// ---------------------------------------------------------------------------

/**
 * Count lines in a file by streaming through it.
 * Used to pre-size the flat TypedArrays before the main streaming pass.
 */
async function countFileLines(filePath: string): Promise<number> {
  return new Promise((resolve, reject) => {
    let count = 0;
    const stream = fs.createReadStream(filePath, { encoding: 'utf8', highWaterMark: 64 * 1024 });
    stream.on('data', (chunk: string) => {
      for (let i = 0; i < chunk.length; i++) {
        if (chunk.charCodeAt(i) === 10 /* '\n' */) count++;
      }
    });
    stream.on('end', () => resolve(count));
    stream.on('error', reject);
  });
}

/**
 * Build a PreSet by streaming an NDJSON fingerprint file line-by-line.
 *
 * Hash strings are converted to flat Uint32Arrays immediately and then
 * discarded — they are NEVER accumulated in a large JS array.  Only the
 * compact per-frame data (frameIndex, timestamp, signature) is kept in the
 * fps array, cutting peak RAM from ~6-8 GB to ~400 MB for a 2-hour movie.
 */
async function streamPrecomputeFromNDJSON(filePath: string): Promise<PreSet> {
  const totalFrames = await countFileLines(filePath);
  if (totalFrames === 0) return precompute([]);

  // These are allocated at full size up-front (TypedArrays → outside JS heap)
  let aFlat:  Uint32Array | null = null;
  let faFlat: Uint32Array | null = null;
  let dFlat:  Uint32Array | null = null;
  let fdFlat: Uint32Array | null = null;
  let pFlat:  Uint32Array | null = null;
  const tDeltaBuf = new Float32Array(totalFrames * 48);
  const tMagBuf   = new Float32Array(totalFrames);

  let variantNames: string[] = [];
  let numVariants = 0;
  let aBits = 256, aWords = 8, dBits = 0, dWords = 0, pBits = 0, pWords = 0;
  let hasFlip = false, hasD = false, hasP = false;
  const variantIdx = new Map<string, number>();

  // Compact fps — only what the matching logic actually uses after precompute
  const compactFps: FPData[] = [];
  let allHaveSig = true;
  let prevColorGrid: number[] | null = null;
  let fi = 0;

  await new Promise<void>((resolve, reject) => {
    const rl = readline.createInterface({
      input: fs.createReadStream(filePath, { encoding: 'utf8' }),
      crlfDelay: Infinity,
    });

    rl.on('line', (line: string) => {
      if (!line.trim()) return;

      let frame: any;
      try { frame = JSON.parse(line); } catch { return; }

      // ── First-frame initialisation ────────────────────────────────────
      if (fi === 0) {
        variantNames = Object.keys(frame.variants || {});
        numVariants  = variantNames.length;
        const fv     = frame.variants?.[variantNames[0]];
        aBits   = fv?.hash?.length  || 256;
        aWords  = Math.max(1, Math.ceil(aBits / 32));
        hasD    = typeof fv?.dhash  === 'string' && fv.dhash.length  > 0;
        hasFlip = typeof fv?.fhash  === 'string' && fv.fhash.length  > 0;
        dBits   = hasD ? fv.dhash.length : 0;
        dWords  = hasD ? Math.max(1, Math.ceil(dBits / 32)) : 0;
        hasP    = typeof fv?.phash  === 'string' && fv.phash.length  > 0;
        pBits   = hasP ? fv.phash.length : 0;
        pWords  = hasP ? Math.max(1, Math.ceil(pBits / 32)) : 0;
        variantNames.forEach((n, i) => variantIdx.set(n, i));

        aFlat  = new Uint32Array(totalFrames * numVariants * aWords);
        faFlat = hasFlip ? new Uint32Array(totalFrames * numVariants * aWords) : null;
        dFlat  = hasD    ? new Uint32Array(totalFrames * numVariants * dWords) : null;
        fdFlat = (hasD && hasFlip) ? new Uint32Array(totalFrames * numVariants * dWords) : null;
        pFlat  = hasP    ? new Uint32Array(totalFrames * numVariants * pWords) : null;
      }

      // ── Fill flat hash arrays ─────────────────────────────────────────
      for (let vi = 0; vi < numVariants; vi++) {
        const v    = frame.variants?.[variantNames[vi]];
        const aOff = (fi * numVariants + vi) * aWords;
        aFlat!.set(hashToU32(v?.hash  ?? '', aWords), aOff);
        if (faFlat) faFlat.set(hashToU32(v?.fhash ?? '', aWords), aOff);
        if (dFlat) {
          const dOff = (fi * numVariants + vi) * dWords;
          dFlat.set(hashToU32(v?.dhash  ?? '', dWords), dOff);
          if (fdFlat) fdFlat.set(hashToU32(v?.fdhash ?? '', dWords), dOff);
        }
        if (pFlat) {
          const pOff = (fi * numVariants + vi) * pWords;
          pFlat.set(hashToU32(v?.phash ?? '', pWords), pOff);
        }
      }

      // ── Temporal colour-delta ─────────────────────────────────────────
      const sig = frame.signature as FrameSignature | undefined;
      if (sig?.colorGrid?.length === 48 && prevColorGrid && fi > 0) {
        let mag = 0;
        for (let k = 0; k < 48; k++) {
          const d = sig.colorGrid[k] - prevColorGrid[k];
          tDeltaBuf[fi * 48 + k] = d;
          mag += d * d;
        }
        tMagBuf[fi] = Math.sqrt(mag);
      }
      prevColorGrid = sig?.colorGrid ?? null;
      if (!sig || sig.colorGrid?.length !== 48) allHaveSig = false;

      // ── Compact fps entry (no variant hash strings) ───────────────────
      compactFps.push({
        frameIndex: frame.frameIndex,
        timestamp:  frame.timestamp,
        variants:   {},   // hash data lives in flat arrays — strings freed
        signature:  sig,
      } as FPData);

      fi++;
      // The `frame` object goes out of scope here and is eligible for GC.
    });

    rl.on('close', resolve);
    rl.on('error', reject);
  });

  return {
    fps: compactFps,
    variantNames,
    numVariants,
    aFlat:  aFlat  ?? new Uint32Array(0),
    faFlat: hasFlip ? faFlat : null,
    dFlat:  hasD   ? dFlat  : null,
    fdFlat: (hasD && hasFlip) ? fdFlat : null,
    pFlat:  hasP   ? pFlat  : null,
    aBits, aWords, dBits, dWords, pBits, pWords,
    variantIdx,
    tDelta: (allHaveSig && fi > 1) ? tDeltaBuf : null,
    tMag:   (allHaveSig && fi > 1) ? tMagBuf   : null,
    embFlat: null, embDim: 0,
  };
}

/**
 * Build a PreSet from a fingerprint result file.
 *
 * Supports two formats:
 *  - **NDJSON** (new, default): one JSON object per line — streamed line-by-line
 *    so hash strings are never accumulated in memory.
 *  - **JSON array** (legacy): `[{...},{...},...]` — parsed all at once for
 *    backward compatibility with result files created before this change.
 */
export async function streamPrecomputeFromFile(filePath: string): Promise<PreSet> {
  // Peek at the first byte to detect format.
  const fd = fs.openSync(filePath, 'r');
  const peek = Buffer.alloc(1);
  fs.readSync(fd, peek, 0, 1, 0);
  fs.closeSync(fd);
  const firstChar = peek.toString('utf8');

  if (firstChar === '[') {
    // Legacy JSON array — load with JSON.parse (backward compat).
    console.log('[Precompute] Legacy JSON array format — loading into memory');
    const fps: FPData[] = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    return precompute(fps);
  }

  // NDJSON — memory-efficient streaming path.
  return streamPrecomputeFromNDJSON(filePath);
}

// ---------------------------------------------------------------------------
// Chunked movie matching — loads movie data in fixed-size windows so that
// RAM consumption is bounded regardless of movie length.
//
// Activated automatically by matchVideosFromFiles when the estimated movie
// PreSet would exceed 4 GB or when free system RAM is tight.
// ---------------------------------------------------------------------------

/** Variant-level metadata extracted from the first line of an NDJSON file. */
interface VariantMeta {
  variantNames: string[];
  numVariants:  number;
  aBits:  number; aWords: number;
  dBits:  number; dWords: number;
  pBits:  number; pWords: number;
  hasFlip: boolean;
  hasD:    boolean;
  hasP:    boolean;
}

/**
 * Single pass through the movie NDJSON to build a byte-offset index for each
 * line (frame).  Returns the index and variant metadata from the first frame.
 * Time: O(file size); RAM: O(frameCount × 8 bytes) — ~1.4 MB per 180 K frames.
 */
async function buildMovieLineIndex(filePath: string): Promise<{
  byteOffsets: Float64Array;   // byteOffsets[i] = byte start of NDJSON line i
  totalFrames: number;
  meta: VariantMeta;
}> {
  const offsets: number[] = [];
  let meta: VariantMeta | null = null;
  let bytePos  = 0;
  let lineStart = 0;
  let partial   = '';                // carries incomplete line across chunks

  await new Promise<void>((resolve, reject) => {
    const stream = fs.createReadStream(filePath, { highWaterMark: 256 * 1024 });
    stream.on('data', (raw: Buffer | string) => {
      const chunk: Buffer = Buffer.isBuffer(raw) ? raw : Buffer.from(raw as string, 'binary');
      let chunkOff = 0;
      for (let i = 0; i < chunk.length; i++) {
        if (chunk[i] === 10 /* '\n' */) {
          const lineText = partial + chunk.subarray(chunkOff, i).toString('utf8');
          partial   = '';
          chunkOff  = i + 1;
          if (lineText.trim()) {
            offsets.push(lineStart);
            if (!meta) {
              try {
                const frame = JSON.parse(lineText);
                const vn    = Object.keys(frame.variants || {});
                const fv    = frame.variants?.[vn[0]];
                const aBits = fv?.hash?.length  || 256;
                const dBits = typeof fv?.dhash === 'string' ? fv.dhash.length : 0;
                const pBits = typeof fv?.phash === 'string' ? fv.phash.length : 0;
                meta = {
                  variantNames: vn, numVariants: vn.length,
                  aBits, aWords: Math.max(1, Math.ceil(aBits / 32)),
                  dBits, dWords: dBits > 0 ? Math.max(1, Math.ceil(dBits / 32)) : 0,
                  pBits, pWords: pBits > 0 ? Math.max(1, Math.ceil(pBits / 32)) : 0,
                  hasFlip: typeof fv?.fhash === 'string' && fv.fhash.length > 0,
                  hasD:    dBits > 0,
                  hasP:    pBits > 0,
                };
              } catch { /* keep meta null, retry next line */ }
            }
          }
          lineStart = bytePos + i + 1;
        }
      }
      partial  += chunk.subarray(chunkOff).toString('utf8');
      bytePos  += chunk.length;
    });
    stream.on('end', () => {
      if (partial.trim()) offsets.push(lineStart);
      resolve();
    });
    stream.on('error', reject);
  });

  const defaultMeta: VariantMeta = {
    variantNames: [], numVariants: 0, aBits: 256, aWords: 8,
    dBits: 0, dWords: 0, pBits: 0, pWords: 0, hasFlip: false, hasD: false, hasP: false,
  };
  return {
    byteOffsets: new Float64Array(offsets),
    totalFrames: offsets.length,
    meta:        meta ?? defaultMeta,
  };
}

/**
 * Load a contiguous range of movie frames [globalStart, globalEnd] from an
 * NDJSON file using pre-built byte offsets, and return a PreSet whose frame
 * indices run 0 .. (globalEnd − globalStart).
 *
 * The returned PreSet has correct timestamps from the file; the caller must
 * add `globalStart` to any `mi` values before using them with a global fps array.
 */
async function loadMovieWindowPreset(
  filePath:     string,
  byteOffsets:  Float64Array,
  globalStart:  number,
  globalEnd:    number,
  meta:         VariantMeta,
): Promise<PreSet> {
  const count = globalEnd - globalStart + 1;
  if (count <= 0) return precompute([]);

  const startByte = byteOffsets[globalStart];
  const endByte   = (globalEnd + 1 < byteOffsets.length)
    ? byteOffsets[globalEnd + 1]
    : fs.statSync(filePath).size;

  const rawBuf = Buffer.alloc(endByte - startByte);
  const fd = fs.openSync(filePath, 'r');
  fs.readSync(fd, rawBuf, 0, rawBuf.length, startByte);
  fs.closeSync(fd);

  const lines = rawBuf.toString('utf8').split('\n').filter(l => l.trim());
  const { numVariants, variantNames, aBits, aWords, dBits, dWords, pBits, pWords, hasFlip, hasD, hasP } = meta;
  const variantIdx = new Map<string, number>(variantNames.map((n, i) => [n, i]));

  const aFlat  = new Uint32Array(count * numVariants * aWords);
  const faFlat = hasFlip ? new Uint32Array(count * numVariants * aWords) : null;
  const dFlat  = hasD   ? new Uint32Array(count * numVariants * dWords) : null;
  const fdFlat = (hasD && hasFlip) ? new Uint32Array(count * numVariants * dWords) : null;
  const pFlat  = hasP   ? new Uint32Array(count * numVariants * pWords) : null;
  const tDeltaBuf = new Float32Array(count * 48);
  const tMagBuf   = new Float32Array(count);

  const compactFps: FPData[] = [];
  let prevColorGrid: number[] | null = null;
  let allHaveSig = true;

  for (let li = 0; li < Math.min(lines.length, count); li++) {
    let frame: any;
    try { frame = JSON.parse(lines[li]); } catch { continue; }

    for (let vi = 0; vi < numVariants; vi++) {
      const v    = frame.variants?.[variantNames[vi]];
      const aOff = (li * numVariants + vi) * aWords;
      aFlat.set(hashToU32(v?.hash  ?? '', aWords), aOff);
      if (faFlat) faFlat.set(hashToU32(v?.fhash ?? '', aWords), aOff);
      if (dFlat) {
        const dOff = (li * numVariants + vi) * dWords;
        dFlat.set(hashToU32(v?.dhash  ?? '', dWords), dOff);
        if (fdFlat) fdFlat.set(hashToU32(v?.fdhash ?? '', dWords), dOff);
      }
      if (pFlat) {
        const pOff = (li * numVariants + vi) * pWords;
        pFlat.set(hashToU32(v?.phash ?? '', pWords), pOff);
      }
    }

    const sig = frame.signature as FrameSignature | undefined;
    if (sig?.colorGrid?.length === 48 && prevColorGrid && li > 0) {
      let mag = 0;
      for (let k = 0; k < 48; k++) {
        const d = sig.colorGrid[k] - prevColorGrid[k];
        tDeltaBuf[li * 48 + k] = d;
        mag += d * d;
      }
      tMagBuf[li] = Math.sqrt(mag);
    }
    prevColorGrid = sig?.colorGrid ?? null;
    if (!sig || sig.colorGrid?.length !== 48) allHaveSig = false;

    compactFps.push({
      frameIndex: frame.frameIndex,
      timestamp:  frame.timestamp,
      variants:   {},
      signature:  sig,
    } as FPData);
  }

  return {
    fps: compactFps,
    variantNames, numVariants, variantIdx,
    aFlat, faFlat,
    dFlat:  hasD              ? dFlat!  : null,
    fdFlat: (hasD && hasFlip) ? fdFlat! : null,
    pFlat:  hasP              ? pFlat!  : null,
    aBits, aWords, dBits, dWords, pBits, pWords,
    tDelta: (allHaveSig && compactFps.length > 1) ? tDeltaBuf : null,
    tMag:   (allHaveSig && compactFps.length > 1) ? tMagBuf   : null,
    embFlat: null, embDim: 0,
  };
}

/** Estimate bytes consumed by a fully-loaded movie PreSet. */
function estimateMoviePresetBytes(
  frameCount:  number,
  numVariants: number,
  aWords:      number,
  dWords:      number,
  pWords:      number,
): number {
  return (
    frameCount * numVariants * (aWords + dWords) * 2 * 4 + // aFlat + dFlat + mirrors
    frameCount * numVariants * pWords * 4 +                 // pFlat (no flip variant)
    frameCount * 48 * 4 +                                   // tDelta
    frameCount * 4 +                                        // tMag
    frameCount * 1_400                                      // JS objects + signatures
  );
}

/**
 * Chunked matching orchestrator.
 *
 * Strategy:
 *  1. Build a byte-offset index for the movie NDJSON (single fast pass).
 *  2. Scan the movie in CHUNK_FRAMES-frame windows; for each window build a
 *     small PreSet, run hashSimFastCross against the short scene seeds, and
 *     accumulate global candidate lists.  The window is immediately freed.
 *  3. For each scene chunk, verify the top candidates and run a bounded
 *     bidirectional walk using a ±WALK_WINDOW frame window around the seed.
 *  4. Post-process (merge, dedup, context-validate) exactly as the full path.
 *
 * RAM peak ≈ max(CHUNK_FRAMES, 2×WALK_WINDOW) × bytes_per_frame, not
 * totalFrames × bytes_per_frame.
 */
async function groundMatchedSegmentsChunked(
  shortSet:       PreSet,
  movieFilePath:  string,
  byteOffsets:    Float64Array,
  movieMetaFps:   FPData[],           // lightweight global fps (timestamps only, no hashes)
  meta:           VariantMeta,
  minSimilarity:  number,
  minConsFrames:  number,
  frameDrift:     number,
  onProgress?:    (info: MatchProgressInfo) => void,
  externalCuts?:  Uint8Array,
  hnswIndex?:     MovieVectorIndex | null,
): Promise<MatchResult> {
  const CHUNK_FRAMES = 10_000;  // movie frames per scan window (~4 MB aFlat for 13v/8w)
  const WALK_WINDOW  = 3_000;   // half-width around each seed for the bidirectional walk

  const shortFps        = shortSet.fps;
  const totalMovieFrames = movieMetaFps.length;
  const fastFloor        = minSimilarity - 20;

  // ── 1. Scene cut detection on short clip ─────────────────────────────────
  const isCut            = detectSceneCuts(shortSet);
  const numThresholdCuts = isCut.reduce((n, v) => n + v, 0);

  // OR in TransNetV2 shot boundaries (strictly additive)
  if (externalCuts) {
    for (let i = 1; i < Math.min(isCut.length, externalCuts.length); i++) {
      if (externalCuts[i] && !isCut[i]) {
        isCut[i] = 1;
        console.log(
          `[ShotBoundary] TransNetV2 detected additional cut at frame ${i}` +
          ` (t=${shortFps[i]?.timestamp.toFixed(2)}s, missed by threshold-based detectSceneCuts).`
        );
      }
    }
  }
  const numCuts         = isCut.reduce((n, v) => n + v, 0);
  const numTransNetCuts = numCuts - numThresholdCuts;

  const chunks = splitBySceneCuts(shortFps, isCut);
  console.log(
    `[MatchChunked] Scene cuts: ${numThresholdCuts} threshold-based` +
    (numTransNetCuts > 0 ? ` + ${numTransNetCuts} TransNetV2` : '') +
    ` = ${numCuts} total → ${chunks.length} chunk(s).` +
    ` Scanning ${totalMovieFrames} movie frames in chunks of ${CHUNK_FRAMES}…`
  );

  // Collect seed candidate lists: allCands[si] = [{mi_global, sim}, ...]
  const allCands = new Map<number, Array<{ mi: number; sim: number }>>();

  // Seed positions to probe (5 strategic points per scene chunk)
  const seedSiSet = new Set<number>();
  for (const sc of chunks) {
    const sz = sc.end - sc.start + 1;
    for (let p = 0; p <= 4; p++) {
      seedSiSet.add(sc.start + Math.round(p * (sz - 1) / 4));
    }
  }
  for (const si of seedSiSet) allCands.set(si, []);

  // ── 2. Chunked scan ───────────────────────────────────────────────────────
  for (let chunkStart = 0; chunkStart < totalMovieFrames; chunkStart += CHUNK_FRAMES) {
    const chunkEnd = Math.min(chunkStart + CHUNK_FRAMES - 1, totalMovieFrames - 1);
    const chunkSet = await loadMovieWindowPreset(movieFilePath, byteOffsets, chunkStart, chunkEnd, meta);
    const chunkLen = chunkEnd - chunkStart + 1;

    for (const si of seedSiSet) {
      const list = allCands.get(si)!;
      let lastCand: { mi: number; sim: number } | null = null;

      for (let localMi = 0; localMi < chunkLen; localMi++) {
        const s = hashSimFastCross(shortSet, si, chunkSet, localMi);
        if (s < fastFloor) continue;
        const globalMi = chunkStart + localMi;
        if (lastCand && globalMi - lastCand.mi < SEED_SEPARATION) {
          if (s > lastCand.sim) lastCand.sim = s;
        } else {
          lastCand = { mi: globalMi, sim: s };
          list.push(lastCand);
        }
      }

      // Prune to top MAX_SEED_CANDIDATES periodically to cap list growth
      if (list.length > MAX_SEED_CANDIDATES * 4) {
        list.sort((a, b) => b.sim - a.sim);
        list.splice(MAX_SEED_CANDIDATES * 2);
      }
    }

    console.log(`[MatchChunked] Scanned frames ${chunkStart}–${chunkEnd}`);
    onProgress?.({
      phase: 'scanning',
      pct: 20 + Math.round(((chunkEnd + 1) / totalMovieFrames) * 52),
    });
  }

  // Final sort + trim
  for (const [, list] of allCands) {
    list.sort((a, b) => b.sim - a.sim);
    if (list.length > MAX_SEED_CANDIDATES) list.splice(MAX_SEED_CANDIDATES);
  }

  // ── HNSW vector-search augmentation ──────────────────────────────────────
  // After the full sequential hash scan, query the HNSW index for each seed
  // position to surface candidate movie regions the chunk-by-chunk scan might
  // have missed entirely (e.g. due to large speed variation beyond the drift
  // window).  New HNSW candidates are merged into allCands and flow through
  // the same Passes 1/2/3 pipeline — identical quality bar, no special treatment.
  // Gated on ENABLE_CLIP_MATCHING — hnswIndex is null when CLIP is off.
  if (hnswIndex && shortSet.embFlat && shortSet.embDim > 0) {
    const tHnsw0   = Date.now();
    let   totalNew = 0;

    for (const si of seedSiSet) {
      const eOff = si * shortSet.embDim;
      if (eOff + shortSet.embDim > shortSet.embFlat.length) continue;

      const shortEmb = shortSet.embFlat.subarray(eOff, eOff + shortSet.embDim);
      const hnswHits = findNearestMovieFrames(hnswIndex, shortEmb, 20);
      const existing = allCands.get(si) ?? [];

      for (const hc of hnswHits) {
        const sim100 = hnswDistToSim100(hc.distance);
        if (sim100 < fastFloor) continue;
        const alreadyCovered = existing.some(
          c => Math.abs(c.mi - hc.movieFrameIndex) < SEED_SEPARATION
        );
        if (!alreadyCovered) {
          existing.push({ mi: hc.movieFrameIndex, sim: sim100 });
          totalNew++;
          // Find the scene chunk this seed belongs to for context in the log message
          const sc = chunks.find(c => si >= c.start && si <= c.end);
          console.log(
            `[VectorIndex] HNSW found alternate candidate movie region at` +
            ` ${movieMetaFps[hc.movieFrameIndex]?.timestamp?.toFixed(2)}s` +
            ` for short-clip chunk [${shortFps[sc?.start ?? si]?.timestamp?.toFixed(2)}-${shortFps[sc?.end ?? si]?.timestamp?.toFixed(2)}s]` +
            ` (not reached by sequential walk) — passed to standard matching pipeline for verification.`
          );
        }
      }

      existing.sort((a, b) => b.sim - a.sim);
      if (existing.length > MAX_SEED_CANDIDATES) existing.splice(MAX_SEED_CANDIDATES);
      allCands.set(si, existing);
    }

    console.log(
      `[VectorIndex] HNSW augmentation complete: ${totalNew} new candidate(s)` +
      ` across ${seedSiSet.size} seed position(s) — ${Date.now() - tHnsw0} ms.`
    );
  }

  // ── 3. Passes 1 & 2: walk from seeds ─────────────────────────────────────
  const usedShort = new Uint8Array(shortFps.length);
  const segments: MatchedSegment[] = [];

  for (let pass = 1; pass <= 2; pass++) {
    const passMinSim = pass === 1 ? minSimilarity : 40;
    const isApprox   = pass === 2;
    let   passCount  = 0;

    for (const sc of chunks) {
      const scSize = sc.end - sc.start + 1;
      let hasUnmatched = false;
      for (let si = sc.start; si <= sc.end; si++) {
        if (!usedShort[si]) { hasUnmatched = true; break; }
      }
      if (!hasUnmatched) continue;

      const chunkMinFrames = Math.min(minConsFrames, Math.max(3, Math.floor(scSize * 0.4)));
      const seedPositions  = new Set<number>();
      for (let p = 0; p <= 4; p++) {
        seedPositions.add(sc.start + Math.round(p * (scSize - 1) / 4));
      }

      let bestSeq: RawSeq[] | null = null;
      let bestSeqConf = 0;
      let bestWinStart = 0;

      for (const scanSi of seedPositions) {
        let si = scanSi;
        if (usedShort[si]) {
          let found = false;
          for (let d = 1; d <= scSize; d++) {
            if (si + d <= sc.end && !usedShort[si + d]) { si = si + d; found = true; break; }
            if (si - d >= sc.start && !usedShort[si - d]) { si = si - d; found = true; break; }
          }
          if (!found) continue;
        }

        const cands = (allCands.get(si) ?? []).filter(c => c.sim >= passMinSim - 18);
        if (cands.length === 0) continue;

        for (const cand of cands.slice(0, MAX_SEED_CANDIDATES)) {
          const winStart = Math.max(0, cand.mi - WALK_WINDOW);
          const winEnd   = Math.min(totalMovieFrames - 1, cand.mi + WALK_WINDOW);
          const localMi  = cand.mi - winStart;

          const winSet  = await loadMovieWindowPreset(movieFilePath, byteOffsets, winStart, winEnd, meta);
          const seedSim = frameSim(shortSet, si, winSet, localMi);
          if (seedSim < passMinSim) continue;

          const seq = buildSegment(shortSet, winSet, si, localMi, seedSim,
            usedShort, isCut, frameDrift, sc.start, sc.end);
          if (seq.length < chunkMinFrames) continue;

          const conf = seq.reduce((a, f) => a + f.sim, 0) / seq.length;
          if (bestSeq === null || seq.length > bestSeq.length ||
              (seq.length === bestSeq.length && conf > bestSeqConf)) {
            bestSeq      = seq;
            bestSeqConf  = conf;
            bestWinStart = winStart;
          }
        }
      }

      if (!bestSeq) continue;

      // Convert window-local mi → global mi; use movieMetaFps (global timestamps)
      const globalSeq = bestSeq.map(f => ({ ...f, mi: f.mi + bestWinStart }));
      for (const item of globalSeq) usedShort[item.si] = 1;
      segments.push(acceptSegment(globalSeq, shortFps, movieMetaFps, isApprox));
      passCount++;
    }

    console.log(`[MatchChunked] Pass ${pass} (minSim=${passMinSim}%): ${passCount} chunk(s) matched.`);
  }

  onProgress?.({ phase: 'finalizing', pct: 92 });

  // ── 4. Pass 3: forced best-match ─────────────────────────────────────────
  const MIN_FORCED_FRAMES = 10;
  for (const sc of chunks) {
    const remaining: number[] = [];
    for (let si = sc.start; si <= sc.end; si++) {
      if (!usedShort[si]) remaining.push(si);
    }
    if (remaining.length === 0 || remaining.length < MIN_FORCED_FRAMES) continue;

    console.log(`[MatchChunked] Pass 3 (forced): chunk [${sc.start}–${sc.end}], ${remaining.length} frames`);

    const bestOf: Array<{ si: number; mi: number; sim: number }> = [];
    for (const si of remaining) {
      const cands = allCands.get(si) ?? [];
      if (cands.length > 0) bestOf.push({ si, mi: cands[0].mi, sim: cands[0].sim });
    }

    const avgSim = bestOf.length > 0 ? bestOf.reduce((s, f) => s + f.sim, 0) / bestOf.length : 0;
    if (avgSim < 65) continue;

    let k = 0;
    while (k < bestOf.length) {
      const group = [bestOf[k]];
      let curMi   = bestOf[k].mi;
      for (let j = k + 1; j < bestOf.length; j++) {
        const item  = bestOf[j];
        const siGap = item.si - bestOf[j - 1].si;
        if (Math.abs(item.mi - (curMi + siGap)) <= LOOK_AHEAD * 2) {
          group.push(item); curMi = item.mi;
        } else break;
      }
      for (const item of group) usedShort[item.si] = 1;
      segments.push(acceptSegment(group, shortFps, movieMetaFps, true));
      k += Math.max(1, group.length);
    }
  }

  // ── 5. Post-process (same as full path) ───────────────────────────────────
  // FIX-4: hard-trim low-similarity leading/trailing frames before merging
  const trimmedSegments2 = trimLowSimFrames(segments);
  if (trimmedSegments2.length !== segments.length) {
    console.log(`[MatchChunked] TrimLowSim: ${segments.length - trimmedSegments2.length} segment(s) dropped.`);
  }
  // FIX-2: pass isCut + shortFps so merges never cross a hard scene cut.
  const merged = mergeAdjacentSegments(trimmedSegments2, isCut, shortFps);
  console.log(`[MatchChunked] After merge: ${merged.length} segment(s) (was ${trimmedSegments2.length}).`);

  merged.sort((a, b) => b.confidence - a.confidence);
  const deduped: MatchedSegment[] = [];
  for (const seg of merged) {
    const overlaps = deduped.some(k => {
      const oStart = Math.max(k.shortStart, seg.shortStart);
      const oEnd   = Math.min(k.shortEnd,   seg.shortEnd);
      return oEnd - oStart > 0.15;
    });
    if (!overlaps) deduped.push(seg);
  }
  deduped.sort((a, b) => a.shortStart - b.shortStart);

  const contextValidated2 = contextValidateSegments(deduped);
  if (contextValidated2.length !== deduped.length) {
    console.log(`[MatchChunked] Context validation: dropped ${deduped.length - contextValidated2.length} segment(s).`);
  }

  // SpeedRatio validation: drop segments with implausible temporal duration ratios.
  const speedRatioValidated2 = speedRatioFilterSegments(contextValidated2);
  if (speedRatioValidated2.length !== contextValidated2.length) {
    console.log(`[MatchChunked] SpeedRatio validation: dropped ${contextValidated2.length - speedRatioValidated2.length} segment(s).`);
  }

  // FIX-3: reject segments with frame-stagnation (movie time stuck while short advances).
  const stagnationFiltered2 = frameStagnationFilter(speedRatioValidated2);
  if (stagnationFiltered2.length !== speedRatioValidated2.length) {
    console.log(`[MatchChunked] Stagnation filter: dropped ${speedRatioValidated2.length - stagnationFiltered2.length} segment(s).`);
  }

  // Sequence-consistency validation: reject isolated position outliers.
  const validated = sequenceConsistencyFilter(stagnationFiltered2);
  if (validated.length !== speedRatioValidated2.length) {
    console.log(`[MatchChunked] Sequence validation: dropped ${speedRatioValidated2.length - validated.length} segment(s).`);
  }

  const tToSi = new Map<string, number>();
  shortFps.forEach((fp, si) => tToSi.set(fp.timestamp.toFixed(4), si));
  const usedFinal = new Uint8Array(shortFps.length);
  for (const seg of validated) {
    for (const frame of seg.matchSequence) {
      const si = tToSi.get(frame.shortTime.toFixed(4));
      if (si !== undefined) usedFinal[si] = 1;
    }
  }

  console.log(`[MatchChunked] Final: ${validated.length} segment(s).`);
  return { segments: validated, unmatchedRanges: computeUnmatched(shortFps, usedFinal) };
}

// ---------------------------------------------------------------------------
// CLIP embedding attachment — loads .embeddings.bin next to a result file
// and attaches the flat array to an existing PreSet.
// ---------------------------------------------------------------------------

/**
 * Load the .embeddings.bin sidecar file for a result path and attach to the
 * given PreSet.  No-op (and no error) when the file does not exist.
 *
 * Backward compatible: old jobs without the sidecar just get embFlat=null
 * and fall back to hash-only matching automatically.
 */
function attachEmbeddings(preset: PreSet, resultPath: string): void {
  const binPath = resultPath + '.embeddings.bin';
  const flat = loadEmbeddingsFile(binPath);
  if (!flat || flat.length === 0) return;

  const frameCount = preset.fps.length;
  if (frameCount === 0) return;

  const dim = Math.floor(flat.length / frameCount);
  if (dim <= 0 || dim * frameCount !== flat.length) {
    console.warn(
      `[CLIP] Embeddings file size mismatch: ${flat.length} floats / ${frameCount} frames` +
      ` = ${flat.length / frameCount} (non-integer) — skipping CLIP for this job`
    );
    return;
  }

  preset.embFlat = flat;
  preset.embDim  = dim;
  console.log(`[CLIP] Loaded embeddings: ${binPath} — ${frameCount} frames, dim=${dim}`);
}

/**
 * Memory-efficient public API: builds both PreSets by streaming their
 * fingerprint files then runs the full matching pipeline.
 *
 * Automatically switches to the chunked path when the estimated movie PreSet
 * RAM exceeds 4 GB or when available system RAM is tight (< 2 GB headroom).
 *
 * Peak RAM for a 2-hour movie on the full path:  ~350 MB
 * Peak RAM for a 10-hour movie on the full path: ~1.7 GB
 * Chunked path cap: max(CHUNK_FRAMES, 2×WALK_WINDOW) frames in RAM at once
 */
export async function matchVideosFromFiles(
  shortResultPath: string,
  movieResultPath: string,
  opts: {
    minSimilarity?:       number;
    minConsecutiveFrames?: number;
    frameDrift?:          number;
    onProgress?:          (info: MatchProgressInfo) => void;
  } = {}
): Promise<MatchResult & { movieFrames: number; shortFrames: number }> {
  const {
    minSimilarity        = 82,
    minConsecutiveFrames = 9,
    frameDrift           = 3,
    onProgress,
  } = opts;

  onProgress?.({ phase: 'loading_short', pct: 3 });
  console.log('[Match] Streaming precompute: short fingerprints…');
  const shortPreSet = await streamPrecomputeFromFile(shortResultPath);
  const shortFrames = shortPreSet.fps.length;

  // ── Attach CLIP embeddings for short clip (both paths benefit) ────────────
  attachEmbeddings(shortPreSet, shortResultPath);

  // ── TransNetV2 shot boundary detection (additive signal, short clip only) ─
  // Runs once regardless of which matching path is used.  The result is OR'd
  // with the threshold-based detectSceneCuts inside groundMatchedSegments /
  // groundMatchedSegmentsChunked — it never removes an existing cut.
  let transNetV2Cuts: Uint8Array | undefined;
  if (isShotBoundaryEnabled()) {
    const shortVideoPath = await getVideoPathFromResultPath(shortResultPath);
    if (shortVideoPath) {
      const cuts = await detectShotBoundaries(shortVideoPath, shortFrames);
      if (cuts) transNetV2Cuts = cuts;
    } else {
      console.warn('[ShotBoundary] Could not resolve video path for short clip — TransNetV2 skipped.');
    }
  }

  // ── Decide: full load vs chunked ──────────────────────────────────────────
  // Peek at movie frame count via line index (fast: one sequential pass).
  // This also gives us byte offsets ready for the chunked path if needed.
  onProgress?.({ phase: 'indexing', pct: 10 });
  console.log('[Match] Building movie line index…');
  const { byteOffsets, totalFrames: movieFrames, meta } = await buildMovieLineIndex(movieResultPath);

  const estimatedBytes = estimateMoviePresetBytes(
    movieFrames, meta.numVariants || 13, meta.aWords || 8, meta.dWords || 0, meta.pWords || 0
  );
  const RAM_MATCH_LIMIT = 4 * 1024 * 1024 * 1024; // 4 GB
  const freeRam         = os.freemem();
  const useChunked      = estimatedBytes > RAM_MATCH_LIMIT || estimatedBytes > freeRam - 500_000_000;

  console.log(
    `[Match] Movie: ${movieFrames} frames — estimated PreSet ${(estimatedBytes / 1e9).toFixed(2)} GB` +
    ` — free RAM ${(freeRam / 1e9).toFixed(2)} GB — path: ${useChunked ? 'CHUNKED' : 'full-load'}`
  );

  if (useChunked) {
    // ── Chunked path ─────────────────────────────────────────────────────────
    // CLIP embeddings for the short clip are on shortPreSet.embFlat (attached above).
    // Movie chunk PreSets are loaded window-by-window and cannot carry embeddings.
    // embeddingSim() returns -1 when mSet.embFlat is null, so frameSim() degrades
    // gracefully to hash+signature for movie frames in chunked mode.
    onProgress?.({ phase: 'loading_movie', pct: 18 });
    console.log('[Match] Loading movie metadata (timestamps)…');
    const movieMetaFps: FPData[] = [];
    const rl = readline.createInterface({
      input: fs.createReadStream(movieResultPath, { encoding: 'utf8' }),
      crlfDelay: Infinity,
    });
    for await (const line of rl) {
      if (!line.trim()) continue;
      try {
        const f = JSON.parse(line);
        movieMetaFps.push({ frameIndex: f.frameIndex, timestamp: f.timestamp, variants: {}, signature: undefined });
      } catch { /* skip */ }
    }

    // ── Build HNSW vector index (chunked path) ────────────────────────────────
    // The full movie PreSet is not loaded on the chunked path, but CLIP embeddings
    // are in a separate sidecar (.embeddings.bin) — load that directly to build
    // the index.  One-time build; cached to *_result.json.hnsw.bin.  Skipped when
    // the sidecar is absent or CLIP is off.
    let movieHnswIndex: MovieVectorIndex | null = null;
    if (shortPreSet.embDim > 0) {
      const movieEmbFlat = loadEmbeddingsFile(movieResultPath + '.embeddings.bin');
      if (movieEmbFlat && movieEmbFlat.length > 0) {
        const movieEmbDim = Math.floor(movieEmbFlat.length / movieFrames);
        if (movieEmbDim === shortPreSet.embDim && movieEmbDim > 0) {
          movieHnswIndex = await loadOrBuildHnswIndex(
            movieEmbFlat, movieEmbDim, movieResultPath + '.hnsw.bin'
          );
        } else {
          console.warn(
            `[VectorIndex] Chunked path: movie embedding dim ${movieEmbDim}` +
            ` ≠ short ${shortPreSet.embDim} — HNSW skipped.`
          );
        }
      }
    }

    onProgress?.({ phase: 'scanning', pct: 20 });
    let result = await groundMatchedSegmentsChunked(
      shortPreSet, movieResultPath, byteOffsets, movieMetaFps, meta,
      minSimilarity, minConsecutiveFrames, frameDrift, onProgress, transNetV2Cuts,
      movieHnswIndex,
    );
    result = await enhanceBorderlineSegments(result, shortResultPath, movieResultPath);
    return { ...result, movieFrames, shortFrames };
  }

  // ── Full-load path ────────────────────────────────────────────────────────
  onProgress?.({ phase: 'loading_movie', pct: 18 });
  console.log('[Match] Streaming precompute: movie fingerprints…');
  const moviePreSet = await streamPrecomputeFromFile(movieResultPath);

  // Attach CLIP embeddings for movie (full-load path only)
  attachEmbeddings(moviePreSet, movieResultPath);

  // ── Build HNSW vector index (full-load path) ──────────────────────────────
  // One-time build per movie upload; cached to *_result.json.hnsw.bin.
  // Subsequent matches load the cache in < 200 ms rather than rebuilding.
  // Query cost per short-clip chunk: < 1 ms at ef=64.  Skipped when CLIP off.
  let movieHnswIndex: MovieVectorIndex | null = null;
  if (shortPreSet.embDim > 0 && moviePreSet.embDim > 0 &&
      shortPreSet.embDim === moviePreSet.embDim) {
    const hnswCachePath = movieResultPath + '.hnsw.bin';
    movieHnswIndex = await loadOrBuildHnswIndex(
      moviePreSet.embFlat, moviePreSet.embDim, hnswCachePath
    );
  }

  const clipStatus = (shortPreSet.embDim > 0 && moviePreSet.embDim > 0)
    ? `CLIP on (dim=${shortPreSet.embDim})`
    : (shortPreSet.embDim > 0 ? 'CLIP short-only (movie embeddings missing)' : 'CLIP off');
  onProgress?.({ phase: 'matching', pct: 25 });
  console.log(
    `[Match] Loaded ${movieFrames} movie frames, ${shortFrames} short frames.` +
    ` ${clipStatus}. Running matching…`
  );

  let result = await groundMatchedSegments(
    shortPreSet.fps,
    moviePreSet.fps,
    minSimilarity,
    minConsecutiveFrames,
    frameDrift,
    shortPreSet,
    moviePreSet,
    onProgress,
    transNetV2Cuts,
    movieHnswIndex,
  );

  result = await enhanceBorderlineSegments(result, shortResultPath, movieResultPath);

  onProgress?.({ phase: 'finalizing', pct: 97 });
  return { ...result, movieFrames, shortFrames };
}


// ---------------------------------------------------------------------------
// Face/Hand Landmark enhancement for borderline matches
// ---------------------------------------------------------------------------
import { exec } from 'child_process';
import { promisify } from 'util';
import * as path from 'path';
import { getObjectSignal, compareObjectSignals, initObjectSignal } from './object-signal';
import { getFaceSignal, compareFaceSignals, initFaceSignal } from './face-signal';
import { getKeypointSignal, compareKeypointSignals, initKeypointSignal } from './keypoint-signal';
import { getSubjectMask, compareSubjectSignals, initSubjectSignal } from './subject-signal';
import { createCanvas, loadImage } from 'canvas';

const execAsync = promisify(exec);

async function getVideoPathFromResultPath(resultPath: string): Promise<string | null> {
  const metaPath = resultPath.replace('_result.json', '_meta.json');
  if (!fs.existsSync(metaPath)) return null;
  try {
    const meta = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
    const jobId = path.basename(resultPath).replace('_result.json', '');
    const dir = path.dirname(resultPath);
    const videoPath = path.join(dir, `${jobId}-${meta.originalName}`);
    if (fs.existsSync(videoPath)) return videoPath;
    
    // Check legacy temp format just in case
    const files = fs.readdirSync(dir);
    for (const f of files) {
      if (f.startsWith(jobId) && f.endsWith('.mp4')) return path.join(dir, f);
    }
  } catch(e) {
    console.error('Error finding video for result path', resultPath, e);
  }
  return null;
}

async function extractFrame(videoPath: string, timestampSec: number): Promise<{ data: Uint8ClampedArray, width: number, height: number } | null> {
  try {
    const tempImage = `/tmp/frame_${Date.now()}_${Math.random().toString(36).substring(7)}.jpg`;
    await execAsync(`ffmpeg -y -ss ${timestampSec} -i "${videoPath}" -vframes 1 -q:v 2 "${tempImage}"`);
    
    const image = await loadImage(tempImage);
    const canvas = createCanvas(image.width, image.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0);
    const imageData = ctx.getImageData(0, 0, image.width, image.height);
    
    fs.unlinkSync(tempImage);
    return imageData;
  } catch (e) {
    console.error('Failed to extract frame at', timestampSec, 'from', videoPath);
    return null;
  }
}

async function enhanceBorderlineSegments(result: MatchResult, shortResultPath: string, movieResultPath: string): Promise<MatchResult> {
  const BORDERLINE_MIN = 55;
  const BORDERLINE_MAX = 75;
  const FACE_SIM_THRESHOLD = 80;
  const OBJECT_SIM_THRESHOLD = 60;
  const KEYPOINT_SIM_THRESHOLD = 30;  // keypointSim (0–100) required to confirm a borderline match
  
  const borderlineSegments = result.segments.filter(s => s.confidence >= BORDERLINE_MIN && s.confidence <= BORDERLINE_MAX);
  if (borderlineSegments.length === 0) return result;

  console.log(`[Enhancer] Found ${borderlineSegments.length} borderline segments. Preparing to enhance...`);
  
  await initFaceSignal();
  await initObjectSignal();
  initKeypointSignal();
  await initSubjectSignal();
  
  const shortVideoPath = await getVideoPathFromResultPath(shortResultPath);
  const movieVideoPath = await getVideoPathFromResultPath(movieResultPath);
  
  if (!shortVideoPath || !movieVideoPath) {
    console.warn('[Enhancer] Could not find original video files for face/object signal enhancement.');
    return result;
  }
  
  for (const seg of borderlineSegments) {
    // Pick the middle frame of the segment for comparison
    const midIdx = Math.floor(seg.matchSequence.length / 2);
    const frameMatch = seg.matchSequence[midIdx];
    
    console.log(`[Enhancer] Checking borderline segment (conf: ${seg.confidence.toFixed(1)}%) at short ${frameMatch.shortTime}s, movie ${frameMatch.movieTime}s`);
    
    const shortImageData = await extractFrame(shortVideoPath, frameMatch.shortTime);
    const movieImageData = await extractFrame(movieVideoPath, frameMatch.movieTime);
    
    if (!shortImageData || !movieImageData) continue;
    
    const shortSignal = getFaceSignal(shortImageData);
    const movieSignal = getFaceSignal(movieImageData);
    
    let boosted = false;

    if (shortSignal.hasFace && movieSignal.hasFace) {
      const faceSim = compareFaceSignals(shortSignal, movieSignal);
      console.log(`[Enhancer] Face/Hand similarity: ${faceSim.toFixed(1)}%`);
      
      if (faceSim > FACE_SIM_THRESHOLD) {
        // Boost confidence to the minimum acceptance threshold (e.g. 82) + a small bonus
        const boost = Math.max(82 - seg.confidence + 1, 0);
        seg.confidence += boost;
        boosted = true;
        console.log(`[Enhancer] Boosted segment confidence to ${seg.confidence.toFixed(1)}% via face similarity`);
      }
    } else {
      console.log(`[Enhancer] No face/hand detected in one or both frames.`);
    }

    if (!boosted) {
      // Try Object detection as a fallback/additional check
      const shortObjSignal = getObjectSignal(shortImageData);
      const movieObjSignal = getObjectSignal(movieImageData);

      if (shortObjSignal.hasObjects && movieObjSignal.hasObjects) {
        const objSimRes = compareObjectSignals(shortObjSignal, movieObjSignal, shortImageData.width, shortImageData.height);
        console.log(`[Enhancer] Object similarity: ${objSimRes.score.toFixed(1)}%, shared categories: ${objSimRes.sharedCategories.join(', ')}`);

        if (objSimRes.score > OBJECT_SIM_THRESHOLD) {
          const boost = Math.max(82 - seg.confidence + 1, 0);
          seg.confidence += boost;
          boosted = true;
          console.log(`[ObjectSignal] Borderline match at frame ${frameMatch.shortTime}s confirmed by object similarity (objectSim=${objSimRes.score.toFixed(1)}%, shared categories: ${objSimRes.sharedCategories.join(', ')}). Boosted confidence to ${seg.confidence.toFixed(1)}%.`);
        }
      } else {
        console.log(`[Enhancer] No objects detected in one or both frames.`);
      }
    }

    if (!boosted) {
      // ── ORB-lite keypoint signal ─────────────────────────────────────────
      // Additive confirming signal only — same constraint as face/object:
      // never overrides a clear non-match, never downgrades a clear match,
      // only resolves genuinely ambiguous borderline cases.
      //
      // Particularly useful when heavy cropping, perspective shift, or partial
      // occlusion causes hash/CLIP similarity to drop while distinctive
      // geometric points remain geometrically consistent between the two frames.
      //
      // Docker size increase: 0 (pure TypeScript, no new npm deps).
      // Added second-pass time: ~2–8 ms per segment pair.
      const kpT0 = Date.now();
      const shortKp = getKeypointSignal(shortImageData);
      const movieKp = getKeypointSignal(movieImageData);
      const kpMatch = compareKeypointSignals(shortKp, movieKp);
      const kpMs    = Date.now() - kpT0;

      console.log(
        `[KeypointSignal] Borderline match at frame ${frameMatch.shortTime.toFixed(2)}s` +
        ` — ORB keypoints: short=${shortKp.count} movie=${movieKp.count}` +
        ` good=${kpMatch.goodMatches} keypointSim=${kpMatch.keypointSim.toFixed(1)}%` +
        ` (${kpMs} ms)`
      );

      if (kpMatch.keypointSim > KEYPOINT_SIM_THRESHOLD) {
        const boost = Math.max(82 - seg.confidence + 1, 0);
        seg.confidence += boost;
        console.log(
          `[KeypointSignal] Borderline match at frame ${frameMatch.shortTime.toFixed(2)}s` +
          ` confirmed by ORB keypoint matching` +
          ` (${kpMatch.goodMatches} strong matches out of ${Math.min(shortKp.count, movieKp.count)} detected,` +
          ` keypointSim=${kpMatch.keypointSim.toFixed(1)}%).` +
          ` Boosted confidence to ${seg.confidence.toFixed(1)}%.`
        );
      }
    }

    // ── Foreground subject signal ──────────────────────────────────────────────
    // Runs AFTER all additive signals (face/object/keypoint), but only when the
    // segment is still within the 55–75% borderline range (i.e. no prior boost).
    // Unlike those signals this one can REDUCE confidence: if the background
    // matches well but the main foreground subjects clearly differ, the match is
    // likely a false positive driven by shared scenery.
    //
    // Uses MediaPipe ImageSegmenter (selfie_segmenter.tflite, ~230 KB, CPU).
    // Docker size impact : +~230 KB model file, no new npm packages.
    // Added second-pass time: ~15–60 ms per segment pair (model + histogram).
    //
    // Constraint: only operates within 55–75%; never touches segments that have
    // already been boosted above or dropped below that range.
    if (seg.confidence >= BORDERLINE_MIN && seg.confidence <= BORDERLINE_MAX) {
      const subjectT0 = Date.now();
      const shortSubject = getSubjectMask(shortImageData);
      const movieSubject = getSubjectMask(movieImageData);
      const subjectMs = Date.now() - subjectT0;

      console.log(
        `[SubjectSignal] Borderline match at frame ${frameMatch.shortTime.toFixed(2)}s` +
        ` — coverage: short=${(shortSubject.maskCoverage * 100).toFixed(1)}%` +
        ` movie=${(movieSubject.maskCoverage * 100).toFixed(1)}%` +
        ` hasSubject: short=${shortSubject.hasSubject} movie=${movieSubject.hasSubject}` +
        ` (${subjectMs} ms)`
      );

      if (!shortSubject.hasSubject && !movieSubject.hasSubject) {
        // Pure background/landscape — no subject to compare; skip this signal.
        console.log(
          `[SubjectSignal] No distinct subject in either frame — skipping` +
          ` (pure background/landscape scene).`
        );
      } else {
        const comparison = compareSubjectSignals(shortSubject, movieSubject);
        console.log(
          `[SubjectSignal] subjectSim=${comparison.subjectSim.toFixed(1)}%` +
          ` backgroundSim=${comparison.backgroundSim.toFixed(1)}%` +
          ` isBackgroundOnlyMatch=${comparison.isBackgroundOnlyMatch}`
        );

        if (comparison.isBackgroundOnlyMatch) {
          // Penalise: lower confidence below BORDERLINE_MIN to mark as rejected.
          const SUBJECT_PENALTY = 20;
          const prevConf = seg.confidence;
          seg.confidence = Math.max(BORDERLINE_MIN - 6, seg.confidence - SUBJECT_PENALTY);
          console.log(
            `[SubjectSignal] Borderline match at frame ${frameMatch.shortTime.toFixed(2)}s` +
            ` downgraded — background similarity high but subject regions disagree` +
            ` (subjectSim=${comparison.subjectSim.toFixed(1)}%).` +
            ` Confidence: ${prevConf.toFixed(1)}% → ${seg.confidence.toFixed(1)}%.`
          );
        }
      }
    }
  }
  
  return result;
}

