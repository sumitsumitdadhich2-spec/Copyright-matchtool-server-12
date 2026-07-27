/**
 * Foreground / subject segmentation signal for Nexus Video Match.
 *
 * Uses MediaPipe ImageSegmenter (selfie_segmenter.tflite, ~230 KB) to separate
 * the main foreground subject from the background in a frame.
 *
 * Exposes:
 *   initSubjectSignal()  — download model (once) and warm up the segmenter
 *   getSubjectMask()     — segment one frame; returns coverage + colour histograms
 *   compareSubjectSignals() — decide whether two frames share the same foreground
 *
 * Docker size impact  : +0 npm packages, +~230 KB model file
 * Per-segment runtime : ~15–60 ms for two frames (CPU delegate, variable resolution)
 *
 * Self-hosted / offline: after the one-time download the model is local; no
 * external API calls at runtime.
 */

import { ImageSegmenter, FilesetResolver } from '@mediapipe/tasks-vision';
import * as path  from 'path';
import * as fs    from 'fs';
import * as https from 'https';
import * as http  from 'http';

// MediaPipe tasks-vision reads `navigator.userAgent` during FilesetResolver
// initialisation, which doesn't exist in Node.js.  Polyfill it here so that
// initSubjectSignal() works in server-side environments.
if (typeof (globalThis as any).navigator === 'undefined') {
  (globalThis as any).navigator = { userAgent: 'Node.js' };
}

// ── Constants ─────────────────────────────────────────────────────────────────

/**
 * Lightweight MediaPipe selfie-segmenter (float16 variant).
 * Produces a 2-class confidence mask: class-0 = background, class-1 = foreground.
 * Works well for person/character subjects; gracefully returns low confidence for
 * pure landscapes, making it safe to skip when neither frame has a clear subject.
 */
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite';
const MODEL_PATH = path.resolve(process.cwd(), 'server', 'models', 'selfie_segmenter.tflite');

/** Minimum fraction of frame that must be labelled as foreground to treat it as
 *  "having a distinct subject" (avoids noise / tiny artefacts being flagged). */
const SUBJECT_COVERAGE_MIN = 0.04;   // 4 %

/** Pixel confidence threshold (foreground class) above which a pixel is subject. */
const CONFIDENCE_THRESHOLD = 0.5;

/** Histogram bins per channel (R, G, B). Total histogram length = HIST_BINS * 3 = 48. */
const HIST_BINS = 16;

// ── Runtime state ─────────────────────────────────────────────────────────────

let segmenter: ImageSegmenter | null = null;
let initPromise: Promise<void> | null = null;

// ── Public types ──────────────────────────────────────────────────────────────

export interface BoundingBox {
  x:      number;   // normalised 0–1 from left edge
  y:      number;   // normalised 0–1 from top edge
  width:  number;   // normalised 0–1
  height: number;   // normalised 0–1
}

export interface SubjectMaskResult {
  /** Whether a distinct foreground subject was detected. */
  hasSubject: boolean;
  /** Fraction of frame covered by the subject (0–1). */
  maskCoverage: number;
  /** Bounding box of the detected subject (normalised). Empty box when !hasSubject. */
  subjectRegion: BoundingBox;
  /**
   * Normalised 48-element RGB colour histogram (16 bins × 3 channels) computed
   * over subject pixels only.  Empty array when hasSubject = false.
   */
  subjectColorHist: number[];
  /**
   * Normalised 48-element RGB colour histogram of background (non-subject) pixels.
   * Always populated — when !hasSubject this covers the entire frame.
   */
  backgroundColorHist: number[];
}

// ── Model download ─────────────────────────────────────────────────────────────

function downloadModel(): Promise<void> {
  return new Promise((resolve, reject) => {
    if (fs.existsSync(MODEL_PATH)) { resolve(); return; }

    console.log('[SubjectSignal] selfie_segmenter.tflite not found — downloading (~230 KB)…');
    fs.mkdirSync(path.dirname(MODEL_PATH), { recursive: true });

    const tmp = MODEL_PATH + '.tmp';
    const file = fs.createWriteStream(tmp);

    const get = (url: string, redirects = 0): void => {
      const mod = url.startsWith('https') ? https : http;
      mod.get(url, res => {
        if ((res.statusCode === 301 || res.statusCode === 302) && res.headers.location) {
          if (redirects >= 5) { reject(new Error('Too many redirects')); return; }
          get(res.headers.location, redirects + 1);
          return;
        }
        if (res.statusCode !== 200) {
          reject(new Error(`HTTP ${res.statusCode} downloading selfie_segmenter.tflite`));
          return;
        }
        res.pipe(file);
        file.on('finish', () => {
          file.close(() => {
            fs.renameSync(tmp, MODEL_PATH);
            console.log('[SubjectSignal] Model saved to', MODEL_PATH);
            resolve();
          });
        });
      }).on('error', err => { try { fs.unlinkSync(tmp); } catch {} reject(err); });
    };

    get(MODEL_URL);
  });
}

// ── Init ──────────────────────────────────────────────────────────────────────

/**
 * Download the model (if needed) and initialise the MediaPipe ImageSegmenter.
 * Safe to call multiple times — subsequent calls are no-ops.
 */
export async function initSubjectSignal(): Promise<void> {
  if (segmenter) return;
  if (initPromise) return initPromise;

  initPromise = (async () => {
    try {
      await downloadModel();

      const wasmPath = path.resolve(
        process.cwd(),
        'node_modules/@mediapipe/tasks-vision/wasm',
      );
      const vision = await FilesetResolver.forVisionTasks(wasmPath);

      segmenter = await ImageSegmenter.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_PATH, delegate: 'CPU' },
        runningMode: 'IMAGE',
        outputCategoryMask:    false,
        outputConfidenceMasks: true,
      });

      console.log('[SubjectSignal] MediaPipe ImageSegmenter initialised (selfie model, CPU).');
    } catch (err) {
      console.error('[SubjectSignal] Initialisation failed:', (err as Error).message);
    } finally {
      initPromise = null;
    }
  })();

  return initPromise;
}

// ── Histogram helpers ─────────────────────────────────────────────────────────

/**
 * Build a normalised 48-element RGB histogram (16 bins × 3 channels) from pixels
 * where the mask is true.  Each channel is independently normalised to sum to 1.
 */
function buildHistogram(
  data: Uint8ClampedArray | Uint8Array,
  mask: boolean[],
  pixelCount: number,
): number[] {
  const hist = new Float64Array(HIST_BINS * 3);
  const binSize = 256 / HIST_BINS;
  let covered = 0;

  for (let i = 0; i < pixelCount; i++) {
    if (!mask[i]) continue;
    covered++;
    const base = i * 4;
    hist[Math.floor(data[base]     / binSize)]              += 1;
    hist[Math.floor(data[base + 1] / binSize) + HIST_BINS ] += 1;
    hist[Math.floor(data[base + 2] / binSize) + HIST_BINS * 2] += 1;
  }

  if (covered === 0) return Array(HIST_BINS * 3).fill(0);

  // Normalise each channel.
  for (let c = 0; c < 3; c++) {
    let total = 0;
    for (let b = 0; b < HIST_BINS; b++) total += hist[c * HIST_BINS + b];
    if (total > 0)
      for (let b = 0; b < HIST_BINS; b++) hist[c * HIST_BINS + b] /= total;
  }

  return Array.from(hist);
}

/** Histogram-intersection similarity: 0 (no overlap) → 1 (identical). */
function histIntersection(a: number[], b: number[]): number {
  if (a.length === 0 || b.length === 0) return 0;
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.min(a[i], b[i]);
  // Each of 3 channels sums to 1 → perfect match gives 3; normalise to 0-1.
  return s / 3;
}

// ── getSubjectMask ────────────────────────────────────────────────────────────

/**
 * Segment a frame into foreground (subject) and background.
 *
 * Returns safely (hasSubject = false) when the segmenter is not initialised,
 * so callers do not need to guard against early invocation.
 */
export function getSubjectMask(
  imageData: { data: Uint8ClampedArray | Uint8Array; width: number; height: number },
): SubjectMaskResult {
  const { data, width, height } = imageData;
  const pixelCount = width * height;

  const emptyBox: BoundingBox = { x: 0, y: 0, width: 0, height: 0 };
  const noResult: SubjectMaskResult = {
    hasSubject: false, maskCoverage: 0,
    subjectRegion: emptyBox, subjectColorHist: [], backgroundColorHist: [],
  };

  if (!segmenter) return noResult;

  // Run segmentation.
  let mpResult: ReturnType<typeof segmenter.segment>;
  try {
    mpResult = segmenter.segment(imageData as any);
  } catch (err) {
    console.warn('[SubjectSignal] segment() error:', (err as Error).message);
    return noResult;
  }

  if (!mpResult?.confidenceMasks || mpResult.confidenceMasks.length < 2) {
    mpResult?.close?.();
    return noResult;
  }

  // Extract foreground confidence map (class 1 = foreground/selfie).
  let fgConf: Float32Array;
  try {
    fgConf = mpResult.confidenceMasks[1].getAsFloat32Array();
  } catch (err) {
    console.warn('[SubjectSignal] getAsFloat32Array() error:', (err as Error).message);
    mpResult.close?.();
    return noResult;
  }
  mpResult.close?.();

  // Build binary mask and bounding box.
  const subjectMask: boolean[] = new Array(pixelCount).fill(false);
  let subjectPixels = 0;
  let minX = width, maxX = -1, minY = height, maxY = -1;

  for (let i = 0; i < pixelCount; i++) {
    if ((fgConf[i] ?? 0) <= CONFIDENCE_THRESHOLD) continue;
    subjectMask[i] = true;
    subjectPixels++;
    const x = i % width;
    const y = Math.floor(i / width);
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }

  const maskCoverage  = subjectPixels / pixelCount;
  const hasSubject    = maskCoverage >= SUBJECT_COVERAGE_MIN;
  const backgroundMask = subjectMask.map(v => !v);

  // Always build background histogram (covers full frame when !hasSubject).
  const backgroundColorHist = buildHistogram(data, backgroundMask, pixelCount);

  if (!hasSubject) {
    return { hasSubject: false, maskCoverage, subjectRegion: emptyBox,
             subjectColorHist: [], backgroundColorHist };
  }

  const subjectColorHist = buildHistogram(data, subjectMask, pixelCount);
  const subjectRegion: BoundingBox = {
    x:      minX / width,
    y:      minY / height,
    width:  (maxX - minX + 1) / width,
    height: (maxY - minY + 1) / height,
  };

  return { hasSubject, maskCoverage, subjectRegion, subjectColorHist, backgroundColorHist };
}

// ── compareSubjectSignals ─────────────────────────────────────────────────────

/**
 * Compare two segmentation results to determine whether the borderline match
 * is driven primarily by background similarity while foreground subjects differ.
 *
 * Returns:
 *   subjectSim           — 0–100; how similar the subject regions look
 *   backgroundSim        — 0–100; how similar the background regions look
 *   isBackgroundOnlyMatch — true when background is suspiciously similar but
 *                           subject content clearly disagrees (likely false positive)
 */
export function compareSubjectSignals(
  shortMask: SubjectMaskResult,
  movieMask: SubjectMaskResult,
): { subjectSim: number; backgroundSim: number; isBackgroundOnlyMatch: boolean } {

  /** Foreground-similarity below this level → subjects clearly differ. */
  const SUBJECT_SIM_THRESHOLD     = 30;  // 0-100
  /** Background similarity must exceed this to conclude BG drove the match. */
  const BACKGROUND_SIM_THRESHOLD  = 50;  // 0-100

  const bgSim = histIntersection(
    shortMask.backgroundColorHist,
    movieMask.backgroundColorHist,
  ) * 100;

  // Neither frame has a distinct subject → legitimate background/landscape shot.
  // Do not flag; defer to existing signals.
  if (!shortMask.hasSubject && !movieMask.hasSubject) {
    return { subjectSim: 100, backgroundSim: bgSim, isBackgroundOnlyMatch: false };
  }

  // Exactly one frame has a subject → clear foreground mismatch.
  if (shortMask.hasSubject !== movieMask.hasSubject) {
    return {
      subjectSim: 0,
      backgroundSim: bgSim,
      isBackgroundOnlyMatch: bgSim >= BACKGROUND_SIM_THRESHOLD,
    };
  }

  // Both frames have subjects → compare subject-region colour content.
  const subjectSim = histIntersection(
    shortMask.subjectColorHist,
    movieMask.subjectColorHist,
  ) * 100;

  const isBackgroundOnlyMatch =
    subjectSim < SUBJECT_SIM_THRESHOLD &&
    bgSim      >= BACKGROUND_SIM_THRESHOLD;

  return { subjectSim, backgroundSim: bgSim, isBackgroundOnlyMatch };
}
