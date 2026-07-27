/**
 * TransNetV2-based shot boundary detection — server/shot-boundary.ts
 *
 * Loads models/transnetv2.onnx once and runs it on the short clip before the
 * scene-chunk matching passes.  Results are OR'd with the existing threshold-
 * based detectSceneCuts output — strictly additive, never removes an existing cut.
 *
 * Model: elya5/transnetv2 on HuggingFace (MIT licence, ~31 MB)
 *   Input:  'input'  [1, 100, 27, 48, 3]  float32, values in [0, 1]
 *   Output: '534'    [1, 100, 1]            soft-cut probability per frame
 *           '535'    [1, 100, 1]            hard-cut probability per frame
 *
 * We take max(soft, hard) per frame as the combined boundary score and
 * threshold at CUT_THRESHOLD (0.5) to produce the binary cut mask.
 *
 * Processing cost on a 2-minute short clip (3000 frames, 25 fps):
 *   • ffmpeg frame extraction at 48×27:  ~0.5 s
 *   • ONNX inference (40 windows):       ~1–3 s on CPU
 *   Total added latency:                 ~1.5–3.5 s (measured and logged)
 */

import * as fs    from 'fs';
import * as path  from 'path';
import * as https from 'https';
import { spawn }  from 'child_process';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MODELS_DIR = path.join(process.cwd(), 'models');
const MODEL_FILE = 'transnetv2.onnx';

export const TRANSNETV2_MODEL_PATH = path.join(MODELS_DIR, MODEL_FILE);

/**
 * HuggingFace ONNX export of TransNetV2 from elya5/transnetv2 (~31 MB, MIT).
 * Downloaded once at build time (Dockerfile) or on first use at runtime.
 */
const MODEL_URL =
  'https://huggingface.co/elya5/transnetv2/resolve/main/transnetv2.onnx';

/** Fixed window size required by the ONNX model's static input shape. */
const WINDOW_FRAMES = 100;

/**
 * Stride between consecutive inference windows.
 * 50-frame overlap (50 % stride) lets boundary frames be seen in two windows;
 * the higher of the two probabilities is kept (max aggregation).
 */
const WINDOW_STRIDE = 50;

/** Frame resolution expected by the model (width × height). */
const FRAME_W = 48;
const FRAME_H = 27;

/** Bytes per frame in raw rgb24 format (48 × 27 × 3 = 3 888). */
const FRAME_BYTES = FRAME_W * FRAME_H * 3;

/**
 * Per-frame probability above which a frame is classified as a shot boundary.
 * 0.5 is the standard TransNetV2 threshold from the original paper.
 */
const CUT_THRESHOLD = 0.5;

// ---------------------------------------------------------------------------
// Feature flag
// ---------------------------------------------------------------------------

let _enabled: boolean | null = null;

/** Returns false only when ENABLE_SHOT_BOUNDARY=false is explicitly set. */
export function isShotBoundaryEnabled(): boolean {
  if (_enabled === null) {
    _enabled = process.env.ENABLE_SHOT_BOUNDARY !== 'false';
  }
  return _enabled;
}

// ---------------------------------------------------------------------------
// Model download
// ---------------------------------------------------------------------------

/** Download the ONNX model to MODEL_PATH if it does not already exist. */
export async function ensureTransNetV2Model(): Promise<boolean> {
  if (fs.existsSync(TRANSNETV2_MODEL_PATH)) return true;

  fs.mkdirSync(MODELS_DIR, { recursive: true });
  console.log('[ShotBoundary] TransNetV2 model not found locally. Downloading from HuggingFace (~31 MB)…');
  console.log(`[ShotBoundary]   URL:  ${MODEL_URL}`);
  console.log(`[ShotBoundary]   Dest: ${TRANSNETV2_MODEL_PATH}`);

  const started = Date.now();
  try {
    await downloadFile(MODEL_URL, TRANSNETV2_MODEL_PATH);
    const elapsed = ((Date.now() - started) / 1000).toFixed(1);
    const sizeMB  = (fs.statSync(TRANSNETV2_MODEL_PATH).size / 1_048_576).toFixed(1);
    console.log(`[ShotBoundary] Model downloaded: ${sizeMB} MB in ${elapsed}s`);
    return true;
  } catch (err: any) {
    console.error('[ShotBoundary] Model download failed:', err.message);
    try { fs.unlinkSync(TRANSNETV2_MODEL_PATH); } catch { /* partial */ }
    return false;
  }
}

function downloadFile(url: string, dest: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);

    const doRequest = (reqUrl: string, redirects = 0) => {
      if (redirects > 8) { reject(new Error('Too many redirects')); return; }

      const mod: typeof https =
        reqUrl.startsWith('https') ? https : (require('http') as typeof https);

      mod.get(reqUrl, { headers: { 'User-Agent': 'node/20' } } as any, (res: any) => {
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          const loc = (res.headers.location as string).startsWith('http')
            ? res.headers.location
            : `https://huggingface.co${res.headers.location}`;
          doRequest(loc, redirects + 1);
          return;
        }
        if (res.statusCode !== 200) {
          reject(new Error(`HTTP ${res.statusCode} downloading TransNetV2 model`));
          return;
        }
        res.pipe(file);
        file.on('finish', () => file.close(() => resolve()));
        file.on('error', reject);
      }).on('error', reject);
    };
    doRequest(url);
  });
}

// ---------------------------------------------------------------------------
// ONNX session management (lazy, singleton)
// ---------------------------------------------------------------------------

let _session:    any    = null;
let _loading:    boolean = false;
let _loadError:  string | null = null;

/** Load (or return the cached) InferenceSession for the TransNetV2 model. */
export async function loadTransNetV2Session(): Promise<any | null> {
  if (_session)    return _session;
  if (_loadError)  return null;

  // Simple spin-wait if another call is already loading the model.
  if (_loading) {
    for (let i = 0; i < 30; i++) {
      await new Promise<void>(r => setTimeout(r, 200));
      if (_session || _loadError) break;
    }
    return _session;
  }

  _loading = true;
  try {
    const ok = await ensureTransNetV2Model();
    if (!ok) { _loadError = 'model unavailable'; return null; }

    const ort = await import('onnxruntime-node' as any);
    _session  = await ort.InferenceSession.create(TRANSNETV2_MODEL_PATH, {
      executionProviders: ['cpu'],
      intraOpNumThreads:  2,
      interOpNumThreads:  1,
    });
    console.log('[ShotBoundary] TransNetV2 ONNX session loaded.');
    return _session;
  } catch (err: any) {
    _loadError = err.message;
    console.error('[ShotBoundary] Failed to load TransNetV2 session:', err.message);
    return null;
  } finally {
    _loading = false;
  }
}

// ---------------------------------------------------------------------------
// Frame extraction via ffmpeg
// ---------------------------------------------------------------------------

/**
 * Strip the system lib paths that break Nix-provided ffmpeg binaries
 * (same pattern as pipeline.ts makeCleanEnv).
 */
function makeCleanEnv(): NodeJS.ProcessEnv {
  const env = { ...process.env };
  if (env.LD_LIBRARY_PATH) {
    const cleaned = env.LD_LIBRARY_PATH
      .split(':')
      .filter(p => p !== '/lib/x86_64-linux-gnu' && p !== '/usr/lib/x86_64-linux-gnu')
      .join(':');
    if (cleaned) env.LD_LIBRARY_PATH = cleaned;
    else         delete env.LD_LIBRARY_PATH;
  }
  return env;
}

/**
 * Extract all frames from videoPath at FRAME_W×FRAME_H in rgb24 format.
 * Returns a Buffer of raw frame data (FRAME_BYTES bytes per frame).
 *
 * Memory: at 48×27×3 = 3 888 bytes/frame, a 5-minute clip at 25 fps
 * (7 500 frames) produces ~29 MB — well within Node's buffer limit.
 */
function extractFramesRaw(videoPath: string, fps = 25): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    const proc = spawn('ffmpeg', [
      '-i',       videoPath,
      '-vf',      `scale=${FRAME_W}:${FRAME_H}`,
      '-pix_fmt', 'rgb24',
      '-r',       String(fps),
      '-f',       'rawvideo',
      '-',
    ], { env: makeCleanEnv(), stdio: ['ignore', 'pipe', 'ignore'] });

    proc.stdout.on('data', (chunk: Buffer) => chunks.push(chunk));
    proc.on('close', (code) => {
      // ffmpeg exits 0 even when it writes some frames and then hits EOF —
      // any non-zero exit is an error only if we got zero data.
      const buf = Buffer.concat(chunks);
      if (buf.length === 0 && code !== 0) {
        reject(new Error(`ffmpeg exited ${code} with no output`));
      } else {
        resolve(buf);
      }
    });
    proc.on('error', reject);
  });
}

// ---------------------------------------------------------------------------
// TransNetV2 inference (sliding-window)
// ---------------------------------------------------------------------------

/**
 * Run one inference window of WINDOW_FRAMES frames.
 * Input tensor: [1, WINDOW_FRAMES, FRAME_H, FRAME_W, 3], float32, values 0–1.
 * Returns [soft, hard] arrays each of length WINDOW_FRAMES.
 */
async function runOneWindow(
  session: any,
  ort: any,
  rawFrames: Buffer,
  winStart: number,
  numFrames: number,
): Promise<{ soft: Float32Array; hard: Float32Array }> {
  const data = new Float32Array(WINDOW_FRAMES * FRAME_H * FRAME_W * 3);

  for (let fi = 0; fi < WINDOW_FRAMES; fi++) {
    // Clamp to last real frame when window extends beyond the clip.
    const srcFi  = Math.min(winStart + fi, numFrames - 1);
    const srcOff = srcFi * FRAME_BYTES;
    const dstOff = fi   * FRAME_BYTES;
    // Divide by 255 to get [0, 1] range required by the model.
    for (let px = 0; px < FRAME_BYTES; px++) {
      data[dstOff + px] = rawFrames[srcOff + px] / 255;
    }
  }

  const tensor = new ort.Tensor('float32', data, [1, WINDOW_FRAMES, FRAME_H, FRAME_W, 3]);
  const out    = await session.run({ input: tensor });

  return {
    soft: out['534'].data as Float32Array,  // [100] soft-cut probabilities
    hard: out['535'].data as Float32Array,  // [100] hard-cut probabilities
  };
}

/**
 * Slide a 100-frame window across all frames with WINDOW_STRIDE stride.
 * Each frame's final score is the max probability seen across all windows
 * that covered it (max aggregation over the 50 % overlap).
 */
async function runTransNetV2(
  session: any,
  rawFrames: Buffer,
): Promise<Float32Array> {
  const ort       = await import('onnxruntime-node' as any);
  const numFrames = Math.floor(rawFrames.length / FRAME_BYTES);
  const scores    = new Float32Array(numFrames).fill(0);

  for (let winStart = 0; winStart < numFrames; winStart += WINDOW_STRIDE) {
    const winEnd = Math.min(winStart + WINDOW_FRAMES - 1, numFrames - 1);
    const winLen = winEnd - winStart + 1;

    const { soft, hard } = await runOneWindow(session, ort, rawFrames, winStart, numFrames);

    for (let fi = 0; fi < winLen; fi++) {
      const globalFi = winStart + fi;
      const sc = Math.max(soft[fi], hard[fi]);
      // Max aggregation: keep the highest score seen across overlapping windows.
      if (sc > scores[globalFi]) scores[globalFi] = sc;
    }
  }

  return scores;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Detect shot boundaries in a short-clip video using TransNetV2.
 *
 * This is run once per match operation, only on the short clip (not the full
 * movie).  The returned mask is OR'd with detectSceneCuts in
 * groundMatchedSegments — it is strictly additive and never replaces or
 * removes an existing threshold-based cut.
 *
 * @param videoPath   Absolute path to the short-clip video file.
 * @param frameCount  Number of fingerprint frames (determines mask length).
 * @param fps         Frame rate used during fingerprinting (default 25).
 * @returns           Uint8Array[frameCount] where 1 = shot boundary frame,
 *                    or null on any error (caller falls back to threshold-only).
 */
export async function detectShotBoundaries(
  videoPath: string,
  frameCount: number,
  fps = 25,
): Promise<Uint8Array | null> {
  const session = await loadTransNetV2Session();
  if (!session) return null;

  const t0 = Date.now();

  // ── Extract raw frames ──────────────────────────────────────────────────
  let rawFrames: Buffer;
  try {
    rawFrames = await extractFramesRaw(videoPath, fps);
  } catch (err: any) {
    console.error('[ShotBoundary] Frame extraction failed:', err.message);
    return null;
  }

  const extractedFrames = Math.floor(rawFrames.length / FRAME_BYTES);
  if (extractedFrames === 0) {
    console.warn('[ShotBoundary] No frames extracted from', videoPath);
    return null;
  }

  const tExtract = Date.now() - t0;

  // ── Run inference ────────────────────────────────────────────────────────
  let scores: Float32Array;
  try {
    scores = await runTransNetV2(session, rawFrames);
  } catch (err: any) {
    console.error('[ShotBoundary] TransNetV2 inference failed:', err.message);
    return null;
  }

  const tInfer = Date.now() - t0 - tExtract;

  // ── Build binary cut mask ────────────────────────────────────────────────
  // Frame index 0 is never a cut (it's the first frame of the clip).
  const cuts = new Uint8Array(frameCount);
  let detected = 0;
  for (let fi = 1; fi < Math.min(extractedFrames, frameCount); fi++) {
    if (scores[fi] >= CUT_THRESHOLD) {
      cuts[fi] = 1;
      detected++;
    }
  }

  const numWindows = Math.ceil(extractedFrames / WINDOW_STRIDE);
  const tTotal = Date.now() - t0;
  console.log(
    `[ShotBoundary] TransNetV2: ${extractedFrames} frame(s)` +
    ` → ${numWindows} window(s)` +
    ` → ${detected} cut(s) detected` +
    ` (extract ${tExtract} ms + infer ${tInfer} ms = ${tTotal} ms total).`
  );

  return cuts;
}
