/**
 * CLIP ViT-B/32 image embedding support for Nexus Video Match.
 *
 * Uses onnxruntime-node to run a quantized CLIP vision encoder locally on CPU.
 * The model is downloaded once to models/clip_vision_quantized.onnx and reused.
 *
 * Controlled by ENABLE_CLIP_MATCHING=true environment variable.
 * Falls back cleanly to hash-only matching if disabled or model unavailable.
 *
 * CLIP preprocessing (official openai/clip values):
 *   Resize to 224×224 → float32 [C,H,W] → normalize per channel with
 *   mean = [0.48145466, 0.4578275,  0.40821073]
 *   std  = [0.26862954, 0.26130258, 0.27577711]
 */

import * as fs   from 'fs';
import * as path from 'path';
import * as https from 'https';
import * as http  from 'http';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

export const CLIP_INPUT_SIZE = 224;
export const CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073];
export const CLIP_STD  = [0.26862954, 0.26130258, 0.27577711];

const MODELS_DIR  = path.join(process.cwd(), 'models');
const MODEL_FILE  = 'clip_vision_quantized.onnx';
export const MODEL_PATH = path.join(MODELS_DIR, MODEL_FILE);

/** HuggingFace URL for the Xenova quantized CLIP ViT-B/32 image encoder. */
const MODEL_URL =
  'https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model_quantized.onnx';

// ---------------------------------------------------------------------------
// Runtime state
// ---------------------------------------------------------------------------

let _clipEnabled: boolean | null = null;
export function isClipEnabled(): boolean {
  if (_clipEnabled === null) {
    _clipEnabled = process.env.ENABLE_CLIP_MATCHING === 'true';
  }
  return _clipEnabled;
}

// ---------------------------------------------------------------------------
// Model download
// ---------------------------------------------------------------------------

/** Download the ONNX model to MODEL_PATH if it doesn't already exist. */
export async function ensureModelDownloaded(): Promise<boolean> {
  if (fs.existsSync(MODEL_PATH)) return true;

  fs.mkdirSync(MODELS_DIR, { recursive: true });
  console.log('[CLIP] Model not found locally. Downloading from HuggingFace (~88 MB)…');
  console.log(`[CLIP]   URL:  ${MODEL_URL}`);
  console.log(`[CLIP]   Dest: ${MODEL_PATH}`);

  const started = Date.now();
  try {
    await downloadFile(MODEL_URL, MODEL_PATH);
    const elapsed = ((Date.now() - started) / 1000).toFixed(1);
    const sizeMB  = (fs.statSync(MODEL_PATH).size / 1_048_576).toFixed(1);
    console.log(`[CLIP] Model downloaded: ${sizeMB} MB in ${elapsed}s`);
    return true;
  } catch (err: any) {
    console.error('[CLIP] Model download failed:', err.message);
    try { fs.unlinkSync(MODEL_PATH); } catch { /* partial file */ }
    return false;
  }
}

function downloadFile(url: string, dest: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    const protocol: typeof https | typeof http = url.startsWith('https') ? https : http;

    const doRequest = (reqUrl: string, redirects = 0) => {
      if (redirects > 5) { reject(new Error('Too many redirects')); return; }
      protocol.get(reqUrl, (res) => {
        if (res.statusCode && res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          doRequest(res.headers.location, redirects + 1);
          return;
        }
        if (res.statusCode !== 200) {
          reject(new Error(`HTTP ${res.statusCode}`));
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
// Preprocessing helpers (pure math — no canvas dependency)
// ---------------------------------------------------------------------------

/**
 * Bilinear resize of RGBA pixel data (Uint8ClampedArray) to 224×224,
 * then convert to float32 CHW tensor with CLIP normalisation.
 *
 * Returns a Float32Array of length 3 * 224 * 224 in [C,H,W] order.
 */
export function preprocessRgba(
  rgba: Uint8ClampedArray,
  srcW: number,
  srcH: number,
): Float32Array {
  const DST = CLIP_INPUT_SIZE;
  const out  = new Float32Array(3 * DST * DST);

  const scaleX = srcW / DST;
  const scaleY = srcH / DST;

  for (let dy = 0; dy < DST; dy++) {
    const sy  = dy * scaleY;
    const sy0 = Math.min(Math.floor(sy), srcH - 1);
    const sy1 = Math.min(sy0 + 1, srcH - 1);
    const fy  = sy - sy0;

    for (let dx = 0; dx < DST; dx++) {
      const sx  = dx * scaleX;
      const sx0 = Math.min(Math.floor(sx), srcW - 1);
      const sx1 = Math.min(sx0 + 1, srcW - 1);
      const fx  = sx - sx0;

      // Bilinear sample
      for (let c = 0; c < 3; c++) {
        const i00 = (sy0 * srcW + sx0) * 4 + c;
        const i10 = (sy0 * srcW + sx1) * 4 + c;
        const i01 = (sy1 * srcW + sx0) * 4 + c;
        const i11 = (sy1 * srcW + sx1) * 4 + c;

        const v = (rgba[i00] * (1 - fx) + rgba[i10] * fx) * (1 - fy)
                + (rgba[i01] * (1 - fx) + rgba[i11] * fx) * fy;

        // Normalize: pixel/255 → (x - mean) / std, store CHW
        out[c * DST * DST + dy * DST + dx] = (v / 255 - CLIP_MEAN[c]) / CLIP_STD[c];
      }
    }
  }

  return out;
}

// ---------------------------------------------------------------------------
// ONNX inference
// ---------------------------------------------------------------------------

/**
 * Run the CLIP vision encoder on a preprocessed [3,224,224] float32 tensor.
 * Returns a L2-normalised 768-dim embedding (or whatever dim the model outputs).
 *
 * @param session   onnxruntime InferenceSession (already loaded)
 * @param tensor    Float32Array [3,224,224] from preprocessRgba
 */
export async function runClipInference(
  session: any,
  tensor: Float32Array,
): Promise<Float32Array> {
  // onnxruntime-node is imported dynamically so the server still starts even
  // if the package is missing (graceful degradation for hash-only mode).
  const ort = await import('onnxruntime-node' as any);

  const inputTensor = new ort.Tensor('float32', tensor, [1, 3, CLIP_INPUT_SIZE, CLIP_INPUT_SIZE]);
  const feeds: Record<string, any> = { pixel_values: inputTensor };

  const result = await session.run(feeds);

  // Extract embedding: prefer pooler_output; fall back to CLS token of last_hidden_state
  let raw: Float32Array;
  if (result['pooler_output']) {
    const t = result['pooler_output'];
    raw = t.data as Float32Array;
  } else if (result['last_hidden_state']) {
    const t = result['last_hidden_state'];
    // Shape [1, seq_len, hidden_dim] — take CLS token (index 0)
    const data     = t.data as Float32Array;
    const hiddenDim = t.dims[2] as number;
    raw = data.slice(0, hiddenDim);
  } else {
    // Unknown output — return the first tensor's data
    const key = Object.keys(result)[0];
    raw = result[key].data as Float32Array;
  }

  return l2Normalize(raw);
}

/** Create an onnxruntime InferenceSession for the CLIP vision model. */
export async function createOrtSession(): Promise<any> {
  const ort = await import('onnxruntime-node' as any);
  const session = await ort.InferenceSession.create(MODEL_PATH, {
    executionProviders: ['cpu'],
    // 1 thread per worker session — multiple workers run in parallel at the process level
    intraOpNumThreads: 1,
    interOpNumThreads: 1,
  });
  return session;
}

// ---------------------------------------------------------------------------
// Math utilities
// ---------------------------------------------------------------------------

function l2Normalize(v: Float32Array): Float32Array {
  let norm = 0;
  for (let i = 0; i < v.length; i++) norm += v[i] * v[i];
  norm = Math.sqrt(norm);
  if (norm < 1e-9) return v;
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / norm;
  return out;
}

/**
 * Cosine similarity between two L2-normalised embeddings.
 * Returns value in [–1, 1]; mapped to 0–100 by the caller.
 */
export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  const len = Math.min(a.length, b.length);
  let dot = 0;
  for (let i = 0; i < len; i++) dot += a[i] * b[i];
  return Math.max(-1, Math.min(1, dot));
}

// ---------------------------------------------------------------------------
// Embeddings binary file helpers
// ---------------------------------------------------------------------------

/**
 * Load a .embeddings.bin file as a flat Float32Array.
 *
 * Format: raw IEEE 754 float32 values, little-endian, row-major.
 *   embDim  = fileSize / frameCount / 4
 *   frame i starts at byte: i * embDim * 4
 *
 * Returns null if the file does not exist or cannot be read.
 */
export function loadEmbeddingsFile(binPath: string): Float32Array | null {
  if (!fs.existsSync(binPath)) return null;
  try {
    const buf = fs.readFileSync(binPath);
    return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  } catch (err: any) {
    console.warn(`[CLIP] Could not load embeddings: ${binPath} — ${err.message}`);
    return null;
  }
}

/**
 * Write a flat Float32Array of embeddings to a binary file.
 * Appends `dim` float values for each frame.
 */
export function writeEmbeddingsBatch(
  fd: number,
  embeddings: Float32Array[],
): void {
  for (const emb of embeddings) {
    const buf = Buffer.from(emb.buffer, emb.byteOffset, emb.byteLength);
    fs.writeSync(fd, buf);
  }
}

/**
 * Retrieve a single frame embedding from a pre-loaded flat array.
 * Returns null if embFlat is null or frameIdx is out of range.
 */
export function getEmbedding(
  embFlat: Float32Array,
  embDim: number,
  frameIdx: number,
): Float32Array | null {
  if (!embFlat || embDim <= 0) return null;
  const off = frameIdx * embDim;
  if (off + embDim > embFlat.length) return null;
  return embFlat.subarray(off, off + embDim);
}
