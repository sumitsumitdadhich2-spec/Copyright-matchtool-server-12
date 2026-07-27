import { parentPort } from 'worker_threads';
import { createCanvas } from 'canvas';
import { getCropRects, processSubtitles, computeHashAndFeatures, computeSignature, FrameSignature, VariantHashes } from '../src/shared/fingerprint';
import {
  MODEL_PATH,
  CLIP_INPUT_SIZE,
  preprocessRgba,
  runClipInference,
  createOrtSession,
} from './embedding';
import * as fs from 'fs';

// ---------------------------------------------------------------------------
// Lazy CLIP session — initialised once per worker on first enableClip message
// ---------------------------------------------------------------------------

let _clipSession:   any    = null;
let _clipInitDone:  boolean = false;
let _clipInitError: string | null = null;
/** undefined = not yet measured, otherwise ms taken for last inference */
let _clipInferMs:   number | undefined;

async function initClipSession(): Promise<void> {
  if (_clipInitDone) return;
  _clipInitDone = true;

  if (!fs.existsSync(MODEL_PATH)) {
    _clipInitError = `Model not found at ${MODEL_PATH}`;
    return;
  }
  try {
    _clipSession = await createOrtSession();
  } catch (err: any) {
    _clipInitError = err.message || String(err);
  }
}

parentPort?.on('message', async (message) => {
  const { id, frameBuffer, width, height, enableClip } = message;
  
  try {
    const canvas = createCanvas(width, height);
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(width, height);
    imgData.data.set(new Uint8ClampedArray(frameBuffer));
    ctx.putImageData(imgData, 0, 0);
    
    const rects = getCropRects(width, height);
    const variants: Record<string, VariantHashes> = {};
    
    // Downscale full frame to a standard intermediate size
    const H_down = 120;
    const W_down = Math.round(width * (H_down / height));
    
    const fullDownCanvas = createCanvas(W_down, H_down);
    const fullDownCtx = fullDownCanvas.getContext('2d');
    fullDownCtx.patternQuality = 'best';
    fullDownCtx.quality = 'best';
    fullDownCtx.imageSmoothingEnabled = true;
    
    fullDownCtx.fillStyle = '#000000';
    fullDownCtx.fillRect(0, 0, W_down, H_down);
    fullDownCtx.drawImage(canvas, 0, 0, width, height, 0, 0, W_down, H_down);
    
    const imgDataDown = fullDownCtx.getImageData(0, 0, W_down, H_down);
    const { changed, maskCoverage } = processSubtitles(imgDataDown as any, false);
    if (changed) {
      fullDownCtx.putImageData(imgDataDown, 0, 0);
    }
    
    const finalCanvas = createCanvas(16, 16);
    const finalCtx = finalCanvas.getContext('2d');
    finalCtx.patternQuality = 'best';
    finalCtx.quality = 'best';
    finalCtx.imageSmoothingEnabled = true;
    
    const scaleX = W_down / width;
    const scaleY = H_down / height;

    let signature: FrameSignature | undefined;
    
    for (const rect of rects) {
      finalCtx.fillStyle = '#000000';
      finalCtx.fillRect(0, 0, 16, 16);
      if (!changed) {
        finalCtx.drawImage(canvas, rect.sx, rect.sy, rect.sw, rect.sh, 0, 0, 16, 16);
      } else {
        const sx = rect.sx * scaleX;
        const sy = rect.sy * scaleY;
        const sw = rect.sw * scaleX;
        const sh = rect.sh * scaleY;
        finalCtx.drawImage(fullDownCanvas, sx, sy, sw, sh, 0, 0, 16, 16);
      }
      const finalImgData = finalCtx.getImageData(0, 0, 16, 16);

      // Compute signature only for the 'full' variant (one per frame)
      const isFullVariant = rect.name === 'full';
      const features = computeHashAndFeatures(finalImgData as any, isFullVariant);
      variants[rect.name] = {
        hash: features.hash,
        dhash: features.dhash,
        fhash: features.fhash,
        fdhash: features.fdhash,
        phash: features.phash,
      };
      if (isFullVariant && features.signature) {
        signature = features.signature;
      }
    }

    // ── CLIP embedding (optional, only when enableClip = true) ──────────────
    let embedding: number[] | undefined;
    let clipInferMs: number | undefined;
    let clipError:   string  | undefined;

    if (enableClip) {
      // Lazy init the ONNX session on first CLIP-enabled frame
      if (!_clipInitDone) await initClipSession();

      if (_clipSession) {
        try {
          // Downsample the original frame to 224×224 using canvas
          const clipCanvas = createCanvas(CLIP_INPUT_SIZE, CLIP_INPUT_SIZE);
          const clipCtx    = clipCanvas.getContext('2d');
          clipCtx.imageSmoothingEnabled = true;
          (clipCtx as any).patternQuality = 'best';
          (clipCtx as any).quality        = 'best';
          clipCtx.fillStyle = '#000000';
          clipCtx.fillRect(0, 0, CLIP_INPUT_SIZE, CLIP_INPUT_SIZE);
          clipCtx.drawImage(canvas, 0, 0, width, height, 0, 0, CLIP_INPUT_SIZE, CLIP_INPUT_SIZE);
          const clipRgba = clipCtx.getImageData(0, 0, CLIP_INPUT_SIZE, CLIP_INPUT_SIZE);

          const tensor = preprocessRgba(
            clipRgba.data as unknown as Uint8ClampedArray,
            CLIP_INPUT_SIZE,
            CLIP_INPUT_SIZE,
          );

          const t0  = Date.now();
          const emb = await runClipInference(_clipSession, tensor);
          clipInferMs = Date.now() - t0;
          _clipInferMs = clipInferMs;

          // Transfer as plain number[] so postMessage serialises it cleanly
          embedding = Array.from(emb);
        } catch (err: any) {
          clipError = err.message || String(err);
        }
      } else if (_clipInitError) {
        clipError = _clipInitError;
      }
    }
    
    parentPort?.postMessage({
      id,
      result: { variants, signature, embedding, clipInferMs, clipError, maskCoverage: changed ? maskCoverage : 0 },
    });
  } catch (error: any) {
    parentPort?.postMessage({ id, error: error.message || String(error) });
  }
});
