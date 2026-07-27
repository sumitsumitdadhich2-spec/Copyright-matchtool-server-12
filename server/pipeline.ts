import { spawn, execSync } from 'child_process';
import { Worker } from 'worker_threads';
import * as os from 'os';
import * as path from 'path';
import * as fs from 'fs';
import { fileURLToPath } from 'url';
import { FrameSignature } from '../src/shared/fingerprint';
import { isClipEnabled } from './embedding';

/** True when ENABLE_CLIP_MATCHING=true in the process environment. */
const CLIP_ENABLED = isClipEnabled();

const getDirname = () => {
  if (typeof __dirname !== 'undefined') {
    return __dirname;
  }
  try {
    const metaUrl = typeof import.meta !== 'undefined' && import.meta.url ? import.meta.url : '';
    if (metaUrl) {
      return path.dirname(fileURLToPath(metaUrl));
    }
  } catch (e) {
    // ignore
  }
  return process.cwd();
};

const currentDirname = getDirname();

export const NUM_WORKERS = Math.max(1, Math.min(os.cpus().length, 128));

/**
 * Build a clean env for Nix-provided binaries (ffprobe, ffmpeg).
 * When server.ts prepends /lib/x86_64-linux-gnu to LD_LIBRARY_PATH so that
 * canvas.node can find libuuid.so.1, the system libmount.so.1 ends up before
 * the Nix-provided one in the search order.  The system copy is older and
 * lacks the MOUNT_2_40 versioned symbol that Nix glib requires, causing:
 *   ffprobe: libmount.so.1: version `MOUNT_2_40' not found
 * Fix: strip the system lib paths from LD_LIBRARY_PATH before spawning any
 * Nix binary.  canvas workers inherit the unmodified process env and still
 * find libuuid via the $ORIGIN symlink in node_modules/canvas/build/Release/.
 */
function makeCleanEnv(): NodeJS.ProcessEnv {
  const env = { ...process.env };
  if (env.LD_LIBRARY_PATH) {
    const cleaned = env.LD_LIBRARY_PATH
      .split(':')
      .filter(p => p !== '/lib/x86_64-linux-gnu' && p !== '/usr/lib/x86_64-linux-gnu')
      .join(':');
    if (cleaned) {
      env.LD_LIBRARY_PATH = cleaned;
    } else {
      delete env.LD_LIBRARY_PATH;
    }
  }
  return env;
}

// ---------------------------------------------------------------------------
// How often to attempt a flush (every N processed frames)
// ---------------------------------------------------------------------------
const FLUSH_EVERY = 100;

// Flush when contiguous completed frames in Map >= this many
const FLUSH_BATCH = 1500;

// Flush when process RSS exceeds this (bytes).  5.5 GB gives headroom below
// the user's 6 GB limit while still leaving room for the OS + workers.
const RAM_FLUSH_THRESHOLD_BYTES = 5.5 * 1024 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Checkpoint: save progress to disk every N frames flushed so processing can
// resume after a server restart without re-processing from scratch.
// ---------------------------------------------------------------------------
const CHECKPOINT_EVERY = 5000;

/** Options for resumable extraction */
export interface ExtractionOptions {
  /** Number of frames already written to outputPath from a previous run (0 = fresh start) */
  resumeFrom?: number;
  /** Full path of the checkpoint JSON file to create/update during processing */
  checkpointPath?: string;
  /** Job ID — stored in the checkpoint so server.ts can match it on resume */
  jobId?: string;
  /** "filename:filesize" key — stored in checkpoint for fast lookup */
  checkpointKey?: string;
  /**
   * AbortSignal to cancel the pipeline mid-run.
   * When aborted, ffmpeg and workers are killed, the write stream is destroyed,
   * and the returned Promise rejects with Error('STOPPED').
   */
  abortSignal?: AbortSignal;
}

export interface FingerprintResult {
  frameIndex: number;
  timestamp: number;
  variants: Record<string, { hash: string }>;
  signature?: FrameSignature;
  maskCoverage?: number;
}

/**
 * Extract per-frame fingerprints from a video file and write them as NDJSON
 * (one JSON object per line) directly to `outputPath`.
 *
 * RAM usage is kept low by flushing completed frames to disk as they arrive,
 * rather than accumulating the entire fingerprint set in memory until the end.
 *
 * @returns Promise that resolves with the total frame count once done.
 */
export function extractFingerprints(
  videoPath: string,
  outputPath: string,
  onProgress?: (decoded: number, processed: number) => void,
  options: ExtractionOptions = {}
): Promise<number> {
  const { resumeFrom = 0, checkpointPath, jobId, checkpointKey, abortSignal } = options;

  return new Promise((resolve, reject) => {
    const workers: Worker[] = [];
    let idleWorkers: Worker[] = [];
    const activeTasks = new Map<number, { resolve: Function; reject: Function }>();
    let taskIdCounter = 0;
    let decoded = 0;
    let processed = 0;
    // Frames skipped at the start of a resumed run (already written in a previous run).
    let skipped = 0;
    // Holds frames that have been computed but not yet written to disk.
    const fingerprints = new Map<number, {
      variants: any;
      signature?: FrameSignature;
      embedding?: number[];
      clipInferMs?: number;
      clipError?: string;
      maskCoverage?: number;
    }>();
    const taskQueue: { id: number; frameBuffer: Buffer; width: number; height: number; frameIndex: number }[] = [];

    // ── CLIP embedding state (only used when CLIP_ENABLED) ─────────────────
    // embeddings[i] is a Float32Array (or null for failed frames) for frameIndex i.
    // Keyed by the same 1-based frameIndex used by `fingerprints`.
    const embMap = new Map<number, Float32Array | null>();
    let embDim = 0;
    let clipFrameCount   = 0;
    let clipErrorCount   = 0;
    let clipTotalInferMs = 0;
    let ffmpegProcess: any = null;
    let isFinished = false;

    // Track the highest frame index already written to disk.
    // Initialised to resumeFrom so flushToStream starts writing from the right offset.
    let lastFlushedFrame = resumeFrom;

    // Checkpoint state — prevents concurrent writes and tracks last saved position.
    let checkpointPending = false;
    let lastCheckpointAt = resumeFrom; // lastFlushedFrame value at last checkpoint save

    // Set after ffprobe; controls how many raw-pixel frames we allow in the
    // task queue at once.  At 1 080p a frame is ~8 MB; at 4K it is ~33 MB.
    // Hardcoding 100 was safe for 1 080p (800 MB queue) but blew past 8 GB for
    // 4K content (100 × 33 MB = 3.3 GB before any fingerprints are computed).
    let frameBytes       = 0;
    let dynamicQueueLimit = 100; // default until ffprobe fills this in

    // ── Write stream ─────────────────────────────────────────────────────────
    // Use append mode when resuming so already-written frames are preserved.
    let writeStreamErr: Error | null = null;
    const writeStream = fs.createWriteStream(outputPath, {
      encoding: 'utf8',
      flags: resumeFrom > 0 ? 'a' : 'w',
    });
    writeStream.on('error', (err) => { writeStreamErr = err; });

    // ── Abort / cancel support ────────────────────────────────────────────────
    // When the caller signals cancellation (e.g. from the Stop button), we kill
    // ffmpeg, terminate all workers, destroy the write stream, and reject the
    // Promise with the sentinel string 'STOPPED' so server.ts can distinguish
    // a user-initiated stop from a processing error.
    if (abortSignal) {
      if (abortSignal.aborted) {
        // Already cancelled before the pipeline even started — bail immediately.
        writeStream.destroy();
        reject(new Error('STOPPED'));
        return;
      }
      abortSignal.addEventListener('abort', () => {
        if (isFinished) return;
        isFinished = true;
        try { ffmpegProcess?.kill('SIGTERM'); } catch { /* ignore */ }
        cleanupWorkers();
        try { writeStream.destroy(); } catch { /* ignore */ }
        reject(new Error('STOPPED'));
      }, { once: true });
    }

    // ── Flush helper ─────────────────────────────────────────────────────────
    /**
     * Write the longest contiguous run of completed frames (starting at
     * lastFlushedFrame+1) to disk, then delete them from the Map.
     *
     * Called periodically from the worker message handler and once more
     * (force=true) after all frames are done.
     */
    function flushToStream(force = false): void {
      const rss = process.memoryUsage().rss;
      const shouldFlush =
        force ||
        rss >= RAM_FLUSH_THRESHOLD_BYTES ||
        fingerprints.size >= FLUSH_BATCH;
      if (!shouldFlush) return;

      // Walk forward from the last flushed position while frames are present.
      let hi = lastFlushedFrame;
      while (fingerprints.has(hi + 1)) hi++;
      if (hi === lastFlushedFrame) return; // nothing contiguous to flush

      const FRAME_RATE = 25;
      for (let i = lastFlushedFrame + 1; i <= hi; i++) {
        const fp = fingerprints.get(i);
        if (fp) {
          const line =
            JSON.stringify({
              frameIndex: i,
              timestamp: (i - 1) / FRAME_RATE,
              variants: fp.variants,
              signature: fp.signature,
              ...(fp.maskCoverage ? { maskCoverage: fp.maskCoverage } : {}),
            }) + '\n';
          writeStream.write(line);
          fingerprints.delete(i); // free memory
        }
      }
      lastFlushedFrame = hi;

      // ── Async checkpoint write (non-blocking, every CHECKPOINT_EVERY frames) ─
      // We write after lastFlushedFrame advances so the checkpoint always reflects
      // data that is actually on disk. checkpointPending prevents concurrent writes.
      if (
        checkpointPath && jobId && checkpointKey &&
        lastFlushedFrame - lastCheckpointAt >= CHECKPOINT_EVERY &&
        !checkpointPending
      ) {
        lastCheckpointAt = lastFlushedFrame;
        checkpointPending = true;
        const cpData = JSON.stringify({ jobId, checkpointKey, updatedAt: Date.now() });
        fs.promises.writeFile(checkpointPath, cpData)
          .catch(e => console.error(`[Checkpoint] Write failed for ${jobId}:`, e))
          .finally(() => { checkpointPending = false; });
      }
    }

    // ── Worker lifecycle ─────────────────────────────────────────────────────
    const cleanupWorkers = () => {
      for (const w of workers) {
        w.terminate().catch(() => {});
      }
    };

    function assignTasks() {
      while (idleWorkers.length > 0 && taskQueue.length > 0) {
        const worker = idleWorkers.pop()!;
        const task = taskQueue.shift()!;
        worker.postMessage({
          id: task.id,
          frameBuffer: task.frameBuffer,
          width: task.width,
          height: task.height,
          enableClip: CLIP_ENABLED,
        });
      }
      // Resume only when the queue has drained to half the limit AND RSS is
      // comfortably below the flush threshold (prevents yo-yo pausing).
      const resumeAt = Math.max(2, Math.floor(dynamicQueueLimit / 2));
      if (
        taskQueue.length < resumeAt &&
        process.memoryUsage().rss < RAM_FLUSH_THRESHOLD_BYTES * 0.85 &&
        ffmpegProcess?.stdout.isPaused()
      ) {
        ffmpegProcess.stdout.resume();
      }
    }

    try {
      const isProd = process.env.NODE_ENV === 'production';
      // In dev, import.meta.url is undefined under tsx ESM mode, so getDirname()
      // falls back to process.cwd() (project root) — not server/.  Resolve
      // explicitly from CWD so the path is always correct regardless of how
      // tsx initialises import.meta.
      const workerPath = isProd
        ? path.join(currentDirname, 'worker.cjs')
        : path.resolve(process.cwd(), 'server', 'worker.ts');

      for (let i = 0; i < NUM_WORKERS; i++) {
        const worker = new Worker(workerPath, isProd ? {} : {
          execArgv: ['-r', 'tsx/cjs']
        });

        worker.on('message', (msg) => {
          idleWorkers.push(worker);
          const task = activeTasks.get(msg.id);
          if (task) {
            activeTasks.delete(msg.id);
            if (msg.error) {
              task.reject(new Error(msg.error));
            } else {
              processed++;
              task.resolve(msg.result);
              if (onProgress) {
                onProgress(decoded, processed + skipped);
              }
            }
          }
          // Periodically attempt to flush completed frames to disk.
          if (processed % FLUSH_EVERY === 0) {
            flushToStream();
          }
          assignTasks();
        });

        worker.on('error', (err) => {
          console.error(`Worker error:`, err);
          const index = workers.indexOf(worker);
          if (index !== -1) workers.splice(index, 1);
          const idleIndex = idleWorkers.indexOf(worker);
          if (idleIndex !== -1) idleWorkers.splice(idleIndex, 1);
          assignTasks();
        });

        workers.push(worker);
        idleWorkers.push(worker);
      }

      // ── Query video dimensions via ffprobe ───────────────────────────────
      let width = 0;
      let height = 0;
      try {
        const ffprobeOutput = execSync(
          `ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "${videoPath}"`,
          { env: makeCleanEnv() }
        ).toString().trim();
        const [w, h] = ffprobeOutput.split('x').map(Number);
        if (!w || !h || isNaN(w) || isNaN(h)) {
          throw new Error(`Failed to parse resolution: ${ffprobeOutput}`);
        }
        width = w;
        height = h;
      } catch (err: any) {
        console.error(`ffprobe error on ${videoPath}:`, err);
        throw new Error(`Could not determine video dimensions: ${err.message}`);
      }

      // ── Dynamic queue limit ──────────────────────────────────────────────
      // Cap raw-frame RAM in the task queue at ~1.5 GB regardless of resolution.
      //   1 080p  (8.3 MB/frame) → ~180 frames in queue  (~1.5 GB)
      //   4K      (33  MB/frame) →  ~45 frames in queue  (~1.5 GB)
      frameBytes        = width * height * 4;
      const QUEUE_RAM_CAP = 1.5 * 1024 * 1024 * 1024;
      dynamicQueueLimit = Math.max(4, Math.min(500, Math.floor(QUEUE_RAM_CAP / frameBytes)));
      console.log(
        `Pipeline starting for ${videoPath} (${width}x${height})` +
        ` — frame ${(frameBytes / 1_048_576).toFixed(1)} MB` +
        ` — queue limit ${dynamicQueueLimit} frames (~${(dynamicQueueLimit * frameBytes / 1_073_741_824).toFixed(2)} GB)`
      );

      ffmpegProcess = spawn('ffmpeg', [
        '-i', videoPath,
        '-f', 'rawvideo',
        '-pix_fmt', 'rgba',
        '-r', '25',
        '-'
      ], { env: makeCleanEnv() });

      let buffer = Buffer.alloc(0);
      const frameSize = width * height * 4;

      ffmpegProcess.stdout.on('data', (chunk: Buffer) => {
        buffer = Buffer.concat([buffer, chunk]);

        while (buffer.length >= frameSize) {
          const frameBuffer = buffer.slice(0, frameSize);
          buffer = buffer.slice(frameSize);

          decoded++;

          // ── Resume skip: discard frames already written in a previous run ──
          // ffmpeg decodes from the start; we simply throw away raw pixel data
          // for frames we already have.  No workers are involved, so fingerprint
          // quality for the resumed portion is identical to a fresh run.
          if (decoded <= resumeFrom) {
            skipped++;
            if (onProgress && skipped % 1000 === 0) {
              onProgress(decoded, skipped); // show fast-forward progress
            }
            // Don't pause ffmpeg during skip — queue is empty, no backpressure.
            continue;
          }

          const id = ++taskIdCounter;
          const currentFrame = decoded;

          const p = new Promise<{
            variants: any;
            signature?: FrameSignature;
            embedding?: number[];
            clipInferMs?: number;
            clipError?: string;
            maskCoverage?: number;
          }>((res, rej) => {
            activeTasks.set(id, { resolve: res, reject: rej });
          });

          p.then((result) => {
            fingerprints.set(currentFrame, result);
            // ── Collect CLIP embedding (when enabled) ──────────────────────
            if (CLIP_ENABLED) {
              if (result.embedding && result.embedding.length > 0) {
                embMap.set(currentFrame, new Float32Array(result.embedding));
                if (!embDim) embDim = result.embedding.length;
                clipFrameCount++;
                if (result.clipInferMs) clipTotalInferMs += result.clipInferMs;
              } else {
                embMap.set(currentFrame, null); // failed or unavailable
                if (result.clipError) clipErrorCount++;
              }
            }
          }).catch((err) => {
            console.error(`Error processing frame ${currentFrame}:`, err);
            processed++;
            if (onProgress) {
              onProgress(decoded, processed + skipped);
            }
          });

          taskQueue.push({
            id,
            frameBuffer,
            width,
            height,
            frameIndex: currentFrame
          });

          // Pause ffmpeg when the queue is full OR when total RSS is high.
          // Both conditions are checked so a very large video resolution
          // triggers pause even before the frame count limit is reached.
          const rssNow = process.memoryUsage().rss;
          if (
            (taskQueue.length >= dynamicQueueLimit || rssNow >= RAM_FLUSH_THRESHOLD_BYTES) &&
            !ffmpegProcess.stdout.isPaused()
          ) {
            ffmpegProcess.stdout.pause();
          }

          assignTasks();
        }
      });

      ffmpegProcess.stderr.on('data', (_data: Buffer) => {
        // suppress ffmpeg stderr
      });

      ffmpegProcess.on('error', (err: any) => {
        console.error('ffmpeg process error:', err);
        cleanupWorkers();
        reject(err);
      });

      ffmpegProcess.on('close', (_code: any) => {
        const checkInterval = setInterval(() => {
          if ((processed + skipped) >= decoded) {
            clearInterval(checkInterval);
            if (!isFinished) {
              isFinished = true;
              cleanupWorkers();

              // Final flush — write any frames still in the Map.
              flushToStream(true);

              writeStream.end(() => {
                if (writeStreamErr) {
                  reject(writeStreamErr);
                  return;
                }

                // ── Write CLIP embeddings binary (when enabled) ───────────
                if (CLIP_ENABLED && embDim > 0) {
                  const embPath = outputPath + '.embeddings.bin';
                  const binStart = Date.now();
                  try {
                    const totalFrames = decoded;
                    const outBuf = Buffer.allocUnsafe(totalFrames * embDim * 4);
                    for (let fi = 1; fi <= totalFrames; fi++) {
                      const emb = embMap.get(fi);
                      const off = (fi - 1) * embDim * 4;
                      if (emb && emb.length >= embDim) {
                        Buffer.from(emb.buffer, emb.byteOffset, embDim * 4).copy(outBuf, off);
                      } else {
                        outBuf.fill(0, off, off + embDim * 4); // 0 = unavailable
                      }
                    }
                    fs.writeFileSync(embPath, outBuf);
                    const avgMs  = clipFrameCount > 0
                      ? (clipTotalInferMs / clipFrameCount).toFixed(1)
                      : 'n/a';
                    const sizeMB = (outBuf.byteLength / 1_048_576).toFixed(1);
                    console.log(
                      `[CLIP] Embeddings written: ${embPath}` +
                      ` — ${sizeMB} MB, dim=${embDim}` +
                      `, ${clipFrameCount}/${totalFrames} frames embedded` +
                      `, avg ${avgMs} ms/frame inference` +
                      `, ${clipErrorCount} error(s)` +
                      `, bin write ${Date.now() - binStart} ms`
                    );
                  } catch (err: any) {
                    console.error(`[CLIP] Failed to write embeddings binary: ${err.message}`);
                    // Non-fatal — matching engine falls back to hash-only
                  }
                } else if (CLIP_ENABLED && clipFrameCount === 0 && decoded > 0) {
                  console.warn(
                    '[CLIP] ENABLE_CLIP_MATCHING=true but no embeddings were produced. ' +
                    'Is the CLIP model downloaded? (models/clip_vision_quantized.onnx)'
                  );
                }

                resolve(decoded);
              });
            }
          }
        }, 100);
      });

    } catch (err) {
      cleanupWorkers();
      reject(err);
    }
  });
}
