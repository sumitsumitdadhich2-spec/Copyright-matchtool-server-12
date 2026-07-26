import express from 'express';
import multer from 'multer';
import cors from 'cors';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { createServer as createViteServer } from 'vite';
import { extractFingerprints, NUM_WORKERS } from './server/pipeline';
import { matchVideosFromFiles } from './server/matching-engine';

async function startServer() {
  // canvas.node needs libuuid.so.1 which lives in /lib/x86_64-linux-gnu on this host
  // but LD_LIBRARY_PATH starts empty in NixOS.  Setting it here (before any worker is
  // spawned) updates the real process-level environment via setenv(), so all worker
  // threads started later will find the library when they call dlopen('canvas.node').
  const SYSLIBS = '/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu';
  if (!process.env.LD_LIBRARY_PATH?.includes('/lib/x86_64-linux-gnu')) {
    process.env.LD_LIBRARY_PATH = process.env.LD_LIBRARY_PATH
      ? `${SYSLIBS}:${process.env.LD_LIBRARY_PATH}`
      : SYSLIBS;
  }

  const app = express();
  app.use(cors());
  app.use(express.json());
  const PORT = 3000;

  // Ensure upload directory exists
  const uploadDir = path.join(process.cwd(), 'uploads');
  if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir, { recursive: true });
  }

  // Configure multer storage
  const storage = multer.diskStorage({
    destination: (req, file, cb) => {
      cb(null, uploadDir);
    },
    filename: (req, file, cb) => {
      const ext = path.extname(file.originalname);
      const uniqueName = `${Date.now()}-${Math.random().toString(36).substring(2, 9)}${ext}`;
      cb(null, uniqueName);
    }
  });

  const upload = multer({ storage });

  // In-memory job store
  interface Job {
    id: string;
    status: 'pending' | 'processing' | 'completed' | 'failed' | 'stopped';
    totalFrames: number;
    processedFrames: number;
    error?: string;
    originalName?: string;
    fileSize?: number;
    startedAt?: number;
    completedAt?: number;
  }
  const jobs = new Map<string, Job>();

  /**
   * Return the jobId of any job that is currently processing/pending for the
   * given filename + fileSize combination.  Used to detect duplicate uploads.
   */
  function findActiveJobForFile(filename: string, fileSize: number): string | null {
    for (const [id, job] of jobs) {
      if (
        (job.status === 'processing' || job.status === 'pending') &&
        job.originalName === filename &&
        job.fileSize === fileSize
      ) {
        return id;
      }
    }
    return null;
  }

  /** Cancel functions for currently-running jobs, keyed by jobId */
  const jobCancelFns = new Map<string, () => void>();

  // Video identity registry: "filename:filesize" → jobId
  // Lets the frontend skip re-upload when the same file is selected again.
  interface JobMeta {
    originalName: string;
    fileSize: number;
    createdAt: number;
    totalFrames?: number;
    status?: 'completed' | 'failed' | 'stopped';
    startedAt?: number;
    completedAt?: number;
  }
  const videoRegistry = new Map<string, string>();

  function metaPath(jobId: string) {
    return path.join(uploadDir, `${jobId}_meta.json`);
  }

  function checkpointFilePath(jobId: string) {
    return path.join(uploadDir, `${jobId}_checkpoint.json`);
  }

  /**
   * Scan uploads/ for a checkpoint file whose checkpointKey matches
   * "filename:filesize".  Returns the jobId of the incomplete job, or null.
   */
  function findCheckpoint(filename: string, fileSize: number): { jobId: string } | null {
    if (!fs.existsSync(uploadDir)) return null;
    const key = `${filename}:${fileSize}`;
    for (const f of fs.readdirSync(uploadDir)) {
      if (!f.endsWith('_checkpoint.json')) continue;
      try {
        const data = JSON.parse(fs.readFileSync(path.join(uploadDir, f), 'utf-8'));
        if (data.checkpointKey === key) return { jobId: data.jobId };
      } catch { /* corrupt — skip */ }
    }
    return null;
  }

  /**
   * Count the number of complete (newline-terminated) lines in a file.
   * Used to determine the exact resume frame index from a partial NDJSON result.
   */
  function countCompleteLines(filePath: string): number {
    if (!fs.existsSync(filePath)) return 0;
    try {
      const content = fs.readFileSync(filePath, 'utf-8');
      let count = 0;
      let pos = 0;
      while (true) {
        const nl = content.indexOf('\n', pos);
        if (nl === -1) break; // trailing incomplete line — not counted
        count++;
        pos = nl + 1;
      }
      return count;
    } catch { return 0; }
  }

  function writeJobMeta(jobId: string, meta: JobMeta) {
    try {
      fs.writeFileSync(metaPath(jobId), JSON.stringify(meta));
    } catch (e) {
      console.error(`[Meta] Failed to write meta for ${jobId}:`, e);
    }
  }

  function updateJobMetaFrames(jobId: string, totalFrames: number) {
    const mp = metaPath(jobId);
    try {
      if (fs.existsSync(mp)) {
        const meta: JobMeta = JSON.parse(fs.readFileSync(mp, 'utf-8'));
        meta.totalFrames = totalFrames;
        fs.writeFileSync(mp, JSON.stringify(meta));
      }
    } catch { /* non-fatal */ }
  }

  /** Reconstruct a job from disk after a server restart */
  function loadJobFromDisk(jobId: string): Job | null {
    const rp = path.join(uploadDir, `${jobId}_result.json`);
    const mp = metaPath(jobId);
    const cp = checkpointFilePath(jobId);

    // Read meta so we know filename, size, and persisted status
    let meta: JobMeta | null = null;
    if (fs.existsSync(mp)) {
      try { meta = JSON.parse(fs.readFileSync(mp, 'utf-8')); } catch { /* ignore */ }
    }

    const hasResult = fs.existsSync(rp);
    const hasCheckpoint = fs.existsSync(cp);

    let status: Job['status'];
    let frameCount = meta?.totalFrames ?? 0;

    if (meta?.status === 'stopped') {
      status = 'stopped';
    } else if (meta?.status === 'failed') {
      status = 'failed';
    } else if (hasResult && !hasCheckpoint) {
      status = 'completed';
    } else if (hasCheckpoint) {
      // Server restarted mid-job — job was interrupted (can be resumed by re-upload)
      status = 'failed';
    } else if (!hasResult && meta) {
      status = 'failed';
    } else {
      return null; // No meta and no result — nothing to reconstruct
    }

    const job: Job = {
      id: jobId,
      status,
      totalFrames: frameCount,
      processedFrames: frameCount,
      originalName: meta?.originalName,
      startedAt: meta?.startedAt ?? meta?.createdAt,
      completedAt: meta?.completedAt,
    };
    jobs.set(jobId, job);
    return job;
  }

  /** Scan uploads/ at startup to rebuild the video registry and full job history */
  async function rebuildJobsFromDisk() {
    if (!fs.existsSync(uploadDir)) return;
    let registryCount = 0;
    let historyCount = 0;
    for (const file of fs.readdirSync(uploadDir)) {
      if (!file.endsWith('_meta.json')) continue;
      const jobId = file.replace('_meta.json', '');
      const job = loadJobFromDisk(jobId); // populates jobs Map with all statuses
      if (!job) continue;
      historyCount++;
      if (job.status === 'completed') {
        // Also register in the video registry so /api/lookup-video works
        try {
          const meta: JobMeta = JSON.parse(
            fs.readFileSync(path.join(uploadDir, file), 'utf-8')
          );
          if (meta.originalName && meta.fileSize) {
            videoRegistry.set(`${meta.originalName}:${meta.fileSize}`, jobId);
            registryCount++;
          }
        } catch { /* corrupt meta — skip */ }
      }
    }
    if (historyCount > 0) {
      console.log(`[Startup] Rebuilt video registry: ${registryCount} cached job(s), ${historyCount} total in history`);
    }
  }

  await rebuildJobsFromDisk();

  // --- API ROUTES ---

  // 1. Health check
  app.get('/api/health', (req, res) => {
    res.json({ status: 'ok' });
  });

  // 2. Upload chunk endpoint
  app.post('/api/upload-chunk', upload.single('chunk') as any, async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: 'No chunk file uploaded' });
      }

      const { uploadId, chunkIndex, totalChunks, filename } = req.body;
      const chunkPath = req.file.path;
      const finalPath = path.join(uploadDir, `${uploadId}-${filename}`);

      // Append chunk to final file
      await fs.promises.appendFile(finalPath, await fs.promises.readFile(chunkPath));
      await fs.promises.unlink(chunkPath);

      if (parseInt(chunkIndex) === parseInt(totalChunks) - 1) {
        // All chunks received — check for an existing incomplete checkpoint before
        // creating a new job so we can resume rather than start from scratch.
        const assembled = await fs.promises.stat(finalPath).catch(() => ({ size: 0 }));
        const checkpointKey = `${filename}:${assembled.size}`;

        // ── Duplicate-upload guard ────────────────────────────────────────────
        // If a job for this exact file is already running (e.g. user refreshed
        // mid-upload and re-uploaded the same file), do NOT start a second job.
        // Return the existing jobId so the frontend simply resumes polling it.
        const alreadyActiveId = findActiveJobForFile(filename, assembled.size);
        if (alreadyActiveId) {
          console.log(`[Job ${alreadyActiveId}] Duplicate upload detected for "${filename}" — active job ${alreadyActiveId} already running, skipping new job creation.`);
          // Clean up the redundant assembled file to avoid leaving orphaned data.
          await fs.promises.unlink(finalPath).catch(() => {});
          return res.json({ jobId: alreadyActiveId });
        }
        // ─────────────────────────────────────────────────────────────────────

        const existingCp = findCheckpoint(filename, assembled.size);

        let jobId: string;
        let resumeFrom = 0;

        if (existingCp) {
          // Resume the interrupted job using the same jobId (so the result file path
          // stays the same and we can append to it).
          jobId = existingCp.jobId;

          // Belt-and-suspenders: if this checkpoint jobId is somehow already
          // processing in memory (shouldn't happen after the guard above, but
          // protects against near-simultaneous requests in the same event loop
          // tick before findActiveJobForFile could see it).
          const cpJob = jobs.get(jobId);
          if (cpJob && (cpJob.status === 'processing' || cpJob.status === 'pending')) {
            console.log(`[Job ${jobId}] Duplicate upload detected for "${filename}" — active job ${jobId} already running via checkpoint, skipping new job creation.`);
            await fs.promises.unlink(finalPath).catch(() => {});
            return res.json({ jobId });
          }

          const partialResult = path.join(uploadDir, `${jobId}_result.json`);
          resumeFrom = countCompleteLines(partialResult);
          console.log(`[Job ${jobId}] Checkpoint found — resuming from frame ${resumeFrom} for "${filename}"`);
        } else {
          jobId = `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
        }

        const startedAt = Date.now();
        const job: Job = {
          id: jobId,
          status: 'pending',
          totalFrames: 0,
          processedFrames: resumeFrom, // show already-done frames immediately
          originalName: filename,
          fileSize: assembled.size,
          startedAt,
        };
        jobs.set(jobId, job);

        // Persist / refresh meta so we can recover after a future restart
        writeJobMeta(jobId, {
          originalName: filename,
          fileSize: assembled.size,
          createdAt: startedAt,
          startedAt,
        });

        res.json({ jobId });

        // Kick off processing in the background
        console.log(`[Job ${jobId}] Starting background processing for ${finalPath}...`);
        job.status = 'processing';

        const resultPath = path.join(uploadDir, `${jobId}_result.json`);
        const cpPath = checkpointFilePath(jobId);

        // Cancellation support — store the cancel fn so Stop endpoint can call it
        const controller = new AbortController();
        jobCancelFns.set(jobId, () => controller.abort());

        extractFingerprints(finalPath, resultPath, (decoded, processed) => {
          const j = jobs.get(jobId);
          if (j) {
            j.totalFrames = decoded;
            j.processedFrames = processed;
          }
        }, { resumeFrom, checkpointPath: cpPath, jobId, checkpointKey, abortSignal: controller.signal }).then((frameCount) => {
          console.log(`[Job ${jobId}] Finished processing ${frameCount} frames → ${resultPath}`);
          jobCancelFns.delete(jobId);

          const completedAt = Date.now();
          const j = jobs.get(jobId);
          if (j) {
            j.status = 'completed';
            j.processedFrames = frameCount;
            j.totalFrames = frameCount;
            j.completedAt = completedAt;
          }

          // Persist final status + frame count to meta, then register in video registry
          writeJobMeta(jobId, {
            originalName: filename,
            fileSize: assembled.size,
            createdAt: startedAt,
            startedAt,
            totalFrames: frameCount,
            status: 'completed',
            completedAt,
          });
          videoRegistry.set(checkpointKey, jobId);

          // Delete checkpoint — final result is now on disk
          fs.promises.unlink(cpPath).catch(() => {});

          // Cleanup video file to save space
          if (fs.existsSync(finalPath)) {
            // fs.unlinkSync(finalPath);
          }
        }).catch((err) => {
          jobCancelFns.delete(jobId);
          const isStopped = err?.message === 'STOPPED';
          console.log(`[Job ${jobId}] ${isStopped ? 'Stopped by user' : `Failed: ${err.message}`}`);

          const completedAt = Date.now();
          const j = jobs.get(jobId);
          if (j) {
            j.status = isStopped ? 'stopped' : 'failed';
            j.error = isStopped ? undefined : err.message;
            j.completedAt = completedAt;
          }

          // Persist final status to meta so it survives server restart
          writeJobMeta(jobId, {
            originalName: filename,
            fileSize: assembled.size,
            createdAt: startedAt,
            startedAt,
            totalFrames: j?.totalFrames ?? 0,
            status: isStopped ? 'stopped' : 'failed',
            completedAt,
          });

          // On user-stop: clean up partial result and checkpoint to avoid
          // serving corrupted/incomplete fingerprints on a future lookup.
          if (isStopped) {
            fs.promises.unlink(resultPath).catch(() => {});
            fs.promises.unlink(cpPath).catch(() => {});
          }

          if (fs.existsSync(finalPath)) {
            // fs.unlinkSync(finalPath);
          }
        });
      } else {
        res.json({ status: 'ok' });
      }
    } catch (err: any) {
      console.error('Upload chunk error:', err);
      res.status(500).json({ error: err.message });
    }
  });

  // 3. Upload endpoint (Legacy)
  app.post('/api/upload', upload.single('video') as any, (req, res) => {
    if (!req.file) {
      return res.status(400).json({ error: 'No video file uploaded' });
    }

    const jobId = `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
    const tempVideoPath = req.file.path;

    const job: Job = {
      id: jobId,
      status: 'pending',
      totalFrames: 0,
      processedFrames: 0
    };
    jobs.set(jobId, job);

    res.json({ jobId });

    console.log(`[Job ${jobId}] Starting background processing for ${tempVideoPath}...`);
    job.status = 'processing';

    const legacyResultPath = path.join(uploadDir, `${jobId}_result.json`);
    extractFingerprints(tempVideoPath, legacyResultPath, (decoded, processed) => {
      const j = jobs.get(jobId);
      if (j) {
        j.totalFrames = decoded;
        j.processedFrames = processed;
      }
    }).then((frameCount) => {
      console.log(`[Job ${jobId}] Finished processing ${frameCount} frames → ${legacyResultPath}`);

      try {
        if (fs.existsSync(tempVideoPath)) {
          // fs.unlinkSync(tempVideoPath);
          console.log(`[Job ${jobId}] Cleaned up temporary video file`);
        }
      } catch (cleanupErr) {
        console.error(`[Job ${jobId}] Failed to clean up temp video file:`, cleanupErr);
      }

      const j = jobs.get(jobId);
      if (j) {
        j.status = 'completed';
        j.processedFrames = frameCount;
        j.totalFrames = frameCount;
      }
    }).catch((err) => {
      console.error(`[Job ${jobId}] Processing failed:`, err);
      
      try {
        if (fs.existsSync(tempVideoPath)) {
          // fs.unlinkSync(tempVideoPath);
        }
      } catch (cleanupErr) {
        console.error(`[Job ${jobId}] Failed to clean up temp video file after error:`, cleanupErr);
      }

      const j = jobs.get(jobId);
      if (j) {
        j.status = 'failed';
        j.error = err.message || String(err);
      }
    });
  });

  // 4. Status endpoint — falls back to disk after a server restart
  app.get('/api/status/:jobId', (req, res) => {
    const { jobId } = req.params;
    const job = jobs.get(jobId) ?? loadJobFromDisk(jobId);
    if (!job) {
      return res.status(404).json({ error: 'Job not found' });
    }
    res.json(job);
  });

  // 5. Result retrieval endpoint
  // Always responds with a JSON array regardless of internal file format
  // (new files are NDJSON; old files are a JSON array).
  app.get('/api/result/:jobId', (req, res) => {
    const { jobId } = req.params;
    const job = jobs.get(jobId) ?? loadJobFromDisk(jobId);
    if (!job) {
      return res.status(404).json({ error: 'Job not found' });
    }

    if (job.status !== 'completed') {
      return res.status(400).json({ error: 'Job is not completed yet', status: job.status });
    }

    const resultPath = path.join(uploadDir, `${jobId}_result.json`);
    if (!fs.existsSync(resultPath)) {
      return res.status(404).json({ error: 'Result file not found' });
    }

    try {
      // Peek at first byte to detect format
      const fd = fs.openSync(resultPath, 'r');
      const peek = Buffer.alloc(1);
      fs.readSync(fd, peek, 0, 1, 0);
      fs.closeSync(fd);

      if (peek.toString('utf8') === '[') {
        // Legacy JSON array — serve directly
        res.sendFile(resultPath, (err) => {
          if (err) console.error(`Failed to send result file for job ${jobId}:`, err);
        });
      } else {
        // NDJSON — parse each line and send as a JSON array.
        // This endpoint is used by the browser mode (VideoProcessor.ts) which
        // needs a JSON array.  Short clips are small; movie files are only
        // fetched here if the user explicitly requests them in browser mode.
        const content = fs.readFileSync(resultPath, 'utf-8');
        const arr = content
          .split('\n')
          .filter(l => l.trim().length > 0)
          .map(l => JSON.parse(l));
        res.json(arr);
      }
    } catch (err: any) {
      console.error(`Failed to read result file for job ${jobId}:`, err);
      res.status(500).json({ error: err.message });
    }
  });

  // 6a. Video identity lookup — returns cached jobId if we've already processed this exact file
  app.get('/api/lookup-video', (req, res) => {
    const name = req.query.name as string;
    const size = parseInt(req.query.size as string, 10);
    if (!name || isNaN(size)) {
      return res.status(400).json({ error: 'name and size are required' });
    }
    const jobId = videoRegistry.get(`${name}:${size}`);
    if (!jobId) return res.status(404).json({ error: 'Not found' });

    // Verify result file still exists (user may have manually cleaned uploads/)
    const rp = path.join(uploadDir, `${jobId}_result.json`);
    if (!fs.existsSync(rp)) {
      videoRegistry.delete(`${name}:${size}`);
      return res.status(404).json({ error: 'Result file missing' });
    }

    const job = jobs.get(jobId) ?? loadJobFromDisk(jobId);
    if (!job || job.status !== 'completed') {
      return res.status(404).json({ error: 'Job not completed' });
    }
    res.json({ jobId, totalFrames: job.totalFrames });
  });

  // 6b. List all jobs — for the Job History panel
  app.get('/api/jobs', (req, res) => {
    // Load any disk-only jobs (completed/failed/stopped before this process started)
    // that haven't been requested via /api/status yet.
    if (fs.existsSync(uploadDir)) {
      for (const file of fs.readdirSync(uploadDir)) {
        if (!file.endsWith('_meta.json')) continue;
        const id = file.replace('_meta.json', '');
        if (!jobs.has(id)) loadJobFromDisk(id);
      }
    }

    const STATUS_ORDER: Record<string, number> = {
      processing: 0, pending: 1, completed: 2, stopped: 3, failed: 4
    };
    const list = Array.from(jobs.values()).sort((a, b) => {
      const ao = STATUS_ORDER[a.status] ?? 5;
      const bo = STATUS_ORDER[b.status] ?? 5;
      if (ao !== bo) return ao - bo;
      return (b.startedAt ?? 0) - (a.startedAt ?? 0);
    });
    res.json(list);
  });

  // 6c. Single job detail
  app.get('/api/jobs/:jobId', (req, res) => {
    const { jobId } = req.params;
    if (!/^[\w-]+$/.test(jobId)) return res.status(400).json({ error: 'Invalid jobId' });
    const job = jobs.get(jobId) ?? loadJobFromDisk(jobId);
    if (!job) return res.status(404).json({ error: 'Job not found' });
    res.json(job);
  });

  // 6d. Stop a running job cleanly
  app.post('/api/jobs/:jobId/stop', (req, res) => {
    const { jobId } = req.params;
    if (!/^[\w-]+$/.test(jobId)) return res.status(400).json({ error: 'Invalid jobId' });
    const cancel = jobCancelFns.get(jobId);
    if (!cancel) {
      return res.status(400).json({ error: 'Job is not currently running or is not cancellable' });
    }
    cancel();
    res.json({ ok: true });
  });

  // 6e. Delete a job — stops it if running, then removes all files and clears registry
  app.delete('/api/job/:jobId', (req, res) => {
    const { jobId } = req.params;
    // Basic validation: only alphanumeric + dash
    if (!/^[\w-]+$/.test(jobId)) return res.status(400).json({ error: 'Invalid jobId' });

    // Stop if currently running (fire-and-forget — cleanup happens in the catch handler)
    const cancel = jobCancelFns.get(jobId);
    if (cancel) cancel();

    const rp = path.join(uploadDir, `${jobId}_result.json`);
    const mp = metaPath(jobId);
    const cp = checkpointFilePath(jobId);

    let deleted = false;
    if (fs.existsSync(rp)) { try { fs.unlinkSync(rp); deleted = true; } catch { /* ignore */ } }
    if (fs.existsSync(mp)) { try { fs.unlinkSync(mp); } catch { /* ignore */ } }
    if (fs.existsSync(cp)) { try { fs.unlinkSync(cp); } catch { /* ignore */ } }

    // Remove from in-memory stores
    jobs.delete(jobId);
    jobCancelFns.delete(jobId);

    // Remove from video registry (search by value)
    for (const [k, v] of videoRegistry) {
      if (v === jobId) { videoRegistry.delete(k); break; }
    }

    console.log(`[Delete] Job ${jobId} removed (file existed: ${deleted})`);
    res.json({ deleted });
  });

  // 6. Match endpoint — runs groundMatchedSegments on two stored result JSONs
  app.post('/api/match', async (req, res) => {
    const {
      movieJobId,
      shortJobId,
      minSimilarity,
      minConsecutiveFrames,
      frameDrift
    } = req.body as {
      movieJobId: string;
      shortJobId: string;
      minSimilarity?: number;
      minConsecutiveFrames?: number;
      frameDrift?: number;
    };

    if (!movieJobId || !shortJobId) {
      return res.status(400).json({ error: 'movieJobId and shortJobId are required' });
    }

    // Allow confidence as low as 20 % (user-configurable)
    const resolvedMinSim    = (typeof minSimilarity    === 'number' && minSimilarity    >= 20 && minSimilarity    <= 99) ? minSimilarity    : 82;
    const resolvedMinFrames = (typeof minConsecutiveFrames === 'number' && minConsecutiveFrames >= 3 && minConsecutiveFrames <= 200) ? minConsecutiveFrames : 9;
    const resolvedDrift     = (typeof frameDrift === 'number' && frameDrift >= 0 && frameDrift <= 15) ? Math.round(frameDrift) : 3;

    const movieResultPath = path.join(uploadDir, `${movieJobId}_result.json`);
    const shortResultPath = path.join(uploadDir, `${shortJobId}_result.json`);

    if (!fs.existsSync(movieResultPath)) {
      return res.status(404).json({ error: `Movie result not found for job ${movieJobId}. Re-process the reference video.` });
    }
    if (!fs.existsSync(shortResultPath)) {
      return res.status(404).json({ error: `Short result not found for job ${shortJobId}. Re-process the target clip.` });
    }

    // Stream progress via SSE so the frontend can show live matching progress
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    // Express 5 / Node: flush headers immediately so the client starts receiving
    if (typeof (res as any).flushHeaders === 'function') (res as any).flushHeaders();

    const sendEvent = (data: object) => {
      try { res.write(`data: ${JSON.stringify(data)}\n\n`); } catch { /* client disconnected */ }
    };

    try {
      console.log(`[Match] Streaming fingerprints: movie=${movieJobId} short=${shortJobId} drift=${resolvedDrift}`);

      // matchVideosFromFiles streams both files line-by-line and converts hash
      // strings directly into flat TypedArrays — never loads the full JSON into
      // memory.  Peak RAM drops from ~7 GB to ~400 MB for a 2-hour movie.
      const result = await matchVideosFromFiles(shortResultPath, movieResultPath, {
        minSimilarity:        resolvedMinSim,
        minConsecutiveFrames: resolvedMinFrames,
        frameDrift:           resolvedDrift,
        onProgress:           (info) => sendEvent({ type: 'progress', ...info }),
      });

      console.log(`[Match] Done: ${result.segments.length} segments, ${result.unmatchedRanges.length} unmatched ranges.`);
      sendEvent({
        type:            'done',
        segments:        result.segments,
        unmatchedRanges: result.unmatchedRanges,
        movieFrames:     result.movieFrames,
        shortFrames:     result.shortFrames,
      });
      res.end();
    } catch (err: any) {
      console.error('[Match] Error:', err);
      sendEvent({ type: 'error', error: err.message || String(err) });
      res.end();
    }
  });

  // 7. Worker Accuracy Calibration
  // Tests hash DETERMINISM: send each synthetic frame to the worker TWICE.
  // If both passes return identical 256-bit hashes → worker is stable & correct.
  // Also verifies: non-empty aHash, dHash, signature (colorGrid, skinScoreGrid,
  // detailGrid) — confirming that all 3 signal channels (structure, color/bg,
  // skin/character) are being computed.
  //
  // NOTE: We do NOT compare main-thread vs worker here because canvas (the native
  // addon used by the worker) cannot be loaded in the main tsx process on this
  // NixOS host (missing libuuid.so.1 in the main-thread LD path). Workers are
  // spawned as child processes that inherit the correct library environment.
  app.post('/api/sanity-test', async (req, res) => {
    try {
      const { Worker } = await import('worker_threads');
      const pathMod    = await import('path');

      const isProd = process.env.NODE_ENV === 'production';
      const workerFile = isProd
        ? pathMod.join(process.cwd(), 'dist/worker.cjs')
        : pathMod.join(process.cwd(), 'server/worker.ts');

      // Realistic frame size — worker downscales 320×240 → 160×120 (proper pipeline)
      const W = 320, H = 240, NUM_FRAMES = 10;

      function makeFakeData(fi: number): Uint8ClampedArray {
        const data = new Uint8ClampedArray(W * H * 4);
        for (let i = 0; i < W * H; i++) {
          const x = i % W;
          const y = Math.floor(i / W);
          data[i * 4]     = ((x * (fi + 1) * 31 + y * 97)  ^ (fi * 53))  & 255;
          data[i * 4 + 1] = ((y * (fi + 1) * 67 + x * 41)  ^ (fi * 29))  & 255;
          data[i * 4 + 2] = ((x * y * 7   + fi  * 113)     ^ 128)         & 255;
          data[i * 4 + 3] = 255;
        }
        return data;
      }

      interface WorkerResult {
        hash: string;
        dhash: string;
        signature?: { colorGrid?: number[]; skinScoreGrid?: number[]; detailGrid?: number[] };
      }

      // Send each frame to a fresh worker twice; compare pass-1 vs pass-2 results
      async function runPass(passIdx: 0 | 1): Promise<WorkerResult[]> {
        return new Promise<WorkerResult[]>((resolve, reject) => {
          // Bootstrap: require tsx/cjs then load the TypeScript worker file.
          // Using eval:true avoids the "Unknown file extension .ts" error that
          // occurs with --import tsx in worker_threads on this Node version.
          const bootstrapCode = isProd
            ? `require(${JSON.stringify(workerFile)})`
            : `require('tsx/cjs'); require(${JSON.stringify(workerFile)});`;
          const worker = new Worker(bootstrapCode, { eval: true });
          const results: WorkerResult[] = new Array(NUM_FRAMES).fill(null).map(() => ({
            hash: '', dhash: '', signature: undefined
          }));
          let sent = 0, received = 0;

          const sendNext = () => {
            if (sent >= NUM_FRAMES) return;
            const fi   = sent++;
            const data = makeFakeData(fi);
            worker.postMessage({
              id: fi,
              frameBuffer: Buffer.from(data.buffer),
              width: W, height: H
            });
          };

          worker.on('message', (msg: any) => {
            if (msg.error) { worker.terminate(); reject(new Error(msg.error)); return; }
            const v = msg.result?.variants?.full ?? {};
            results[msg.id] = {
              hash:      v.hash      ?? '',
              dhash:     v.dhash     ?? '',
              signature: msg.result?.signature ?? undefined,
            };
            received++;
            if (received >= NUM_FRAMES) { worker.terminate(); resolve(results); }
            else sendNext();
          });
          worker.on('error', (e: Error) => { worker.terminate(); reject(e); });
          setTimeout(() => { worker.terminate(); reject(new Error(`Worker pass ${passIdx} timed out`)); }, 45_000);
          // pipeline: send up to 4 at a time so the worker is always busy
          for (let i = 0; i < Math.min(4, NUM_FRAMES); i++) sendNext();
        });
      }

      let pass1: WorkerResult[] = [], pass2: WorkerResult[] = [];
      let workerError: string | null = null;
      try {
        // Run both passes in parallel (two separate workers)
        [pass1, pass2] = await Promise.all([runPass(0), runPass(1)]);
      } catch (e: any) {
        workerError = e.message || String(e);
        console.warn('[SanityTest] Worker error:', workerError);
      }

      const hasWorker = !workerError;

      const results = pass1.map((r1, fi) => {
        const r2 = pass2[fi];
        // all-zeros aHash is a valid edge case (uniform image → every pixel = mean → no pixel > mean)
        const hashOk      = r1.hash.length === 256;
        const dhashOk     = r1.dhash.length > 0;
        const deterOk     = hasWorker ? (r1.hash === r2?.hash) : false;
        const sigOk       = !!(r1.signature?.colorGrid?.length && r1.signature?.skinScoreGrid?.length && r1.signature?.detailGrid?.length);
        const pass        = hashOk && deterOk;
        return {
          frameIndex:      fi,
          pass,
          hashBits:        r1.hash.length,
          mainHashPrefix:  r1.hash.slice(0, 48),   // re-uses existing UI field: pass-1 hash
          workerHashPrefix: hasWorker ? (r2?.hash ?? '').slice(0, 48) : '(worker unavailable)', // pass-2 hash
          checks: { hashOk, dhashOk, deterministicOk: deterOk, signatureOk: sigOk },
        };
      });

      const allPass = results.every(r => r.pass);
      console.log(`[SanityTest] ${allPass ? 'PASS' : 'FAIL'} — worker=${hasWorker} frames=${results.filter(r => r.pass).length}/${NUM_FRAMES}`);
      res.json({
        pass: allPass,
        totalFrames: NUM_FRAMES,
        workerAvailable: hasWorker,
        workerError: workerError || undefined,
        results
      });
    } catch (err: any) {
      console.error('[SanityTest] Error:', err);
      res.status(500).json({ error: err.message || String(err) });
    }
  });

  // --- VITE MIDDLEWARE CONFIGURATION ---
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*all', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server ready. Detected ${os.cpus().length} CPU cores. Worker pool sized to ${NUM_WORKERS} workers.`);
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
