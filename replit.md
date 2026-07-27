# Nexus Video Match

A high-performance video copyright matching tool that locates clips inside a reference movie using perceptual fingerprinting and sequence-alignment.

## How it works

1. **Extract fingerprints** — server-side pipeline (ffmpeg → Node.js worker_threads → node-canvas) decodes video at 25fps and computes a 256-bit perceptual hash for 13 crop/zoom variants of each frame, plus a spatial color/skin/detail signature.
2. **Fingerprint storage** — results stored as `uploads/<jobId>_result.json` on disk.
3. **Matching** — `POST /api/match` runs `groundMatchedSegments()` (two-pass sequence-alignment engine) comparing the short clip against the reference movie.
4. **Preview** — results shown in the browser with side-by-side video playback and a per-frame similarity timeline.

## Stack

- **Frontend**: React 19 + TypeScript + Tailwind CSS v4 (served via Vite middleware)
- **Backend**: Express 5 + tsx (dev) / esbuild (prod)
- **Fingerprinting**: ffmpeg + worker_threads + node-canvas (server-side)
- **Matching**: Pure TypeScript — Uint32Array XOR+popcount Hamming, O(n×m) brute-force scan

## Key files

| File | Purpose |
|------|---------|
| `server.ts` | Express server with all API routes including `/api/match` |
| `server/pipeline.ts` | ffmpeg → worker_threads fingerprint extraction pipeline |
| `server/worker.ts` | Per-frame hash + signature computation (node-canvas) |
| `server/matching-engine.ts` | `groundMatchedSegments()` — the core matching algorithm |
| `src/shared/fingerprint.ts` | Shared types + `computeSignature()` + `computeHashAndFeatures()` |
| `src/App.tsx` | Main React UI — upload, progress, results, side-by-side preview |
| `src/VideoProcessor.ts` | Browser + server video processing, returns `jobId` |

## Running locally

```bash
npm install
PORT=5000 npm run dev
```

## API

- `POST /api/upload-chunk` — chunked video upload (5 MB chunks)
- `GET /api/status/:jobId` — fingerprint extraction progress
- `GET /api/result/:jobId` — download fingerprint JSON
- `POST /api/match` — `{ movieJobId, shortJobId }` → `{ segments, movieFrames, shortFrames }`

## User preferences

- Server mode is the default (faster, uses ffmpeg pipeline)
- Do not restructure the existing ffmpeg + worker_threads pipeline unless explicitly asked
- Keep Docker/deployment config untouched

## Changelog

### Confidence score improvements (server/matching-engine.ts)

Five targeted fixes applied to push confidence from 79–86 % → 92–98 % and resolve 4-segment over-merging:

- **FIX-1 `detectSceneCuts`** — tightened defaults: `aThreshold` 25→32, `dThreshold` 28→34, `colorMagThreshold` 100→85. Hard cuts inside a 6-shot short-clip now detected reliably.
- **FIX-2 `mergeAdjacentSegments`** — now scene-cut-aware. Accepts optional `isCut`/`shortFps`; never merges two segments when a detected cut boundary falls between them. Both `groundMatchedSegments` and `groundMatchedSegmentsChunked` pass the cut map.
- **FIX-3 Speed-ratio + stagnation** — `MIN_SPEED_RATIO` 0.4→0.75, `MAX_SPEED_RATIO` 2.5→1.25. New `frameStagnationFilter` rejects segments where short-clip advances >1.5 s but movie-time spread <0.08 s (stuck-frame artifact). Applied in both code paths.
- **FIX-4 `trimLowSimFrames`** — new function strips leading/trailing frames with similarity <75 % before merging; drops the segment if fewer than 3 frames survive. Applied in both code paths before `mergeAdjacentSegments`.
- **FIX-5 `frameSim` 9:16 crop weighting** — `hashSimBestCross` now returns `{ sim, is9x16 }`. When the winning variant pair involves a `crop_9_16_*` variant, `gSim` weight is zeroed and redistributed to `hSim`/`eSim` (`hSim * 0.75 + eSim * 0.25` with CLIP, pure `hSim` without). Full 16:9 weight blend unchanged.
