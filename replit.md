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
