---
name: CLIP embedding integration
description: How CLIP ViT-B/32 semantic matching was wired into the hash pipeline as an additive second-pass signal.
---

## The Rule
CLIP is additive only — never replaces hash matching. Controlled by `ENABLE_CLIP_MATCHING=true`.

**Why:** Hash pipeline is fast and already accurate. CLIP adds recall-rescue for visually similar but perceptually different frames (e.g., colour-graded versions, letterbox crops). Running CLIP on every frame in the brute-force scan would be too slow (~200ms/frame on CPU).

**How to apply:**
- Set `ENABLE_CLIP_MATCHING=true` in env to enable; default is off (pure hash path — zero code change).
- Model: Xenova CLIP ViT-B/32 quantized, `models/clip_vision_quantized.onnx` (~88 MB). Auto-downloaded on first use from HuggingFace. Dockerfile bakes it in.
- Each worker lazy-inits its own ONNX session (~88 MB × 4 workers = ~352 MB RAM overhead).
- Embeddings are dim=768 (CLS token from ViT-B/32), L2-normalised. Stored as flat IEEE 754 float32 in `uploads/{jobId}_result.json.embeddings.bin`.
- `embeddingSim()` in matching-engine.ts returns -1 when either PreSet has no embeddings — `frameSim()` degrades gracefully to pre-CLIP weights automatically.
- Chunked path (very large movies): CLIP unavailable for movie frames (chunks don't carry embeddings). Short clip embeddings on shortPreSet still help. frameSim degrades gracefully.
- Zero vectors in .embeddings.bin mean "no embedding available for this frame" — `embeddingSim()` detects magnitude < 0.01 and returns -1.

## Files changed
- `server/embedding.ts` — new file (model download, CLIP inference, binary I/O helpers)
- `server/worker.ts` — lazy ONNX session init, 224×224 canvas resize, embedding in postMessage result
- `server/pipeline.ts` — collect embeddings from worker results, write .embeddings.bin at job end
- `server/matching-engine.ts` — PreSet.embFlat/embDim, embeddingSim(), frameSim() CLIP branch, attachEmbeddings()
- `Dockerfile` — model download step during build

## frameSim weight schema
With CLIP (hasEmb=true):
  - hash + sig + temporal + CLIP: 0.60 / 0.12 / 0.14 / 0.14
  - hash + sig + CLIP (no tSim):  0.72 / 0.14 / 0.14
  - hash + CLIP only:             0.84 / 0.16

Without CLIP (original, unchanged):
  - hash + sig + temporal: 0.70 / 0.14 / 0.16
  - hash + sig:            0.84 / 0.16
  - hash only:             1.0
