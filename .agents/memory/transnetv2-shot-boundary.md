---
name: TransNetV2 shot boundary integration
description: How TransNetV2 is wired into the matching pipeline as an additive shot-boundary signal.
---

## Rule
TransNetV2 results are OR'd with `detectSceneCuts` — strictly additive, never removes an existing cut.

## Model
- File: `models/transnetv2.onnx` (~31 MB, MIT licence)
- Source: `elya5/transnetv2` on HuggingFace (community ONNX export of `Sn4kehead/TransNetV2`)
- URL: `https://huggingface.co/elya5/transnetv2/resolve/main/transnetv2.onnx`
- Input: `'input'` tensor `[1, 100, 27, 48, 3]` float32, values in [0, 1] (W=48, H=27)
- Outputs: `'534'` (soft-cut) and `'535'` (hard-cut), both `[1, 100, 1]` float32
- Threshold: 0.5 (standard from the TransNetV2 paper)
- Sliding window: 100-frame window, 50-frame stride, max aggregation over overlap
- Session is a lazy singleton in `server/shot-boundary.ts`

## Integration points
- `server/shot-boundary.ts` — new file; exports `detectShotBoundaries()`, `ensureTransNetV2Model()`, `isShotBoundaryEnabled()`, `loadTransNetV2Session()`
- `server/matching-engine.ts` — imports from `./shot-boundary`; calls `detectShotBoundaries()` in `matchVideosFromFiles()` after `attachEmbeddings()`, passes `transNetV2Cuts` to both `groundMatchedSegments()` and `groundMatchedSegmentsChunked()`
- `Dockerfile` — downloads `transnetv2.onnx` at image build time (same pattern as CLIP model)

## Feature flag
Set `ENABLE_SHOT_BOUNDARY=false` in env to disable; on by default.

## Frame extraction
ffmpeg `scale=48:27` → `rgb24` rawvideo → stdout (spawned async, not execSync).
Memory: 48×27×3 = 3888 bytes/frame; 5-min clip at 25 fps ≈ 29 MB — well within Node buffers.

**Why:** HuggingFace gated models returned 401. The `elya5/transnetv2` model (MIT, 31 MB) was the only accessible ONNX export found via the HF API search. Verified working with test inference.
