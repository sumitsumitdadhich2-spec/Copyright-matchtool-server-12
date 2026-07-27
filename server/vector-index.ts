/**
 * HNSW vector index for Nexus Video Match.
 *
 * Builds an approximate-nearest-neighbor index over a movie's CLIP embeddings
 * using hnswlib-node (HNSW — functionally equivalent to FAISS ANN for our needs
 * with a pure Node.js binding and no Python/C++ FAISS build toolchain required).
 *
 * Exposes:
 *   buildHnswIndex()        — one-time index build; automatically caches to disk
 *   loadOrBuildHnswIndex()  — load from cache or build (main entry point)
 *   findNearestMovieFrames() — query top-k most similar movie frames
 *   hnswDistToSim100()      — convert HNSW cosine distance to 0-100 sim score
 *
 * Cache file: <movieResultPath>.hnsw.bin  (e.g. movie_result.json.hnsw.bin)
 *
 * Self-hosted / offline: hnswlib-node is a compiled native addon, no external API.
 * Index build cost: ~150–600 ms for 50k–200k frames (logged separately).
 * Query cost: < 1 ms per short-clip frame (logged per-query batch).
 */

import { HierarchicalNSW } from 'hnswlib-node';
import * as fs from 'fs';

// ── Types ─────────────────────────────────────────────────────────────────────

export interface HnswCandidate {
  /** Global movie frame index (zero-based). */
  movieFrameIndex: number;
  /**
   * HNSW cosine distance: 0 = identical direction, 2 = opposite.
   * Convert to 0-100 similarity with hnswDistToSim100().
   */
  distance: number;
}

export interface MovieVectorIndex {
  hnsw:       HierarchicalNSW;
  frameCount: number;
  dim:        number;
}

// ── Distance → similarity conversion ─────────────────────────────────────────

/**
 * Convert HNSW cosine distance to embeddingSim-compatible 0-100 score.
 *
 * HNSW cosine distance = 1 - cos_sim, where cos_sim ∈ [-1, 1].
 * embeddingSim maps cos_sim via: ((cos_sim + 1) / 2) × 100
 * → sim100 = ((2 - dist) / 2) × 100
 */
export function hnswDistToSim100(dist: number): number {
  return Math.max(0, Math.min(100, ((2 - dist) / 2) * 100));
}

// ── Index construction ────────────────────────────────────────────────────────

/**
 * Build an HNSW index from a flat float32 embedding array.
 *
 * Each row of embFlat (stride = embDim) becomes one point whose label equals
 * its frame index (0-based).  The built index is saved to cachePath when
 * the path is non-empty so that subsequent calls can skip the build step.
 *
 * HNSW parameters:
 *   M = 16                good recall/memory trade-off for CLIP-dim vectors
 *   efConstruction = 200  high build accuracy (index quality)
 *
 * @param embFlat    Flat float32 array; row i = frame i embedding
 * @param embDim     Embedding dimension (e.g. 512 for CLIP ViT-B/32)
 * @param cachePath  Where to persist the index; pass '' to skip saving
 */
export async function buildHnswIndex(
  embFlat: Float32Array,
  embDim:  number,
  cachePath: string,
): Promise<MovieVectorIndex> {
  const frameCount = Math.floor(embFlat.length / embDim);
  const M              = 16;
  const efConstruction = 200;

  console.log(
    `[VectorIndex] Building HNSW index: ${frameCount} frames,` +
    ` dim=${embDim}, M=${M}, efConstruction=${efConstruction}…`
  );

  const hnsw = new HierarchicalNSW('cosine', embDim);
  hnsw.initIndex(frameCount, M, efConstruction, 100 /* random seed */);

  const t0 = Date.now();
  for (let i = 0; i < frameCount; i++) {
    const start = i * embDim;
    // hnswlib-node requires a JS number[] for addPoint
    const vec = Array.from(embFlat.subarray(start, start + embDim));
    hnsw.addPoint(vec, i);
  }
  const buildMs = Date.now() - t0;

  console.log(
    `[VectorIndex] HNSW index built in ${buildMs} ms` +
    ` (${(frameCount / (buildMs / 1000)).toFixed(0)} frames/s).`
  );

  if (cachePath) {
    try {
      hnsw.writeIndex(cachePath);
      console.log(`[VectorIndex] Index cached → ${cachePath}`);
    } catch (e) {
      console.warn(
        `[VectorIndex] Could not write index cache: ${(e as Error).message}`
      );
    }
  }

  return { hnsw, frameCount, dim: embDim };
}

// ── Load-or-build ─────────────────────────────────────────────────────────────

/**
 * Load a previously built HNSW index from cachePath, or build + save it.
 *
 * Returns null when embFlat is absent/empty or embDim is invalid — callers
 * must guard against null and fall back to hash-only matching normally.
 */
export async function loadOrBuildHnswIndex(
  embFlat:   Float32Array | null,
  embDim:    number,
  cachePath: string,
): Promise<MovieVectorIndex | null> {
  if (!embFlat || embDim <= 0 || embFlat.length === 0) return null;

  const frameCount = Math.floor(embFlat.length / embDim);
  if (frameCount === 0) return null;

  // ── Try loading from cache ────────────────────────────────────────────────
  if (cachePath && fs.existsSync(cachePath)) {
    try {
      const t0   = Date.now();
      const hnsw = new HierarchicalNSW('cosine', embDim);
      hnsw.readIndex(cachePath);
      const loadMs = Date.now() - t0;
      console.log(
        `[VectorIndex] HNSW index loaded from cache: ${frameCount} frames,` +
        ` dim=${embDim} — ${loadMs} ms.`
      );
      return { hnsw, frameCount, dim: embDim };
    } catch (e) {
      console.warn(
        `[VectorIndex] Cache read failed (${(e as Error).message}), rebuilding…`
      );
      // Fall through to rebuild
    }
  }

  return buildHnswIndex(embFlat, embDim, cachePath);
}

// ── Query ─────────────────────────────────────────────────────────────────────

/**
 * Return the top-k most similar movie frames for a given query embedding.
 *
 * Results are sorted ascending by distance (most similar first).
 * When k > frameCount, k is clamped automatically.
 *
 * @param index           MovieVectorIndex from loadOrBuildHnswIndex
 * @param queryEmbedding  Float32Array of length index.dim (CLIP embedding of a short-clip frame)
 * @param k               Number of nearest neighbours to return
 * @param searchEf        HNSW search ef — higher = more accurate, slower (default 64)
 */
export function findNearestMovieFrames(
  index:          MovieVectorIndex,
  queryEmbedding: Float32Array,
  k:              number,
  searchEf        = 64,
): HnswCandidate[] {
  const effectiveK = Math.min(k, index.frameCount);
  if (effectiveK === 0) return [];

  index.hnsw.setEf(Math.max(searchEf, effectiveK));

  const vec    = Array.from(queryEmbedding);
  const result = index.hnsw.searchKnn(vec, effectiveK);

  const out: HnswCandidate[] = [];
  for (let i = 0; i < result.neighbors.length; i++) {
    out.push({
      movieFrameIndex: result.neighbors[i],
      distance:        result.distances[i],
    });
  }

  // Ensure ascending distance order (most similar first)
  out.sort((a, b) => a.distance - b.distance);
  return out;
}
