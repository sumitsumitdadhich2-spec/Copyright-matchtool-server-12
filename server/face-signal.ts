import { FaceLandmarker, HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import path from 'path';





let faceLandmarker: FaceLandmarker | null = null;
let handLandmarker: HandLandmarker | null = null;

/**
 * Initializes the MediaPipe Face and Hand landmarkers.
 * This should be called once at server startup.
 */
export async function initFaceSignal() {
  if (faceLandmarker && handLandmarker) return;

  try {
    console.log('[FaceSignal] Initializing MediaPipe tasks...');
    const wasmPath = path.resolve(process.cwd(), 'node_modules/@mediapipe/tasks-vision/wasm');
    const vision = await FilesetResolver.forVisionTasks(wasmPath);

    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: path.join(__dirname, 'models', 'face_landmarker.task'),
        delegate: 'CPU'
      },
      runningMode: 'IMAGE',
      numFaces: 2,
    });

    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: path.join(__dirname, 'models', 'hand_landmarker.task'),
        delegate: 'CPU'
      },
      runningMode: 'IMAGE',
      numHands: 2,
    });
    console.log('[FaceSignal] MediaPipe tasks initialized successfully.');
  } catch (error) {
    console.error('[FaceSignal] Failed to initialize MediaPipe tasks:', error);
  }
}

export interface FaceSignalResult {
  hasFace: boolean;
  faceLandmarks?: { x: number; y: number; z: number }[][];
  handLandmarks?: { x: number; y: number; z: number }[][];
}

/**
 * Gets face and hand landmarks for a given image.
 * 
 * @param imageData - Raw image data (e.g. from canvas or sharp). For Node.js, 
 * MediaPipe expects an object like { data: Uint8ClampedArray, width, height }.
 */
export function getFaceSignal(imageData: { data: Uint8ClampedArray | Uint8Array, width: number, height: number }): FaceSignalResult {
  if (!faceLandmarker || !handLandmarker) {
    console.warn('[FaceSignal] Landmarkers not initialized.');
    return { hasFace: false };
  }

  try {
    // Note: MediaPipe tasks-vision in Node.js currently has a known limitation where 
    // it attempts to use WebGL2 internally for Image conversion, which causes an error 
    // unless a WebGL canvas polyfill is provided. We wrap this in a try-catch to 
    // safely fallback if the environment doesn't support it.
    const faceResult = faceLandmarker.detect(imageData as any);
    const handResult = handLandmarker.detect(imageData as any);

    const hasFace = faceResult.faceLandmarks && faceResult.faceLandmarks.length > 0;
    
    return {
      hasFace,
      faceLandmarks: hasFace ? faceResult.faceLandmarks : undefined,
      handLandmarks: (handResult.landmarks && handResult.landmarks.length > 0) ? handResult.landmarks : undefined,
    };
  } catch (error) {
    console.warn('[FaceSignal] Detection failed (likely due to Node.js WebGL limitation):', (error as Error).message);
    return { hasFace: false };
  }
}

/**
 * Calculates a similarity score (0-100) between two sets of landmarks using normalized Euclidean distance.
 */
function calculateLandmarkSimilarity(landmarks1?: { x: number; y: number; z: number }[][], landmarks2?: { x: number; y: number; z: number }[][]): number {
  if (!landmarks1 || !landmarks2 || landmarks1.length === 0 || landmarks2.length === 0) {
    return 0;
  }

  // Use the first detected face/hand for comparison
  const pts1 = landmarks1[0];
  const pts2 = landmarks2[0];

  if (pts1.length !== pts2.length) return 0;

  let totalDistance = 0;
  for (let i = 0; i < pts1.length; i++) {
    const dx = pts1[i].x - pts2[i].x;
    const dy = pts1[i].y - pts2[i].y;
    // z is often relative to depth, can include or exclude. Let's include it.
    const dz = pts1[i].z - pts2[i].z;
    totalDistance += Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  const avgDistance = totalDistance / pts1.length;
  // Convert distance to a similarity score 0-100. 
  // Typical normalized distances are small (0 to 1 range since landmarks are normalized by image size).
  // A perfect match is distance 0 (score 100).
  const score = Math.max(0, 100 - (avgDistance * 1000)); // Tune the multiplier as needed
  return score;
}

/**
 * Compares face and hand signals between two frames to return a combined faceSim score (0-100).
 */
export function compareFaceSignals(signal1: FaceSignalResult, signal2: FaceSignalResult): number {
  if (!signal1.hasFace || !signal2.hasFace) return 0;

  const faceSim = calculateLandmarkSimilarity(signal1.faceLandmarks, signal2.faceLandmarks);
  const handSim = calculateLandmarkSimilarity(signal1.handLandmarks, signal2.handLandmarks);

  // If both have hands, average the similarities, otherwise just use face.
  if (signal1.handLandmarks && signal2.handLandmarks) {
    return (faceSim + handSim) / 2;
  }
  
  return faceSim;
}
