import { ObjectDetector, FilesetResolver } from '@mediapipe/tasks-vision';
import path from 'path';
import { fileURLToPath } from 'url';

let objectDetector: ObjectDetector | null = null;

export async function initObjectSignal() {
  if (objectDetector) return;

  try {
    console.log('[ObjectSignal] Initializing MediaPipe ObjectDetector...');
    const wasmPath = path.resolve(process.cwd(), 'node_modules/@mediapipe/tasks-vision/wasm');
    const vision = await FilesetResolver.forVisionTasks(wasmPath);

    objectDetector = await ObjectDetector.createFromOptions(vision, {
      baseOptions: {
        // The model is bundled in the build
        modelAssetPath: path.resolve(process.cwd(), 'server', 'models', 'efficientdet_lite0.tflite'),
        delegate: 'CPU'
      },
      runningMode: 'IMAGE',
      scoreThreshold: 0.3,
      maxResults: 10,
    });
    console.log('[ObjectSignal] MediaPipe ObjectDetector initialized successfully.');
  } catch (error) {
    console.error('[ObjectSignal] Failed to initialize MediaPipe ObjectDetector:', error);
  }
}

export interface DetectedObject {
  category: string;
  confidence: number;
  boundingBox: {
    originX: number;
    originY: number;
    width: number;
    height: number;
  };
}

export interface ObjectSignalResult {
  hasObjects: boolean;
  objects: DetectedObject[];
}

export function getObjectSignal(imageData: { data: Uint8ClampedArray | Uint8Array, width: number, height: number }): ObjectSignalResult {
  if (!objectDetector) {
    console.warn('[ObjectSignal] ObjectDetector not initialized.');
    return { hasObjects: false, objects: [] };
  }

  try {
    const result = objectDetector.detect(imageData as any);
    
    const objects: DetectedObject[] = [];
    if (result.detections) {
      for (const detection of result.detections) {
        if (detection.categories && detection.categories.length > 0 && detection.boundingBox) {
          objects.push({
            category: detection.categories[0].categoryName || 'unknown',
            confidence: detection.categories[0].score || 0,
            boundingBox: {
              originX: detection.boundingBox.originX || 0,
              originY: detection.boundingBox.originY || 0,
              width: detection.boundingBox.width || 0,
              height: detection.boundingBox.height || 0
            }
          });
        }
      }
    }

    return {
      hasObjects: objects.length > 0,
      objects
    };
  } catch (error) {
    console.warn('[ObjectSignal] Object detection failed:', (error as Error).message);
    return { hasObjects: false, objects: [] };
  }
}

/**
 * Calculates a similarity score (0-100) between two sets of detected objects.
 * Considers matching categories and their relative bounding box overlap/position.
 */
export function compareObjectSignals(signal1: ObjectSignalResult, signal2: ObjectSignalResult, imageWidth: number, imageHeight: number): { score: number, sharedCategories: string[] } {
  if (!signal1.hasObjects || !signal2.hasObjects) return { score: 0, sharedCategories: [] };
  if (signal1.objects.length === 0 || signal2.objects.length === 0) return { score: 0, sharedCategories: [] };

  const sharedCategories = new Set<string>();
  let totalScore = 0;
  let matchesCount = 0;

  // Clone objects arrays to keep track of matched ones
  const objs2 = [...signal2.objects];

  for (const obj1 of signal1.objects) {
    // Find matching category in objs2
    let bestMatchIdx = -1;
    let bestIoU = 0;

    for (let i = 0; i < objs2.length; i++) {
      const obj2 = objs2[i];
      if (obj1.category === obj2.category) {
        // Calculate Intersection over Union (IoU)
        const xA = Math.max(obj1.boundingBox.originX, obj2.boundingBox.originX);
        const yA = Math.max(obj1.boundingBox.originY, obj2.boundingBox.originY);
        const xB = Math.min(obj1.boundingBox.originX + obj1.boundingBox.width, obj2.boundingBox.originX + obj2.boundingBox.width);
        const yB = Math.min(obj1.boundingBox.originY + obj1.boundingBox.height, obj2.boundingBox.originY + obj2.boundingBox.height);

        const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
        const box1Area = obj1.boundingBox.width * obj1.boundingBox.height;
        const box2Area = obj2.boundingBox.width * obj2.boundingBox.height;
        const iou = interArea / (box1Area + box2Area - interArea);

        if (iou > bestIoU) {
          bestIoU = iou;
          bestMatchIdx = i;
        }
      }
    }

    if (bestMatchIdx !== -1) {
      sharedCategories.add(obj1.category);
      // Base score for category match is 50, plus up to 50 for bounding box overlap (IoU)
      const matchScore = 50 + (bestIoU * 50);
      totalScore += matchScore;
      matchesCount++;
      // Remove matched object so it's not matched twice
      objs2.splice(bestMatchIdx, 1);
    }
  }

  if (matchesCount === 0) return { score: 0, sharedCategories: [] };

  // Calculate average score based on max objects between the two frames
  // This penalizes if one frame has 10 objects and the other has only 1
  const maxObjects = Math.max(signal1.objects.length, signal2.objects.length);
  const normalizedScore = totalScore / maxObjects;

  return { 
    score: normalizedScore, 
    sharedCategories: Array.from(sharedCategories) 
  };
}
