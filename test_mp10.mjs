import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import pkg from 'canvas';
const { Image, createCanvas } = pkg;
import fs from 'fs';

global.document = {
  createElement: (tag) => {
    if (tag === 'canvas') {
      const c = createCanvas(1,1);
      c.getContext = function(type) {
         if (type === 'webgl2') return null; // Force fallback to CPU?
         return Object.getPrototypeOf(this).getContext.call(this, type);
      };
      return c;
    }
    return {};
  }
};

async function run() {
  const vision = await FilesetResolver.forVisionTasks(
    "node_modules/@mediapipe/tasks-vision/wasm"
  );
  const faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "models/face_landmarker.task",
      delegate: "CPU"
    },
    runningMode: "IMAGE",
    numFaces: 2,
  });
  
  const width = 256;
  const height = 256;
  
  try {
    const result = faceLandmarker.detect({
      data: new Uint8ClampedArray(width * height * 4),
      width, height
    });
    console.log("Detect result:", result);
  } catch (e) {
    console.error("Error detecting:", e);
  }
}
run();
