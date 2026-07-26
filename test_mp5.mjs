import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { createCanvas, ImageData, HTMLCanvasElement } from 'canvas';
import fs from 'fs';

// Setup fake DOM
global.document = {
  createElement: (tag) => {
    if (tag === 'canvas') return createCanvas(1, 1);
    return {};
  }
};
global.HTMLCanvasElement = HTMLCanvasElement;
global.ImageData = ImageData;

async function run() {
  const vision = await FilesetResolver.forVisionTasks(
    "node_modules/@mediapipe/tasks-vision/wasm"
  );
  console.log("Vision resolver loaded");
  const faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "models/face_landmarker.task",
      delegate: "CPU"
    },
    runningMode: "IMAGE",
    numFaces: 2,
  });
  console.log("Face landmarker loaded");
  
  const width = 256;
  const height = 256;
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = 'red';
  ctx.fillRect(0, 0, width, height);
  const imageData = ctx.getImageData(0, 0, width, height);
  
  try {
    const result = faceLandmarker.detect(imageData);
    console.log("Detect result:", result);
  } catch (e) {
    console.error("Error detecting:", e);
  }
}
run();
