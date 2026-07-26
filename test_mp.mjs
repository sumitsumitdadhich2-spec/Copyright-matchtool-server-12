import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import fs from 'fs';
import { createCanvas } from 'canvas';

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
  
  const canvas = createCanvas(256, 256);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = 'red';
  ctx.fillRect(50, 50, 100, 100);
  
  // MediaPipe tasks require image in a format. 
  // Node canvas has HTMLCanvasElement shim if we use canvas itself, let's see.
  // Actually, MediaPipe tasks-vision in Node usually accepts ImageData object
  const imageData = ctx.getImageData(0, 0, 256, 256);
  
  try {
    const result = faceLandmarker.detect(imageData);
    console.log("Detect result:", result);
  } catch (e) {
    console.error("Error detecting:", e);
  }
}
run();
