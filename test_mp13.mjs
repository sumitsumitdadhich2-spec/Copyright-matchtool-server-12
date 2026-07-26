import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import pkg from 'canvas';
const { Image, createCanvas } = pkg;
import fs from 'fs';

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
    numFaces: 2
  });
  
  const width = 256;
  const height = 256;
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = 'red';
  ctx.fillRect(0, 0, width, height);
  
  // Save as image
  const img = new Image();
  img.src = canvas.toBuffer();
  
  try {
    const result = faceLandmarker.detect(img);
    console.log("Detect result:", result);
  } catch(e) {
    console.error("Error detecting:", e.message);
  }
}
run();
