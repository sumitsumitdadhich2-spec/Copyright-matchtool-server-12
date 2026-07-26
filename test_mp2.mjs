import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import fs from 'fs';

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
  
  // Dummy ImageData object
  const width = 256;
  const height = 256;
  const data = new Uint8ClampedArray(width * height * 4);
  data.fill(255);
  const imageData = { data, width, height }; // duck-typed ImageData
  
  try {
    const result = faceLandmarker.detect(imageData);
    console.log("Detect result:", result);
  } catch (e) {
    console.error("Error detecting:", e);
  }
}
run();
