import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import fs from 'fs';

// Node doesn't have ImageData globally, let's mock it
class ImageData {
  constructor(data, width, height) {
    this.data = data;
    this.width = width;
    this.height = height;
  }
}
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
  const data = new Uint8ClampedArray(width * height * 4);
  data.fill(255);
  const imageData = new ImageData(data, width, height);
  
  try {
    const result = faceLandmarker.detect(imageData);
    console.log("Detect result:", result);
  } catch (e) {
    console.error("Error detecting:", e);
  }
}
run();
