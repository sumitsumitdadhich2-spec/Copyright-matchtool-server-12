import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import pkg from 'canvas';
const { createCanvas, ImageData } = pkg;

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
  
  try {
    const result = faceLandmarker.detect(canvas);
    console.log("Detect result:", result);
  } catch (e) {
    console.error("Error detecting:", e);
  }
}
run();
