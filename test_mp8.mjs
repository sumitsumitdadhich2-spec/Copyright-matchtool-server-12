import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import pkg from 'canvas';
const { createCanvas, Image, ImageData } = pkg;

global.HTMLImageElement = Image;
global.HTMLCanvasElement = createCanvas(1, 1).constructor;
global.ImageData = ImageData;

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
