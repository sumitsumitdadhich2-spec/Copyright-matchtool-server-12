import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

async function run() {
  const vision = await FilesetResolver.forVisionTasks(
    "node_modules/@mediapipe/tasks-vision/wasm"
  );
  const faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "models/face_landmarker.task"
    },
    runningMode: "IMAGE",
    numFaces: 2,
  });
  
  const width = 256;
  const height = 256;
  // Try RGB array
  const data = new Uint8Array(width * height * 3);
  data.fill(255);
  
  try {
    const result = faceLandmarker.detect({data, width, height});
    console.log("Detect result:", result);
  } catch (e) {
    console.error("Error detecting RGB:", e.message);
  }
}
run();
