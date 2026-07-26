import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

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
    const result = faceLandmarker.detect(
       { data: new Float32Array(width * height * 4), width: 256, height: 256 }
    );
    console.log("Result:", result);
  } catch(e) {
    console.error("Error detecting:", e.message);
  }
}
run();
