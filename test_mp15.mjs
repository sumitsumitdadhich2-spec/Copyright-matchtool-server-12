import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

async function run() {
  console.log("Starting...");
  const vision = await FilesetResolver.forVisionTasks(
    "node_modules/@mediapipe/tasks-vision/wasm"
  );
  console.log("Vision loaded");
  const faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "models/face_landmarker.task",
      delegate: "CPU"
    },
    runningMode: "IMAGE",
    numFaces: 2,
  });
  console.log("Landmarker loaded");
}
run();
