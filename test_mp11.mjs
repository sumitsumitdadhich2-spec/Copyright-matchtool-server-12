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
  
  // wait, what if we pass a JS object with raw image data?
  // MediaPipe tasks might accept an array of pixels?
  
  try {
    const result = faceLandmarker.detect(
       { data: new Uint8ClampedArray(256 * 256 * 4), width: 256, height: 256 }
    );
    console.log(result);
  } catch(e) {
    console.error(e.message);
  }
}
run();
