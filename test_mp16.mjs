import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

async function run() {
  const vision = await FilesetResolver.forVisionTasks(
    "node_modules/@mediapipe/tasks-vision/wasm"
  );
  
  const canvas = {
    getContext: (type) => {
      console.log("getContext called with", type);
      if (type === 'webgl2' || type === 'webgl') {
        return {
          getExtension: () => null,
          viewport: () => {},
          activeTexture: () => {},
          bindTexture: () => {},
          texImage2D: () => {},
          clearColor: () => {},
          clear: () => {},
          drawArrays: () => {},
          deleteTexture: () => {},
          createTexture: () => ({}),
          TEXTURE0: 1, TEXTURE1: 2, TEXTURE_2D: 3, RGBA: 4, UNSIGNED_BYTE: 5,
          COLOR_BUFFER_BIT: 6, TRIANGLE_FAN: 7, RED: 8, FLOAT: 9, R32F: 10,
        };
      }
      return null;
    }
  };
  
  console.log("Creating landmarker...");
  const faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "models/face_landmarker.task",
      delegate: "CPU"
    },
    runningMode: "IMAGE",
    numFaces: 2,
    canvas: canvas
  });
  console.log("Landmarker created.");
  
  try {
    console.log("Detecting...");
    const result = faceLandmarker.detect(
       { data: new Uint8ClampedArray(256 * 256 * 4), width: 256, height: 256 }
    );
    console.log("Result:", result);
  } catch(e) {
    console.error("Error detecting:", e.message);
  }
}
run();
