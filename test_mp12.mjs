import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

async function run() {
  const vision = await FilesetResolver.forVisionTasks(
    "node_modules/@mediapipe/tasks-vision/wasm"
  );
  
  // mock canvas
  const canvas = {
    getContext: (type) => {
      console.log("getContext called with", type);
      if (type === 'webgl2') {
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
          COLOR_BUFFER_BIT: 6, TRIANGLE_FAN: 7
        };
      }
      return null;
    }
  };
  
  const faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "models/face_landmarker.task",
      delegate: "CPU"
    },
    runningMode: "IMAGE",
    numFaces: 2,
    canvas: canvas
  });
  
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
