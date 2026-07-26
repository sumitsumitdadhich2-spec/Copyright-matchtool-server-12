import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { JSDOM } from 'jsdom';
import pkg from 'canvas';
const { createCanvas } = pkg;

const dom = new JSDOM('<!DOCTYPE html><html><body><canvas id="c"></canvas></body></html>');
global.window = dom.window;
global.document = dom.window.document;
global.HTMLCanvasElement = dom.window.HTMLCanvasElement;
global.HTMLImageElement = dom.window.HTMLImageElement;
global.HTMLVideoElement = dom.window.HTMLVideoElement;
global.ImageData = dom.window.ImageData;

// override getContext
const originalGetContext = global.HTMLCanvasElement.prototype.getContext;
global.HTMLCanvasElement.prototype.getContext = function(type) {
  if (type === 'webgl2' || type === 'webgl') {
     return require('gl')(this.width, this.height);
  }
  return originalGetContext.apply(this, arguments);
};

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
  const imageData = new dom.window.ImageData(data, width, height);
  
  try {
    const result = faceLandmarker.detect(imageData);
    console.log("Detect result:", result);
  } catch (e) {
    console.error("Error detecting:", e);
  }
}
run();
