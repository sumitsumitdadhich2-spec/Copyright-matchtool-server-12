import { initObjectSignal, getObjectSignal } from './server/object-signal.ts';
import pkg from 'canvas';
const { createCanvas } = pkg;

async function run() {
  await initObjectSignal();
  const width = 256;
  const height = 256;
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = 'red';
  ctx.fillRect(0, 0, width, height);
  
  const imageData = ctx.getImageData(0, 0, width, height);
  
  try {
    const signal = getObjectSignal(imageData);
    console.log("Detect result:", signal);
  } catch (e) {
    console.error("Error detecting:", e);
  }
}

// polyfill globals needed by mediapipe
import { JSDOM } from 'jsdom';
const dom = new JSDOM('<!DOCTYPE html><html><body><canvas id="c"></canvas></body></html>');
global.window = dom.window;
global.document = dom.window.document;
global.HTMLCanvasElement = dom.window.HTMLCanvasElement;
global.HTMLImageElement = dom.window.HTMLImageElement;
global.HTMLVideoElement = dom.window.HTMLVideoElement;
global.ImageData = dom.window.ImageData;

const originalGetContext = global.HTMLCanvasElement.prototype.getContext;
global.HTMLCanvasElement.prototype.getContext = function(type) {
  if (type === 'webgl2' || type === 'webgl') {
     return require('gl')(this.width, this.height);
  }
  return originalGetContext.apply(this, arguments);
};

run();
