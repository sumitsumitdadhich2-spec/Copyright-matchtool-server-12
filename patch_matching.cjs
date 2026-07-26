const fs = require('fs');

const codeToInsert = `
// ---------------------------------------------------------------------------
// Face/Hand Landmark enhancement for borderline matches
// ---------------------------------------------------------------------------
import { exec } from 'child_process';
import { promisify } from 'util';
import * as path from 'path';
import { getFaceSignal, compareFaceSignals, initFaceSignal } from './face-signal';
import { createCanvas, loadImage } from 'canvas';

const execAsync = promisify(exec);

async function getVideoPathFromResultPath(resultPath: string): Promise<string | null> {
  const metaPath = resultPath.replace('_result.json', '_meta.json');
  if (!fs.existsSync(metaPath)) return null;
  try {
    const meta = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
    const jobId = path.basename(resultPath).replace('_result.json', '');
    const dir = path.dirname(resultPath);
    const videoPath = path.join(dir, \`\${jobId}-\${meta.originalName}\`);
    if (fs.existsSync(videoPath)) return videoPath;
    
    // Check legacy temp format just in case
    const files = fs.readdirSync(dir);
    for (const f of files) {
      if (f.startsWith(jobId) && f.endsWith('.mp4')) return path.join(dir, f);
    }
  } catch(e) {
    console.error('Error finding video for result path', resultPath, e);
  }
  return null;
}

async function extractFrame(videoPath: string, timestampSec: number): Promise<{ data: Uint8ClampedArray, width: number, height: number } | null> {
  try {
    const tempImage = \`/tmp/frame_\${Date.now()}_\${Math.random().toString(36).substring(7)}.jpg\`;
    await execAsync(\`ffmpeg -y -ss \${timestampSec} -i "\${videoPath}" -vframes 1 -q:v 2 "\${tempImage}"\`);
    
    const image = await loadImage(tempImage);
    const canvas = createCanvas(image.width, image.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0);
    const imageData = ctx.getImageData(0, 0, image.width, image.height);
    
    fs.unlinkSync(tempImage);
    return imageData;
  } catch (e) {
    console.error('Failed to extract frame at', timestampSec, 'from', videoPath);
    return null;
  }
}

async function enhanceBorderlineSegments(result: MatchResult, shortResultPath: string, movieResultPath: string): Promise<MatchResult> {
  const BORDERLINE_MIN = 55;
  const BORDERLINE_MAX = 75;
  const FACE_SIM_THRESHOLD = 80;
  
  const borderlineSegments = result.segments.filter(s => s.confidence >= BORDERLINE_MIN && s.confidence <= BORDERLINE_MAX);
  if (borderlineSegments.length === 0) return result;

  console.log(\`[Enhancer] Found \${borderlineSegments.length} borderline segments. Preparing to enhance...\`);
  
  await initFaceSignal();
  
  const shortVideoPath = await getVideoPathFromResultPath(shortResultPath);
  const movieVideoPath = await getVideoPathFromResultPath(movieResultPath);
  
  if (!shortVideoPath || !movieVideoPath) {
    console.warn('[Enhancer] Could not find original video files for face signal enhancement.');
    return result;
  }
  
  for (const seg of borderlineSegments) {
    // Pick the middle frame of the segment for comparison
    const midIdx = Math.floor(seg.matchSequence.length / 2);
    const frameMatch = seg.matchSequence[midIdx];
    
    console.log(\`[Enhancer] Checking borderline segment (conf: \${seg.confidence.toFixed(1)}%) at short \${frameMatch.shortTime}s, movie \${frameMatch.movieTime}s\`);
    
    const shortImageData = await extractFrame(shortVideoPath, frameMatch.shortTime);
    const movieImageData = await extractFrame(movieVideoPath, frameMatch.movieTime);
    
    if (!shortImageData || !movieImageData) continue;
    
    const shortSignal = getFaceSignal(shortImageData);
    const movieSignal = getFaceSignal(movieImageData);
    
    if (shortSignal.hasFace && movieSignal.hasFace) {
      const faceSim = compareFaceSignals(shortSignal, movieSignal);
      console.log(\`[Enhancer] Face/Hand similarity: \${faceSim.toFixed(1)}%\`);
      
      if (faceSim > FACE_SIM_THRESHOLD) {
        // Boost confidence to the minimum acceptance threshold (e.g. 82) + a small bonus
        const boost = Math.max(82 - seg.confidence + 1, 0);
        seg.confidence += boost;
        console.log(\`[Enhancer] Boosted segment confidence to \${seg.confidence.toFixed(1)}%\`);
      }
    } else {
      console.log(\`[Enhancer] No face/hand detected in one or both frames.\`);
    }
  }
  
  return result;
}
`;

let content = fs.readFileSync('server/matching-engine.ts', 'utf8');

// We need to change `const result = await groundMatchedSegmentsChunked` to `let result = await groundMatchedSegmentsChunked`
content = content.replace('const result = await groundMatchedSegmentsChunked(', 'let result = await groundMatchedSegmentsChunked(');
content = content.replace(
  'return { ...result, movieFrames, shortFrames };\n  }',
  'result = await enhanceBorderlineSegments(result, shortResultPath, movieResultPath);\n    return { ...result, movieFrames, shortFrames };\n  }'
);

// We need to change `const result = await groundMatchedSegments(` to `let result = await groundMatchedSegments(`
content = content.replace('const result = await groundMatchedSegments(', 'let result = await groundMatchedSegments(');
content = content.replace(
  'onProgress?.({ phase: \'finalizing\', pct: 97 });\n  return { ...result, movieFrames, shortFrames };\n}',
  'result = await enhanceBorderlineSegments(result, shortResultPath, movieResultPath);\n\n  onProgress?.({ phase: \'finalizing\', pct: 97 });\n  return { ...result, movieFrames, shortFrames };\n}\n\n' + codeToInsert
);

fs.writeFileSync('server/matching-engine.ts', content);
console.log("Patched server/matching-engine.ts");
