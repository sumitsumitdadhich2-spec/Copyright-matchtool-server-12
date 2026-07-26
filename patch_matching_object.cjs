const fs = require('fs');

let content = fs.readFileSync('server/matching-engine.ts', 'utf8');

// Add import
const importObj = `import { getObjectSignal, compareObjectSignals, initObjectSignal } from './object-signal';`;
content = content.replace("import { getFaceSignal", importObj + "\nimport { getFaceSignal");

// Replace enhanceBorderlineSegments
const oldFunc = `async function enhanceBorderlineSegments(result: MatchResult, shortResultPath: string, movieResultPath: string): Promise<MatchResult> {
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
}`;

const newFunc = `async function enhanceBorderlineSegments(result: MatchResult, shortResultPath: string, movieResultPath: string): Promise<MatchResult> {
  const BORDERLINE_MIN = 55;
  const BORDERLINE_MAX = 75;
  const FACE_SIM_THRESHOLD = 80;
  const OBJECT_SIM_THRESHOLD = 60;
  
  const borderlineSegments = result.segments.filter(s => s.confidence >= BORDERLINE_MIN && s.confidence <= BORDERLINE_MAX);
  if (borderlineSegments.length === 0) return result;

  console.log(\`[Enhancer] Found \${borderlineSegments.length} borderline segments. Preparing to enhance...\`);
  
  await initFaceSignal();
  await initObjectSignal();
  
  const shortVideoPath = await getVideoPathFromResultPath(shortResultPath);
  const movieVideoPath = await getVideoPathFromResultPath(movieResultPath);
  
  if (!shortVideoPath || !movieVideoPath) {
    console.warn('[Enhancer] Could not find original video files for face/object signal enhancement.');
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
    
    let boosted = false;

    if (shortSignal.hasFace && movieSignal.hasFace) {
      const faceSim = compareFaceSignals(shortSignal, movieSignal);
      console.log(\`[Enhancer] Face/Hand similarity: \${faceSim.toFixed(1)}%\`);
      
      if (faceSim > FACE_SIM_THRESHOLD) {
        // Boost confidence to the minimum acceptance threshold (e.g. 82) + a small bonus
        const boost = Math.max(82 - seg.confidence + 1, 0);
        seg.confidence += boost;
        boosted = true;
        console.log(\`[Enhancer] Boosted segment confidence to \${seg.confidence.toFixed(1)}% via face similarity\`);
      }
    } else {
      console.log(\`[Enhancer] No face/hand detected in one or both frames.\`);
    }

    if (!boosted) {
      // Try Object detection as a fallback/additional check
      const shortObjSignal = getObjectSignal(shortImageData);
      const movieObjSignal = getObjectSignal(movieImageData);

      if (shortObjSignal.hasObjects && movieObjSignal.hasObjects) {
        const objSimRes = compareObjectSignals(shortObjSignal, movieObjSignal, shortImageData.width, shortImageData.height);
        console.log(\`[Enhancer] Object similarity: \${objSimRes.score.toFixed(1)}%, shared categories: \${objSimRes.sharedCategories.join(', ')}\`);

        if (objSimRes.score > OBJECT_SIM_THRESHOLD) {
          const boost = Math.max(82 - seg.confidence + 1, 0);
          seg.confidence += boost;
          console.log(\`[ObjectSignal] Borderline match at frame \${frameMatch.shortTime}s confirmed by object similarity (objectSim=\${objSimRes.score.toFixed(1)}%, shared categories: \${objSimRes.sharedCategories.join(', ')}). Boosted confidence to \${seg.confidence.toFixed(1)}%.\`);
        }
      } else {
        console.log(\`[Enhancer] No objects detected in one or both frames.\`);
      }
    }
  }
  
  return result;
}`;

content = content.replace(oldFunc, newFunc);
fs.writeFileSync('server/matching-engine.ts', content);
console.log("Patched server/matching-engine.ts with Object Detector");
