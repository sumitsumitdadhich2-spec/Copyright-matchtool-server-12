# Replit Agent Prompt — Confidence Score 95%+ ke liye 5 Fixes

Mere video-matching tool (`server/matching-engine.ts`) ka confidence score
79%–86% par atka hua hai jabki hona chahiye 92–98%. Maine root-cause analysis
kar liya hai aur poori video 4 bade segments mein merge ho rahi hai jabki
actual mein 11 clean cuts hain. Neeche 5 exact fixes hain — inhe apply karo.
Har fix ke baad `console.log` se pehle/baad ka behavior verify karna, aur
kisi bhi existing working logic (jaise Pass 1/2/3 architecture, DTW alignment,
worker_threads pipeline) ko restructure mat karna — sirf neeche diye gaye
thresholds/logic ko targeted tarike se badalna hai.

---

## Fix 1 — `detectSceneCuts()` ka threshold tight karo (scene-cut detection weak hai)

**File:** `server/matching-engine.ts` — function `detectSceneCuts` (line ~785)

**Problem:** Ek 8.24-second short-clip segment mein 6 alag shots (running →
bottle → cap opening → bloody hands → scared face → throwing bottle) hain,
lekin algorithm inhe 1 hi segment maan raha hai. Isse frame-walk ek shot ke
movie-timestamps ko agle shot par zabardasti drag karta hai, jisse
misaligned-frame match score 75-80% aata hai aur poore segment ka average
confidence gir jaata hai.

**Current params:** `aThreshold = 25, dThreshold = 28, colorMagThreshold = 100`

**Kya karna hai:**
- In teeno thresholds ko tighten karo taaki hard cuts reliably detect ho
  (consecutive frames ke beech similarity ka sudden drop > 25% ho toh turant
  cut mark karo). Real test clips (jinme manually known cut-points hain) par
  tune karke exact values decide karo — start point ke liye `aThreshold` ~30-35,
  `dThreshold` ~32-36, aur `colorMagThreshold` ~80-90 try karo.
- Ensure `splitBySceneCuts()` in tightened cuts ko correctly consume kar raha ho.
- Isko regression test karo: purane false-positive cuts (jinke wajah se
  threshold pehle loosen kiya gaya tha — comment dekho line ~782-784) dobara
  na aayein, isliye A/B test dono directions mein karo.

---

## Fix 2 — `mergeAdjacentSegments()` ko scene-cut-aware banao (over-merging)

**File:** `server/matching-engine.ts` — function `mergeAdjacentSegments` (line ~1381)

**Problem:** 3 alag shots jo movie mein close timestamps (1s-12s ke beech) par
hain, wo 1 segment mein merge ho jaate hain sirf isliye kyunki unka
`shortGap <= SHORT_GAP_MAX (0.52s)` aur `movieGap` proportional lagta hai —
lekin actual mein short-clip ke andar ek hard visual cut tha.

**Kya karna hai:**
- `mergeAdjacentSegments()` ko ek naya parameter/input do: scene-cut
  boundaries (jo Fix 1 ke `detectSceneCuts`/`splitBySceneCuts` se aate hain).
- Rule add karo: **agar do consecutive segments ke beech short-clip mein ek
  detected scene-cut boundary padta hai, toh unhe kabhi merge mat karo** —
  chahe `shortGap` aur `movieGap` dono criteria pass kar rahe hon.
- Ye check `mergeable` boolean ke condition mein ek extra `&& !hasSceneCutBetween(cur, nxt)` 
  jaisा clause add karke implement karo.

---

## Fix 3 — `speedRatioFilterSegments()` ka range tighten karo (frame stagnation / time-locking)

**File:** `server/matching-engine.ts` — line ~140-141 (`MIN_SPEED_RATIO`,
`MAX_SPEED_RATIO`) aur function `speedRatioFilterSegments` (line ~1251)

**Problem:** Ek segment ka `speedRatio: 0.423` hai — jo abnormal hai. Iske
matchSequence mein short-clip time 8.64s se 11.88s tak badhta hai (4 seconds)
lekin movie time 20.88s se 22.40s par hi "locked" reh jaata hai (background
static/smoke ki wajah se matcher ek hi movie-frame par chipak jaata hai).

**Current values:** `MIN_SPEED_RATIO = 0.4`, `MAX_SPEED_RATIO = 2.5`

**Kya karna hai:**
- `MIN_SPEED_RATIO` ko `0.4` se badhaakar **`0.75`** karo, aur
  `MAX_SPEED_RATIO` ko `2.5` se ghataakar **`1.25`** karo — real video edits
  mein speed-ratio isi range mein hoti hai; ismein bahar wale segments
  almost always false-positive/stuck-frame artifacts hote hain.
- Additionally, ek naya check add karo: agar short-clip time continuously
  aage badh raha ho lekin movie-time **1.5 second se zyada** ke liye same
  frame par ruka rahe (frame-stagnation), toh us segment ko turant reject
  karke us window ke liye naya seed dhundo (re-seed), instead of allowing
  it through as one low-confidence segment.
- **Caution:** `MIN_SPEED_RATIO`/`MAX_SPEED_RATIO` global constants hain aur
  shayad kahin aur bhi use ho rahe hon — pehle `grep` karke saari usages
  check karo taaki koi aur pipeline break na ho.

---

## Fix 4 — Low-similarity frames ko hard-trim karo (Pass 2/3 force-matching)

**File:** `server/matching-engine.ts` — Pass 2/Pass 3 looser-threshold logic
aur `isApproximate` flag handling (line ~1111, ~1165, ~1206)

**Problem:** Ek segment ke andar kuch frames (17.40s-17.96s) ka similarity
score 67-72% hai — ye ek Green CGI Monster hai jo original movie mein exist
hi nahi karta. System ne inhe `isApproximate: true` mark karke phir bhi
segment ke andar include kar liya, jo overall confidence ko 79% tak gira
deta hai.

**Kya karna hai:**
- Ek hard cutoff rule add karo: **agar kisi individual frame ki similarity
  score < 75% hai, toh us frame ko segment se trim (kaat) do** — isse
  `isApproximate: true` mark karke silently include mat karo.
- Ye trimming `groundMatchedSegments()` ke andar us jagah lagao jahan
  segment ka final `matchSequence` ban raha ho (line ~1458 ke aas-paas dekho),
  taaki trailing/leading low-score frames drop ho jayein aur segment ka
  `shortEnd`/`shortStart` accordingly adjust ho jaaye.
- Agar trim karne ke baad segment bahut chota (jaise `frameCount < 3`) reh
  jaaye, toh use poora hi drop kar do (already existing filter logic — line
  ~1206 — se consistent rakho, bas threshold ko 75% align karo).

---

## Fix 5 — 9:16 cropped frames ke liye Color Grid Signature disable/normalize karo

**File:** `server/matching-engine.ts` — function `frameSim` ka weight-blend
section (line ~589-608), aur crop-variant detection logic

**Problem:** Ek perfect 100% visual match bhi max 88.60% confidence pe cap ho
raha hai. Current blend: `Hash (60-84%) + Color Signature/gSim (12-16%) +
Motion/tSim (14-16%) + CLIP Embedding/eSim (14-16%)`. Jab short-clip 16:9 se
9:16 vertical crop hoti hai, toh 50% frame area cut jaata hai, isliye
`signatureSim` (Color Grid) artificially low aata hai — chahe visual match
perfect ho.

**Current weight lines (approx):**
```
if (tSim >= 0) return hSim * 0.60 + gSim * 0.12 + tSim * 0.14 + eSim * 0.14;
return hSim * 0.72 + gSim * 0.14 + eSim * 0.14;
if (tSim >= 0) return hSim * 0.70 + gSim * 0.14 + tSim * 0.16;
return hSim * 0.84 + gSim * 0.16;
if (hasEmb) return hSim * 0.84 + eSim * 0.16;
```

**Kya karna hai:**
- `frameSim()` ko pata hona chahiye ki current comparison ek 9:16 crop-variant
  (jaise `crop_9_16_0` — codebase mein 13 crop/zoom variants already exist
  karte hain, unme se identify karo) use kar raha hai ya full 16:9 frame.
- Jab crop-variant 9:16 ho, tab `gSim` (color-signature) ka weight **0 kar
  do ya bahut chhota kar do**, aur us weight ko `hSim` (perceptual hash) aur
  `eSim` (CLIP embedding, agar available hai) mein redistribute kar do.
  Example: agar CLIP available hai to `hSim * 0.75 + eSim * 0.25` (bina
  gSim/tSim ke) 9:16 crop ke case mein.
- Ye pura sirf crop-variant ke liye conditional hona chahiye — full-frame
  (16:9) comparisons ka existing weight-blend bilkul waisa hi rehna chahiye,
  taaki unka accuracy na badle.

---

## General Instructions

1. Sab 5 fixes ek hi PR/change-set mein karo, lekin har fix ko clearly
   comment karo (e.g. `// FIX-1: tightened scene-cut threshold`) taaki
   baad mein individually revert/tune kiya ja sake.
2. Existing test files (`test_mp*.mjs`, `test_e2e_match.mjs`, etc.) use
   karke before/after confidence scores compare karo.
3. Koi bhi naya bug introduce na ho — especially Fix 2 aur Fix 4 pipeline
   ke real-time chunk-processing wale version (`groundMatchedSegmentsChunked`,
   line ~2261) mein bhi equivalent hona chahiye, warna server-mode aur
   chunked-mode ke results mismatch ho jayenge.
4. Fix apply karne ke baad, `replit.md` mein ek changelog note add karo.
5. Doosre kisi module (pipeline.ts, worker.ts, dtw-align.ts, Docker config)
   ko restructure mat karna — sirf `server/matching-engine.ts` ke andar ye
   targeted changes karne hain.

**Expected outcome:** Confidence score 92-98% ke beech aana chahiye, aur
poori video 4 ke bajaye ~11 clean, correctly-separated segments mein
split honi chahiye.
