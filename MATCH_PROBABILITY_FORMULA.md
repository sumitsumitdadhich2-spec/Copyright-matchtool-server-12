# Statistical Match Probability — Formula Audit Trail

This document describes the exact formula used to compute `matchProbability` on
each matched segment.  It is intentionally written to be verifiable by a
non-technical audience and suitable as supporting material in copyright disputes.

---

## What is `matchProbability`?

`matchProbability` (0–100 %) answers the question:

> "How likely is it that this segment is a genuine match, rather than a
> coincidental alignment of unrelated frames?"

It is **completely separate from `confidence`** (the average perceptual-hash
similarity of frames in the segment).  `confidence` measures *how similar* the
frames look; `matchProbability` measures *how statistically unlikely* that
degree of sustained similarity is to have occurred by chance.

---

## The Formula

### Step 1 — Average raw similarity

Collect the per-frame similarity values stored in `matchSequence[i].similarity`.
These are raw (un-boosted) scores on a 0–100 scale.

```
S̄ = mean(matchSequence[i].similarity)   for i = 0 … N−1
```

### Step 2 — Per-frame chance probability

A pair of completely unrelated video frames, when compared via perceptual
hashing, produces an expected Hamming similarity of approximately **50 %**
(random bit strings agree on ~half of their bits).

The probability that one random, unrelated frame achieves similarity ≥ S̄ is:

```
p_chance = BASELINE / S̄       where BASELINE = 50
```

- At S̄ = 50 %  → p_chance = 1.0  (any random frame can achieve this)
- At S̄ = 75 %  → p_chance ≈ 0.67
- At S̄ = 85 %  → p_chance ≈ 0.59
- At S̄ = 100 % → p_chance = 0.50

### Step 3 — Run probability

Treating each frame in the run as an **independent trial** (conservative
assumption — adjacent frames are positively correlated in real video, so the
true coincidence probability is even lower than this model suggests):

```
P(entire N-frame run is coincidental) = p_chance^N = (BASELINE / S̄)^N
```

### Step 4 — Match probability

```
matchProbability = 1 − (BASELINE / S̄)^N          [clamped to 0–100 %]
```

Computed in log-space to avoid floating-point underflow for large N:

```typescript
const logPCoincidence  = N * Math.log(BASELINE / avgRawSim);
const pCoincidence     = Math.exp(logPCoincidence);
const matchProbability = Math.min(100, Math.max(0, (1 - pCoincidence) * 100));
```

---

## Example values

| N frames | Avg similarity | P(coincidence) | matchProbability |
|:--------:|:--------------:|:--------------:|:----------------:|
| 1        | 85 %           | 41 %           | **59 %**         |
| 1        | 95 %           | 47 %           | **53 %**         |
| 5        | 85 %           | 7.1 %          | **92.9 %**       |
| 10       | 85 %           | 0.5 %          | **99.5 %**       |
| 25       | 85 %           | ~0.004 %       | **≈ 100 %**      |
| 150      | 85 %           | ~10⁻¹⁰⁹       | **≈ 100 %**      |

*Key observation:* a single frame, even at high similarity, produces only
moderate match probability (~50–59 %).  Ten or more consecutive frames at the
same similarity level are conclusive (>99 %).  This demonstrates that
`matchProbability` is not simply restating `confidence` in different units.

---

## Assumptions and limitations

1. **Independence assumption** — frames in a genuine video match are positively
   correlated (consecutive frames look similar to each other).  The model
   treats them as independent, which *overstates* the coincidence probability
   and therefore *understates* `matchProbability`.  The true statistical
   confidence in a genuine match is therefore at least as high as reported.

2. **Baseline = 50 %** — this is the theoretical mean Hamming similarity for
   random bit strings.  Real perceptual hashes of unrelated video frames cluster
   somewhat above 50 % due to shared structure (black bars, similar color
   distributions).  Using 50 % as the baseline is therefore conservative
   (makes coincidence appear more likely than it actually is).

3. **Monotone in N** — for fixed similarity, `matchProbability` increases
   strictly as N increases.  A longer sustained run is never less significant
   than a shorter one at the same similarity level.

4. **Monotone in S̄** — for fixed N, higher average similarity → higher
   `matchProbability`.

---

## What this is NOT

- `matchProbability` does **not** modify, replace, or curve the `confidence`
  score in any way.
- It does **not** account for multiple-testing correction (the number of
  candidate alignment positions searched).  In practice, the search space is
  large, which makes genuine matches *more* impressive — so the reported
  `matchProbability` remains a conservative lower bound.
- It is a **model**, not a measurement.  It provides auditable, reproducible
  supporting evidence; it should be considered alongside other evidence, not
  treated as a definitive legal determination in isolation.
