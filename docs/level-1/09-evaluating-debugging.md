# 09 · Evaluating & Debugging Edge Models

An edge model can pass every notebook test and still fail in the field —
because the deployed system is not just the model: it's the model *plus*
quantization *plus* on-device preprocessing *plus* real sensors, and each
addition is a place for accuracy to leak away. This module turns the
"verify every stage" habit into a systematic method: measuring the right
metrics offline, comparing them on-device, profiling latency and memory,
and diagnosing the three failure causes behind almost every "it worked in
Python" bug report.

## Offline accuracy vs. on-device accuracy

They differ, and each gap has a name and a cause:

```text
Keras float accuracy
      │  (conversion: should be ~zero loss — Module 04)
TFLite float accuracy
      │  (quantization drift: small, measurable — Module 05)
TFLite int8 accuracy       ◄── everything above here measured in Python
      │  (preprocessing mismatch + real-sensor gap)
On-device accuracy         ◄── what the user experiences
```

The professional habit is to *measure every rung* on the same frozen test
set, so any drop is attributed to exactly one transition. The first two
gaps you already know how to measure. The last one is where systems die —
and it splits into (a) your C preprocessing not matching Python, and
(b) real sensor data not matching training data (different mounting,
different user, different microphone).

## Beyond accuracy: the confusion matrix

Overall accuracy hides everything interesting about a classifier. A
wake-word detector that's 95% accurate by *never firing at all* is 95%
useless. The confusion matrix shows per-class behavior:

```python
import numpy as np

def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

cm = confusion_matrix(y_test, preds, 3)
print(cm)
# rows = truth, cols = prediction, e.g. for still/walk/shake:
# [[48  2  0]     still: 2 windows mistaken for walk
#  [ 5 41  4]     walk: the messy class
#  [ 0  3 47]]    shake: solid
```

Read it row by row: row `walk` says 5 walking windows were called *still*
and 4 called *shake*. For edge systems, the off-diagonal cells map directly
to product behavior — **false accepts** (device acts when it shouldn't:
annoying, battery-draining) vs. **false rejects** (device ignores the user:
infuriating). You tune the tradeoff with the decision threshold on the
model's output probability, and the right threshold is a product decision,
not an ML one. Always compute the matrix for the **int8** model — Module 05
showed quantization drift is small *on average*; the matrix shows whether it
concentrated in one class.

## Measuring latency and memory honestly

Numbers you should be able to recite for any model you ship:

```cpp
// Latency: median-of-many, on the real device, real input data
uint32_t times[100];
for (int i = 0; i < 100; i++) {
  uint32_t t0 = micros();
  interpreter->Invoke();
  times[i] = micros() - t0;
}
// report median and max, not just the mean — spikes matter
```

- **Latency**: measure `Invoke()` *and* preprocessing (the FFT can cost
  more than the model!). Median for typical, max for worst case. Compare
  against your real-time budget: a 2 s window with 1 s hop means the whole
  pipeline must finish in under 1 s — comfortably, because the CPU has
  other jobs.
- **RAM**: `interpreter.arena_used_bytes()` after allocation, plus your
  buffers (ring buffer, feature vector). 
- **Flash**: model array length + build-output delta with/without TFLM.
- **Energy** (when it matters): average current × inference time; at
  Level 1 the useful proxy is simply latency, since energy ≈ power × time.

## The big three failure causes

When on-device predictions look wrong, it is almost always one of these —
check them in this order:

**1. Preprocessing mismatch** (most common). Python and C disagree about
normalization, window length, scaling, byte order, or axis order.
*Diagnosis:* capture one raw window on-device, log it over serial, run
both feature extractors on it, diff. *Fix:* the golden-file unit test from
Module 08 — feed frozen raw windows through the C code on desktop and
assert equality with Python's features.

**2. Quantization drift or misuse.** Either genuine int8 accuracy loss
(caught in Python if you evaluated the int8 model — you did, right?), a
representative dataset that didn't match reality (Module 05's sabotage
exercise), or code that writes float values into an int8 tensor without
applying scale/zero-point. *Diagnosis:* compare int8-in-Python vs.
on-device outputs on the same input — they should agree to the last step;
then compare float vs. int8 in Python. The first gap is a code bug, the
second a calibration problem.

**3. Train/field data gap.** The model never saw data like the field's:
sensor mounted at a different angle, a different person's gestures, a
noisier room. No amount of code fixes this. *Diagnosis:* collect a small
labeled dataset *from the deployed device* and evaluate offline — accuracy
collapses on it too. *Fix:* retrain with field data mixed in; on-device
data collection mode is a feature worth building (the capstone does).

!!! tip "Bisect along the pipeline"
    All three diagnoses are the same move: the pipeline is a chain of
    stages you can each run in isolation on a recorded input. Feed the same
    bytes in at both ends of a suspect stage and diff. Never debug by
    staring at live sensor behavior — record once, replay forever.

## A debugging session in miniature

A gesture device performs badly. Bisect:

1. Frozen test set → int8 model in Python: 96%. Model and quantization fine.
2. Serial-log one on-device raw window + its on-device feature vector.
   Python features on the same window: **different**. Found it.
3. Diff element-by-element: features 0, 6, 12 (the means) match; RMS values
   are all ~15× larger on-device. The C code skipped the `/ 9.81`
   gravity normalization — one line.
4. Fix, rerun golden-file test, redeploy: field accuracy matches offline.

Total time with recorded data and stage isolation: minutes. Without: days.

## Cheat sheet

| Question | Tool / method |
|---|---|
| Did conversion change the model? | max abs diff, Keras vs. TFLite float (≈1e-7) |
| Did quantization hurt? | same metric, float vs. int8, same frozen test set |
| Which classes suffer? | confusion matrix on the int8 model |
| False accepts vs. rejects | off-diagonal cells; tune decision threshold |
| Real latency | median + max of 100 timed `Invoke()` + preprocessing |
| Real RAM | `arena_used_bytes()` + app buffers |
| C preprocessing correct? | golden-file test: frozen window → assert features match Python |
| Device vs. Python int8 agree? | same input both places; must match to the last step |
| Field data different? | collect labeled windows from the device, evaluate offline |
| Debug method | record inputs, replay through isolated stages, diff |

## Exercise

1. For your Module 08 activity classifier, compute confusion matrices for
   the float and int8 models on the same test set. Which cells changed?
   Report per-class precision and recall for the int8 model.
2. Pick a decision rule for "shake turns the light on": choose a
   probability threshold and compute false-accept and false-reject rates at
   thresholds 0.5, 0.7, 0.9. Which would you ship for (a) a novelty toy,
   (b) an industrial alarm? Why?
3. Build the golden-file test: save 5 raw windows + Python features to
   files, write a C (or desktop C++) program that loads the windows,
   computes features, and asserts agreement within 1e-5. Break the C
   normalization deliberately and confirm the test catches it.
4. Simulate a field gap: retrain your activity model with the "walk"
   sine at 2 Hz, but generate a test set at 3 Hz (a faster walker).
   Measure the drop, then fix it by augmenting training data with
   1.5–3.5 Hz walks. Report before/after accuracy.
