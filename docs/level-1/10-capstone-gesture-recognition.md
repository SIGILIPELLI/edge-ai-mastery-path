# 10 · Capstone — Gesture Recognition

Everything in Level 1, one project: a **gesture recognizer** that
classifies 2-second accelerometer windows as **circle**, **shake**, or
**still** — the classic "magic wand" task. You'll generate labeled data,
extract features, train a small classifier, quantize it to int8, verify
every stage, and export the C array plus a deployment-ready header — the
complete train → convert → quantize → deploy pipeline, executed end-to-end
on your laptop, with an honest final section on what changes when it meets
real hardware.

## Project layout

Create a folder `gesture-capstone/` and build these files as you go:

```text
gesture-capstone/
├── README.md            # what it is, how to run each step, results table
├── make_data.py         # synthetic accelerometer data → data/gestures.npz
├── features.py          # windowing + feature extraction (shared import)
├── train.py             # trains model, saves gesture_model.keras
├── quantize.py          # → gesture_model_int8.tflite (+ float version)
├── verify.py            # stage-by-stage comparison + confusion matrix
├── export_c.py          # → export/gesture_model_data.cc/.h + params
└── data/, export/       # generated artifacts
```

Write the README last, and honestly: one paragraph of what it does, the
commands to reproduce (`python make_data.py && python train.py && ...`),
and your measured results table.

## Step 1 — `make_data.py`: synthetic gestures

Real projects record data from a device (that's the hardware section
below); we synthesize plausible 100 Hz, 3-axis signals so the whole course
stays laptop-runnable — and so you know the ground truth is clean:

```python
import numpy as np

rng = np.random.default_rng(7)
FS, WIN = 100, 200          # 100 Hz, 2 s windows
N_PER_CLASS = 300

def still():
    return rng.normal(0, 0.05, (WIN, 3))

def shake():
    t = np.arange(WIN) / FS
    f = rng.uniform(6, 10)                       # fast oscillation
    sig = np.stack([np.sin(2*np.pi*f*t + rng.uniform(0, 6.28)),
                    np.sin(2*np.pi*f*t + rng.uniform(0, 6.28)),
                    0.3 * np.sin(2*np.pi*f*t)], axis=1)
    return rng.uniform(1.5, 3.0) * sig + rng.normal(0, 0.2, (WIN, 3))

def circle():
    t = np.arange(WIN) / FS
    f = rng.uniform(0.8, 1.5)                    # slow rotation: x,y in quadrature
    sig = np.stack([np.cos(2*np.pi*f*t), np.sin(2*np.pi*f*t),
                    rng.normal(0, 0.1, WIN)], axis=1)
    return rng.uniform(0.8, 1.5) * sig + rng.normal(0, 0.15, (WIN, 3))

makers = [still, shake, circle]                  # labels 0, 1, 2
X = np.stack([makers[c]() for c in range(3) for _ in range(N_PER_CLASS)])
y = np.repeat([0, 1, 2], N_PER_CLASS)

idx = rng.permutation(len(X))
X, y = X[idx].astype(np.float32), y[idx]
n_tr, n_va = 600, 150
np.savez("data/gestures.npz",
         x_train=X[:n_tr], y_train=y[:n_tr],
         x_val=X[n_tr:n_tr+n_va], y_val=y[n_tr:n_tr+n_va],
         x_test=X[n_tr+n_va:], y_test=y[n_tr+n_va:])
print("saved", X.shape)
```

Note the physics being faked: *shake* is high-frequency and high-energy,
*circle* is low-frequency with x/y in quadrature, *still* is noise. The
frequency and amplitude ranges are randomized per sample so the classes
overlap a little — a dataset with zero difficulty teaches zero.

## Step 2 — `features.py`: one source of truth

The Module 08 features, in an importable module used by *every* other
script (rule zero: one implementation, no drift):

```python
import numpy as np

def extract(win):
    """(200, 3) float32 window -> (18,) feature vector."""
    feats = []
    for ch in range(3):
        s = win[:, ch]
        feats += [s.mean(), s.std(), np.sqrt(np.mean(s**2)),
                  s.min(), s.max(), np.mean(np.abs(np.diff(s)))]
    return np.array(feats, dtype=np.float32)

def extract_batch(wins):
    return np.stack([extract(w) for w in wins])
```

## Step 3 — `train.py`: an 18→16→3 classifier

```python
import numpy as np, tensorflow as tf
from tensorflow import keras
from features import extract_batch

d = np.load("data/gestures.npz")
Xtr, Xva = extract_batch(d["x_train"]), extract_batch(d["x_val"])

model = keras.Sequential([
    keras.layers.Input(shape=(18,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(3, activation="softmax"),
])
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(Xtr, d["y_train"], validation_data=(Xva, d["y_val"]),
          epochs=100, batch_size=32, verbose=0)

Xte = extract_batch(d["x_test"])
_, acc = model.evaluate(Xte, d["y_test"], verbose=0)
print(f"float test accuracy: {acc:.3f}")     # expect ~0.97-1.00
model.save("gesture_model.keras")
```

355 parameters. On features this clean, a bigger model buys nothing —
which is the Level 1 lesson in one line. (An alternative worth trying
afterwards: a small 1-D CNN on the *raw* (200, 3) windows — more params,
no hand-crafted features. The exercise asks you to compare.)

## Step 4 — `quantize.py`: full-int8 with real calibration data

```python
import numpy as np, tensorflow as tf
from tensorflow import keras
from features import extract_batch

model = keras.models.load_model("gesture_model.keras")
d = np.load("data/gestures.npz")
Xtr = extract_batch(d["x_train"])

def rep_data():
    for i in range(200):
        yield [Xtr[i:i+1]]

conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep_data
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8
open("gesture_model_int8.tflite", "wb").write(conv.convert())
```

Calibration uses **actual training feature vectors** — the lesson of
Module 05's sabotage exercise, now standard practice.

## Step 5 — `verify.py`: the rungs of the ladder

Evaluate Keras-float and TFLite-int8 on the identical test set; print both
accuracies and the int8 confusion matrix (Module 09 code). Gate the export
on it:

```python
assert int8_acc >= float_acc - 0.02, "quantization cost too much"
```

Expected: float ≈ int8 within a point, and a confusion matrix whose only
off-diagonal leakage is circle↔shake (the classes that share "movement").
If *still* leaks anywhere, your features or calibration are broken — stop
and bisect, Module 09 style.

## Step 6 — `export_c.py`: the deployable artifact

```python
data = open("gesture_model_int8.tflite", "rb").read()
guard = "GESTURE_MODEL_DATA_H_"
with open("export/gesture_model_data.h", "w") as f:
    f.write(f"#ifndef {guard}\n#define {guard}\n"
            "extern const unsigned char g_gesture_model[];\n"
            "extern const unsigned int g_gesture_model_len;\n"
            f"#endif  // {guard}\n")
with open("export/gesture_model_data.cc", "w") as f:
    f.write('#include "gesture_model_data.h"\n'
            "alignas(16) const unsigned char g_gesture_model[] = {\n  " +
            ",".join(f"0x{b:02x}" for b in data) +
            f"\n}};\nconst unsigned int g_gesture_model_len = {len(data)};\n")
print(f"exported {len(data)} bytes")
```

Also have the script print the input/output `(scale, zero_point)` values
into a comment in the header — the C firmware needs them, and hardcoding
stale ones after retraining is a classic Module 09 "big three" bug.

## The honest hardware section: what changes on a real device

Deploying this to an ESP32 with a real accelerometer (e.g. MPU6050) is
Module 06/07 code plus a ring buffer — and these real-world differences:

- **The data gap is now real.** Your synthetic circles are not a human's
  circles. Expect accuracy to drop hard until you *record real gestures* —
  so the first firmware you write should be a data-collection mode that
  streams labeled raw windows over serial. Retrain on those, keep the
  pipeline identical. This is the single biggest change, and it's a data
  change, not a code change.
- **Preprocessing moves to C.** `extract()` must be reimplemented and
  golden-file-tested against Python (Module 08/09) before you trust
  anything. Units matter: MPU6050 readings arrive in raw LSBs — scale to g
  *identically* to training.
- **Windowing becomes streaming.** A ring buffer holds the last 200
  samples; classify every 100 (50% overlap). Add a majority vote over the
  last 3 predictions to stop flicker.
- **Budgets get checked for real.** This model is tiny (arena ~2–4 KB,
  sub-millisecond inference), so the ESP32 barely notices — but run the
  Module 07 measurements anyway and put the numbers in your README.
- **Wokwi note:** Wokwi simulates the MPU6050, so the full firmware can be
  exercised in the browser — you drag sliders instead of shaking a board,
  which is comically bad for circles but perfect for testing the *still*
  class and the plumbing.

## Cheat sheet — the whole course on one card

| Stage | Artifact | Verify with |
|---|---|---|
| Synthesize/collect | `data/gestures.npz` | class-mean RMS sanity check |
| Features | `features.py` (single source) | golden-file test vs. C later |
| Train | `gesture_model.keras` | held-out test accuracy |
| Quantize | `gesture_model_int8.tflite` | int8 acc within 2 pts of float |
| Inspect | confusion matrix | only circle↔shake leakage |
| Export | `gesture_model_data.cc/.h` | byte length; scale/zp in header |
| Deploy (hardware) | Arduino sketch + ring buffer | on-device vs. Python int8 parity |
| Field truth | recorded real gestures | offline eval on device-collected data |

## Exercise

1. Build the project. Fill the README results table: float accuracy, int8
   accuracy, `.tflite` size, C-array size, and the confusion matrix.
2. Make it harder: add a fourth gesture, **tap** (three sharp spikes over
   ~0.5 s, then quiet). Regenerate, retrain, re-verify. Which class does
   *tap* get confused with, and which single feature would you add to fix
   that?
3. Swap the MLP for a 1-D CNN on raw windows (Conv1D(8,16) →
   MaxPool → Conv1D(16,8) → GlobalAveragePooling → Dense(3)). Compare
   parameters, int8 size, and accuracy against the feature-based model.
   Write two sentences: which would you ship, and why?
4. (With hardware or Wokwi) Implement the streaming firmware:
   data-collection mode + inference mode with majority voting. Record 20
   real windows per class, evaluate the synthetic-trained model on them,
   then retrain with your recordings mixed in. Report both numbers — that
   delta *is* the field gap, measured.

---

**Where next?** [Level 2](../level-2/index.md) takes this pipeline to audio
wake words, camera-based vision on MCUs, and models shrunk with pruning and
distillation. The tools stay the same — the budgets get tighter.
