# Edge Impulse End-to-End Workflows

Edge Impulse is a hosted platform that packages the entire embedded-ML
pipeline — data collection, signal processing, model training, and
on-device deployment — into one guided workflow, with device SDKs that
generate ready-to-flash firmware. Everything Levels 1-2 build by hand
(spectrograms, TFLite conversion, quantization, C arrays) exists inside
Edge Impulse as configurable "blocks" you wire together in a browser UI or
drive from a CLI. This module walks the workflow end-to-end and — because
it's a hosted third-party product this course won't create an account
for — describes each stage in prose alongside the equivalent open-source
building blocks you already know, so you can reproduce or leave the
platform at any point.

## The four-stage pipeline

Every Edge Impulse project follows the same shape, regardless of task
(audio, vision, motion/IMU):

1. **Data acquisition** — samples are uploaded from a browser, a phone app,
   or streamed live from a connected device's SDK (over serial or over
   WiFi/BLE) and labeled in the UI.
2. **Impulse design** — you choose a "processing block" (turns raw signal
   into features: e.g. "MFCC" for audio, "Spectral Analysis" for
   accelerometer data, "Image" for camera frames) and a "learning block"
   (the model architecture: a small CNN, a fully-connected network, or an
   anomaly-detection block using k-means/GMM).
3. **Training** — runs in Edge Impulse's cloud, shows a live confusion
   matrix and per-class accuracy, and — critically — a **"Performance
   calculator"** that estimates RAM, flash, and inference latency *for a
   specific target chip* before you ever flash anything.
4. **Deployment** — exports a C++ library, a pre-built firmware binary for
   supported dev boards (many ESP32, Arduino Nano 33 BLE, STM32 boards are
   directly supported), a WebAssembly build for browser testing, or a raw
   `.tflite`/`.lite` file to integrate yourself.

## Stage 1: data acquisition, and why labeling discipline matters here

Edge Impulse's UI makes it trivially easy to record hundreds of samples —
which means labeling mistakes also multiply trivially fast. The platform
splits data into **training** and **testing** sets explicitly (not a random
per-run split), which is a deliberate design choice: it forces you to decide
up front which recording sessions represent truly held-out conditions
(different day, different person, different room) rather than another
random shuffle of the same session's frames — a distinction Module
04 (Data Collection & Dataset Design) covers in depth.

You can inspect what the acquired data actually contains without the
platform, since Edge Impulse lets you export the raw dataset as CSV/JSON or
WAV files. A useful sanity script once you have an export:

```python
import numpy as np
import json

def summarize_labels(manifest_path):
    """manifest_path: an Edge-Impulse-style JSON export listing samples
    with their labels. Purely illustrative -- run against your own export."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    labels = [s["label"] for s in manifest["samples"]]
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"{u:>15}: {c:4d} samples  ({c / len(labels) * 100:.1f}%)")
    imbalance_ratio = counts.max() / counts.min()
    print(f"max/min class ratio: {imbalance_ratio:.1f}x")
```

An imbalance ratio much above 2-3x is worth fixing (collect more of the
minority class, or use class weights in the learning block) before you spend
cloud training minutes on a lopsided dataset.

## Stage 2: the impulse — features + model as one configurable graph

The "impulse" is Edge Impulse's name for the full signal-processing-plus-
model graph. For a keyword-spotting impulse it looks like:

```
Raw audio window (1s, 16kHz)
  -> MFCC processing block (num_cepstral=13, frame_length=0.02, frame_stride=0.01)
  -> Neural Network (2D CNN) learning block
  -> Output: keyword classes + "unknown" + "noise"
```

This is the exact pipeline built by hand in Module 01 — Edge Impulse's MFCC
block runs the same overlapping-FFT-then-Mel-filterbank math as
`log_mel_spectrogram` there, just exposed as UI sliders. Reproducing an
impulse's processing block locally is a good way to sanity-check the
platform's black-box feature extraction:

```python
def mfcc_config_matches(sample_rate, frame_length_s, frame_stride_s, num_cepstral):
    """Translate Edge Impulse's MFCC block parameters into the frame/hop-in-
    samples numbers used by a hand-rolled implementation, for cross-checking."""
    frame_len_samples = int(sample_rate * frame_length_s)
    hop_len_samples = int(sample_rate * frame_stride_s)
    return {
        "frame_len_samples": frame_len_samples,
        "hop_len_samples": hop_len_samples,
        "n_mfcc": num_cepstral,
    }

print(mfcc_config_matches(16000, 0.02, 0.01, 13))
# {'frame_len_samples': 320, 'hop_len_samples': 160, 'n_mfcc': 13}
```

## Stage 3: the EON Tuner and performance calculator

Edge Impulse's **EON Tuner** automates architecture search across processing
+ learning block combinations, scored against your target chip's actual RAM
and flash limits (not just accuracy) — effectively an automated version of
the tradeoff analysis you do by hand in Module 09 (Memory & Latency
Optimization). The **performance calculator** shown alongside every trained
model reports:

- Estimated inference **latency** (ms) on the selected MCU
- Estimated **peak RAM** (the tensor arena, in the terms of Module 02)
- Estimated **flash usage** (quantized model size)

These numbers come from Edge Impulse's own benchmarking database of real
devices running the compiled model, which is more reliable than a manual
estimate — but the *underlying math* (arena = largest adjacent activations,
flash ≈ quantized parameter bytes) is identical to what you calculated by
hand in Module 02.

## Stage 4: deployment options and what each one gives you

| Export type | What you get | When to use it |
|---|---|---|
| Arduino library (.zip) | C++ source + `.ino` example | Arduino-family boards, fastest path to "blinking LED" parity |
| C++ library | Portable inference source, no framework | Custom firmware, non-Arduino toolchains (ESP-IDF, Zephyr) |
| Pre-built firmware | Flash-and-go binary for a supported dev board | Fastest possible demo, no build step |
| WebAssembly | Runs the impulse in a browser tab | Testing/demoing without any hardware |
| `.tflite` file only | Raw model, no glue code | You already have a firmware pipeline (Levels 1-2 style) and just want the trained weights |

The last row is the important escape hatch: nothing locks you into Edge
Impulse's runtime. You can train there, export the raw `.tflite`, and drop
it into a TFLite Micro C array pipeline exactly like Level 1 Module 06 — the
platform is an accelerator for stages 1-3, not a requirement for stage 4.

## Edge-AI tradeoffs

**Speed of iteration vs. control.** Edge Impulse collapses a week of
plumbing (data pipeline, augmentation, training loop, conversion script)
into an afternoon — at the cost of visibility into exactly what the
processing block computes internally, which matters if you need to debug a
subtle accuracy regression.

**Cloud training vs. reproducibility.** Training happens on Edge Impulse's
infrastructure with a UI-driven config, which is fast but harder to pin into
a versioned, scriptable pipeline than a local `train.py`. Projects that need
strict reproducibility often export the dataset and re-run training locally
once the impulse design is settled.

**EON Tuner search cost vs. manual tuning.** An automated architecture
search across processing/learning block combinations can burn far more
compute (and, on paid tiers, credits) than a domain expert manually trying
2-3 configurations informed by the tradeoffs in Modules 05-09.

**Vendor lock-in of convenience features vs. portability.** The
pre-built-firmware and phone-data-collection conveniences are Edge-Impulse-
specific; the exported `.tflite` and C++ inference code are not — always
keep the "raw model export" path in your back pocket.

## Cheat sheet

| Stage | Edge Impulse concept | Equivalent from this course |
|---|---|---|
| Acquire | data acquisition + train/test split | your own data collection (Module 04) |
| Process | "processing block" (MFCC, Spectral, Image) | hand-rolled feature extraction (Module 01/02) |
| Learn | "learning block" (NN, anomaly detection) | Keras model definition |
| Auto-tune | EON Tuner | manual tradeoff analysis (Module 09) |
| Estimate cost | Performance calculator | `arena_used_bytes()`, flash size math (Module 02) |
| Export | Arduino/C++/firmware/WASM/.tflite | TFLite Micro C array pipeline (Level 1 Module 06) |

## Exercise

1. Implement `summarize_labels` against a small synthetic manifest you
   construct yourself (a Python list of dicts with `label` keys) and confirm
   it flags a 5:1 class imbalance correctly.
2. Sketch (in prose, one paragraph) the impulse graph you would design for a
   person-detection task using a camera: which processing block, which
   learning block, and what the output classes would be.
3. Implement `mfcc_config_matches` and use it to compute the frame/hop
   lengths for a hypothetical Edge Impulse project set to 8 kHz sample rate,
   30 ms frames, and 15 ms stride. Compare against the 16 kHz numbers used
   in Module 01 — how much does the frame count for a 1-second clip change?
4. If you have (or can create) a free Edge Impulse account, walk through
   creating a project from a public dataset, note the estimated RAM/flash
   from the performance calculator for two different target chips, and
   compare against a manual estimate using the method from Module 02.
   (Optional — the rest of this module does not depend on having an
   account.)
