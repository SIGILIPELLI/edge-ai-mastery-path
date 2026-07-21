# 02 · The Edge AI Workflow

Every edge AI project — wake word, gesture recognizer, vision sensor —
follows the same four-stage pipeline: **train → convert → quantize →
deploy**. Master this pipeline once with a toy model and you can rerun it
with any model that fits your device. This module maps the pipeline, the
tools used at each stage, and the size/accuracy tradeoffs you'll negotiate
at every step. Modules 03–07 then walk each stage hands-on.

## The pipeline at a glance

```text
┌─────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
│ 1. TRAIN     │ → │ 2. CONVERT    │ → │ 3. QUANTIZE   │ → │ 4. DEPLOY         │
│ Keras/PyTorch│   │ .tflite       │   │ float32→int8  │   │ C array + runtime │
│ on laptop/GPU│   │ flatbuffer    │   │ 4× smaller    │   │ on the MCU        │
└─────────────┘   └──────────────┘   └──────────────┘   └──────────────────┘
   model.keras        model.tflite      model_int8.tflite    model_data.cc
       MBs               ~same             ~4× smaller        lives in flash
```

**Stage 1 — Train.** Normal machine learning, on a normal computer, in
Python. The only edge-specific part is *restraint*: you design the smallest
architecture that solves the problem (a few thousand parameters, not
millions), because everything downstream must fit in kilobytes.

**Stage 2 — Convert.** The training framework's model format (a Keras
`.keras` file) carries Python baggage a microcontroller can't use. The
TFLite converter compiles it into a `.tflite` **flatbuffer** — a compact,
self-contained binary of weights + a static graph of operations that a small
C++ interpreter can execute directly.

**Stage 3 — Quantize.** Weights trained as 32-bit floats are converted to
8-bit integers. The model shrinks ~4× and runs faster on integer-only CPUs,
usually losing only a fraction of a percent of accuracy. On many MCUs this
step is not an optimization but a requirement.

**Stage 4 — Deploy.** The `.tflite` file is embedded in firmware as a C
byte array, and **TFLite-Micro** — an interpreter written in portable C++
with no OS dependencies, no dynamic allocation, ~16 KB core footprint —
executes it on-device inside a pre-allocated "tensor arena."

!!! tip "Verify after every stage"
    The golden rule of this course: after every arrow in that diagram, feed
    the same test inputs to the new artifact and compare outputs against the
    previous stage. Conversion should match training almost exactly;
    quantization should be close; on-device should match your desktop TFLM
    build bit-for-bit. When (not if) something breaks, this tells you
    exactly which stage broke it.

## The tool ecosystem

| Tool | Role in the pipeline | You'll use it in |
|---|---|---|
| **TensorFlow / Keras** | Define and train the model in Python | Module 03 |
| **TFLite / LiteRT** | Converter + flatbuffer format + Python/mobile interpreter | Module 04 |
| *(LiteRT is the new name)* | Google renamed TensorFlow Lite to **LiteRT** in 2024 — same format, same APIs, new branding. Docs use both names. | — |
| **TFLite-Micro (TFLM)** | C++ inference runtime for microcontrollers | Modules 06–07 |
| **`xxd` / Python** | Turn `.tflite` bytes into a C array | Module 06 |
| **Arduino / ESP-IDF** | Firmware toolchain that compiles TFLM for a board | Module 07 |
| **Edge Impulse** | Web platform bundling the whole pipeline (collect → train → deploy) with a GUI | Level 2 |
| **PyTorch → ONNX** | Alternative training route; matters for bigger edge devices | Level 3 |

This course teaches the TensorFlow route because its microcontroller story
(TFLM) is the most mature and the code is fully open. **Edge Impulse**
deserves a special mention as the "no-code/low-code alternative": it hosts
data collection (even from your phone's sensors), trains reasonable models
automatically, and exports ready-to-flash firmware. It's an excellent
product — but using it *without* understanding this pipeline makes every
problem a black-box mystery. Learn the pipeline by hand first; then Edge
Impulse becomes a productivity tool instead of a crutch.

## Sizing intuition: the budget conversation

Every edge project starts with arithmetic like this. Suppose the target is
an ESP32 (520 KB RAM, 4 MB flash, ~240 MHz):

- **Flash budget** — firmware + runtime + *model*. If the app takes 1 MB,
  a model under ~1–2 MB is comfortable. Model bytes ≈ parameter count ×
  bytes per parameter: 50,000 params × 1 byte (int8) ≈ 50 KB. Fine.
- **RAM budget** — the tensor arena must hold the *biggest* intermediate
  activation tensors, not the weights (those stay in flash). A 96×96×8
  conv layer output as int8 ≈ 74 KB. Two of those alive at once ≈ 150 KB.
  Tight but workable.
- **Latency budget** — roughly proportional to multiply-accumulate count.
  An MCU without an accelerator does on the order of 1–50 million MACs per
  inference at interactive speeds. A model with 2M MACs at ~10 inferences/s:
  plausible. 500M MACs: not on this chip.

## The tradeoff triangle

You are always trading between three axes:

```text
        accuracy
          /\
         /  \
        /    \
  size ●──────● latency (& energy)
```

- Bigger model → more accurate, more flash/RAM, slower, more energy.
- Quantization → 4× smaller and faster, small accuracy cost.
- Lower input resolution / fewer features → cheaper everything, less signal.

The craft of TinyML is walking down the size/latency axes and measuring how
slowly you can afford to give up accuracy. Concretely, a wake-word team
might accept dropping from 94% to 92% accuracy to fit the model in 50 KB —
because a model that doesn't fit has 0% accuracy.

## Cheat sheet

| Stage | Input → Output | Tool | Typical command/API |
|---|---|---|---|
| Train | data → `.keras` model | Keras | `model.fit(...)`, `model.save(...)` |
| Convert | `.keras` → `.tflite` | TFLite converter | `tf.lite.TFLiteConverter.from_keras_model(m)` |
| Quantize | float32 → int8 | converter flags | `converter.optimizations`, representative dataset |
| Embed | `.tflite` → C array | `xxd -i` | `xxd -i model.tflite > model_data.cc` |
| Run on-device | C array → predictions | TFLite-Micro | `interpreter.Invoke()` in C++ |
| Verify | any stage → numbers | Python/NumPy | compare outputs stage vs. stage |

## Exercise

1. Draw the four-stage pipeline from memory, labeling the file/artifact that
   crosses each arrow and its rough size for a 50,000-parameter model
   (float32 trained → int8 deployed).
2. Budget check: a candidate model has 400,000 int8 parameters and its
   largest pair of live activation tensors totals 300 KB. Does it fit on
   (a) an ESP32 (520 KB RAM / 4 MB flash), (b) an Arduino Nano 33 BLE
   (256 KB RAM / 1 MB flash)? State which budget fails, if any.
3. In your own words (three sentences max), explain why the "verify after
   every stage" rule saves more debugging time than any other habit in this
   course — what class of bug does it isolate?
