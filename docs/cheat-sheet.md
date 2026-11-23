---
title: "Edge AI Cheat Sheet: Quick Reference, Beginner to Advanced"
description: "A free Edge AI cheat sheet: 19 quick-reference tables from every level of the Edge AI Mastery Path, each linked to its full lesson."
---

# Edge AI Cheat Sheet

Quick-reference tables from **19 lessons** of the free Edge AI Mastery Path, in course order from beginner to advanced. Each section links to the full lesson with explanations and runnable examples.

## Level 1 · Entry

### [01 · What Is Edge AI](level-1/01-what-is-edge-ai.md)

| Term | Meaning |
|---|---|
| Edge AI | Running ML inference on/near the device that produces the data |
| TinyML | Edge AI on microcontroller-class hardware (KBs of RAM, mW of power) |
| Inference | Using a trained model to make predictions (what runs at the edge) |
| Training | Fitting a model to data (almost always done off-device) |
| MCU | Microcontroller unit — CPU + RAM + flash on one chip, no OS |
| NPU / accelerator | Dedicated silicon for fast neural-network math |
| Latency argument | No network round trip → guaranteed millisecond responses |
| Privacy argument | Raw data never leaves the device |
| Power argument | Computing a result locally beats transmitting raw data |
| Cost argument | Local inference is free at the margin; API calls are not |

### [02 · The Edge AI Workflow](level-1/02-edge-ai-workflow.md)

| Stage | Input → Output | Tool | Typical command/API |
|---|---|---|---|
| Train | data → `.keras` model | Keras | `model.fit(...)`, `model.save(...)` |
| Convert | `.keras` → `.tflite` | TFLite converter | `tf.lite.TFLiteConverter.from_keras_model(m)` |
| Quantize | float32 → int8 | converter flags | `converter.optimizations`, representative dataset |
| Embed | `.tflite` → C array | `xxd -i` | `xxd -i model.tflite > model_data.cc` |
| Run on-device | C array → predictions | TFLite-Micro | `interpreter.Invoke()` in C++ |
| Verify | any stage → numbers | Python/NumPy | compare outputs stage vs. stage |

### [03 · Training a Tiny Model](level-1/03-training-a-tiny-model.md)

| Task | Code |
|---|---|
| Define tiny MLP | `keras.Sequential([...Dense(16, activation="relu")...])` |
| Count parameters | `model.summary()` / `model.count_params()` |
| Compile for regression | `model.compile(optimizer="adam", loss="mse", metrics=["mae"])` |
| Train with validation | `model.fit(x, y, validation_data=(xv, yv), epochs=N)` |
| Final unbiased score | `model.evaluate(x_test, y_test)` |
| Predict | `model.predict(x.reshape(-1, 1))` |
| Save model | `model.save("sine_model.keras")` |
| Save arrays | `np.savez("f.npz", a=a, b=b)` / `np.load("f.npz")` |
| Reproducibility | `np.random.default_rng(seed)`, fixed splits |

### [04 · Converting to TFLite/LiteRT](level-1/04-converting-to-tflite.md)

| Task | Code |
|---|---|
| Create converter | `tf.lite.TFLiteConverter.from_keras_model(model)` |
| From SavedModel dir | `tf.lite.TFLiteConverter.from_saved_model(path)` |
| Convert | `tflite_bytes = converter.convert()` |
| Save | `open("m.tflite", "wb").write(tflite_bytes)` |
| Load interpreter | `tf.lite.Interpreter(model_path="m.tflite")` |
| Mandatory before use | `interpreter.allocate_tensors()` |
| Inspect I/O | `get_input_details()` / `get_output_details()` |
| Write input | `set_tensor(index, np_array)` (dtype+shape must match) |
| Run | `interpreter.invoke()` |
| Read output | `get_tensor(out_index)` |
| Verify stage | `np.max(np.abs(keras_out - tflite_out))` |

### [05 · Quantization](level-1/05-quantization.md)

| Task | Code |
|---|---|
| Enable quantization | `converter.optimizations = [tf.lite.Optimize.DEFAULT]` |
| Dynamic range (weights only) | the line above, nothing else |
| Full integer | + `converter.representative_dataset = gen` |
| Enforce pure int8 ops | `converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]` |
| int8 I/O tensors | `converter.inference_input_type = tf.int8` (and output) |
| Read scale/zero-point | `input_details[0]["quantization"]` → `(scale, zp)` |
| Quantize a value | `q = round(x / scale) + zero_point` |
| Dequantize a value | `x = (q − zero_point) × scale` |
| Judge success | size ratio + metric delta on the same held-out test set |

### [06 · TFLite-Micro &amp; C Arrays](level-1/06-tflite-micro-c-arrays.md)

| Task | Command / code |
|---|---|
| Model → C array | `xxd -i model.tflite > model_data.cc` |
| Keep model in flash | `alignas(16) const unsigned char g_model[]` |
| Load model | `tflite::GetModel(g_model)` |
| Register ops (N of them) | `MicroMutableOpResolver<N> r; r.AddFullyConnected();` |
| Scratch memory | `alignas(16) uint8_t arena[SIZE];` (static, no malloc) |
| Create interpreter | `MicroInterpreter(model, resolver, arena, SIZE)` |
| Plan memory | `interpreter.AllocateTensors()` — check `kTfLiteOk` |
| Measure real arena need | `interpreter.arena_used_bytes()` |
| I/O tensors | `interpreter.input(0)` / `interpreter.output(0)` |
| Quantization params in C | `tensor->params.scale`, `tensor->params.zero_point` |
| Run | `interpreter.Invoke()` |
| Desktop build | TFLM repo `Makefile`, `hello_world` target |

### [07 · Deploying to a Microcontroller](level-1/07-deploying-to-microcontroller.md)

| Task | How |
|---|---|
| Add ESP32 boards to Arduino | Boards Manager URL → install **esp32** |
| TFLM library for ESP32 | `esp-tflite-micro` (Espressif) or Library Manager TFLM builds |
| Keep model in flash | `alignas(16) const unsigned char g_model[]` |
| Interpreter objects that survive `setup()` | declare them `static` |
| Measure inference latency | `micros()` before/after `Invoke()` |
| Check flash fit | Arduino IDE build output ("Sketch uses …") |
| Right-size arena | `arena_used_bytes()` + ~10% headroom |
| Fatal init error idiom | print, then `while (true) {}` |
| Serial output | `Serial.begin(115200)` + Serial Monitor |
| Browser simulation | wokwi.com ESP32 project + `libraries.txt` |

### [08 · Sensors &amp; Feature Extraction](level-1/08-sensors-feature-extraction.md)

| Concept | Essentials |
|---|---|
| Windowing | fixed `window_len`, overlapping via `hop`; ring buffer on-device |
| Window length | must contain the event (gesture 1–2 s, keyword ~1 s) |
| Mean | orientation / DC level |
| RMS | `sqrt(mean(s**2))` — energy; the workhorse motion feature |
| Std dev | variability around the mean |
| Mean abs diff | jerkiness / smoothness |
| FFT | `np.abs(np.fft.rfft(s))` — magnitude spectrum |
| Usable band | 0 … sample_rate/2 (Nyquist) |
| Coarse spectrum | average spectrum into 8–32 bins |
| Spectrogram | FFTs of sub-frames stacked over time (audio → image) |
| Rule zero | train-time and device preprocessing must match exactly |

### [09 · Evaluating &amp; Debugging Edge Models](level-1/09-evaluating-debugging.md)

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

### [10 · Capstone — Gesture Recognition](level-1/10-capstone-gesture-recognition.md)

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

## Level 2 · Intermediate

### [Audio Wake Words &amp; Keyword Spotting](level-2/01-audio-wake-words.md)

| Concept | Typical value / choice |
|---|---|
| Sample rate | 16 kHz (speech), 8 kHz on very tight budgets |
| Frame / hop length | 25 ms / 10 ms |
| Mel bands | 32–40 |
| Spectrogram shape (1s clip) | ~98 x 40 |
| Model size | 15–25K params, 15–30 KB int8 `.tflite` |
| Classes | keywords + "unknown" + "silence/background" |
| Sliding window | 1.0–1.5 s, re-run every 200–500 ms |
| Debounce | require 2–3 consecutive positive windows |
| Power trick | cheap VAD gate before running the CNN |
| Latency budget | detection within ~300-500 ms of keyword end |

### [Image Classification on Microcontrollers](level-2/02-image-classification-mcu.md)

| Decision | MCU-friendly choice |
|---|---|
| Conv type | depthwise-separable, except first layer |
| Input size | 96x96 or 64x64, grayscale where possible |
| Head | GlobalAveragePooling2D + small Dense |
| Compute cost of 3x3 conv | `9*Cin*Cout` MACs/pixel |
| Compute cost of depthwise-separable | `9*Cin + Cin*Cout` MACs/pixel |
| RAM driver | tensor arena ~ 2 largest adjacent activations |
| Flash driver | quantized weight bytes (~1 byte/param int8) |
| Preprocessing pitfall | resize/color-convert must match training exactly |
| Measure, don't guess | `interpreter.arena_used_bytes()` on real model |

### [Edge Impulse End-to-End Workflows](level-2/03-edge-impulse-workflows.md)

| Stage | Edge Impulse concept | Equivalent from this course |
|---|---|---|
| Acquire | data acquisition + train/test split | your own data collection (Module 04) |
| Process | "processing block" (MFCC, Spectral, Image) | hand-rolled feature extraction (Module 01/02) |
| Learn | "learning block" (NN, anomaly detection) | Keras model definition |
| Auto-tune | EON Tuner | manual tradeoff analysis (Module 09) |
| Estimate cost | Performance calculator | `arena_used_bytes()`, flash size math (Module 02) |
| Export | Arduino/C++/firmware/WASM/.tflite | TFLite Micro C array pipeline (Level 1 Module 06) |

### [Data Collection &amp; Dataset Design](level-2/04-data-collection-datasets.md)

| Practice | Why |
|---|---|
| Split by session/speaker/device | prevents leakage that inflates test accuracy |
| `class_balance_report` | catches skewed class counts before training |
| `condition_coverage` per class | catches missing conditions (e.g. no dim negatives) |
| Collect real hard negatives | prevents the model learning the wrong signal |
| Track augmented vs. real counts separately | augmentation ≠ new information |
| `flag_label_disagreement` | catches ambiguous labeling instructions early |
| Re-check coverage after every collection round | datasets grow unevenly by default |

### [Model Pruning &amp; Knowledge Distillation](level-2/05-pruning-distillation.md)

| Technique | What it removes | On-MCU benefit |
|---|---|---|
| Unstructured magnitude pruning | individual weights | flash only, and only with sparse storage/runtime support |
| Structured channel pruning | entire channels/neurons | flash + compute + RAM, works on any runtime |
| Iterative pruning | -- | recovers more accuracy than one-shot pruning at same sparsity |
| Knowledge distillation | -- | lets a small "from-scratch" architecture reach closer to teacher accuracy |
| `temperature` (distillation) | -- | higher = softer targets = more dark knowledge, typically 2-5 |
| `alpha` (distillation) | -- | weight on soft-teacher loss vs. hard-label loss |
| Order of operations | prune/distill first, quantize last | quantization further shrinks whatever comes out |

### [Quantization-Aware Training](level-2/06-quantization-aware-training.md)

| Concept | Summary |
|---|---|
| PTQ | quantize after training; fast, usually sufficient |
| QAT | fake-quantize during training; recovers accuracy PTQ loses |
| Fake quant node | quantize-then-dequantize in the forward pass |
| Straight-through estimator | backward pass ignores the rounding op, passes gradient through |
| Typical QAT recipe | start from trained float model, fine-tune a few epochs with fake-quant active |
| `tfmot.quantization.keras.quantize_model()` | TF's automatic QAT wrapper (described only — not installed here) |
| Symmetric quant | `zero_point=0`, used for weights |
| Asymmetric quant | nonzero `zero_point`, used for post-ReLU activations |
| Decision rule | try PTQ first; reach for QAT only if PTQ's accuracy loss is unacceptable |

### [ESP32-CAM Vision Projects](level-2/07-esp32-cam-projects.md)

| Concern | Practical answer |
|---|---|
| Programming | external FTDI adapter + `IO0`-`GND` jumper while flashing |
| Confirm PSRAM present | `psramFound()` before requesting large frames |
| Cheapest usable capture | `PIXFORMAT_GRAYSCALE` + `FRAMESIZE_96X96`, `fb_count=1` |
| Frame buffer size (96x96 gray) | 9,216 bytes, no PSRAM needed |
| Match training preprocessing | capture at model's native resolution to skip resize entirely |
| Release frame buffer | always `esp_camera_fb_return(fb)` after use |
| GPIO conflict to remember | GPIO4 = flash LED and an SD-card pin |
| Vision model arena size | tens to ~100+ KB, not the 4 KB sine-model arena |

### [Person Detection (Visual Wake Words)](level-2/08-person-detection.md)

| Concern | Guidance |
|---|---|
| Task framing | binary classification, not detection — presence only |
| Reference architecture | Visual Wake Words: depthwise-separable CNN + GAP, ~250K params |
| Dominant dataset risk | scale variance — cover near AND far persons |
| Hard negatives | person-shaped non-persons (mannequins, photos, dogs) |
| Sampling rate | 1-5fps typical; rely on temporal smoothing, not raw fps |
| Smoothing | EMA + hysteresis (different enter/exit thresholds) |
| Failure to watch for | model keying on frame brightness/blob size, not features |
| When presence detection is the wrong tool | you need counting or localization |

### [Optimizing Memory &amp; Latency on MCUs](level-2/09-memory-latency-optimization.md)

| Metric | How to measure | Typical target for a small MCU |
|---|---|---|
| Flash usage | build tool output ("Sketch uses…") | fits with room for OTA/other app code |
| RAM / arena | `interpreter->arena_used_bytes()` | fits alongside WiFi/app buffers, Level 1 Module 07 |
| Latency | `micros()`/`millis()` around `Invoke()` | matches your sampling-rate budget (Module 08) |
| Compute distribution | MAC-count table per layer, or truncation-delta profiling | identifies which layer to target first |
| Resolution scaling law | arena/compute scale ~ `(new_side/old_side)^2` | first-pass estimate before remeasuring |
| Combine techniques in order | prune/distill -> quantize (QAT or PTQ) last | smallest model gets the final lossy step |

---

Go deeper with the full [Edge AI Mastery Path](index.md) curriculum or test yourself with the [interview questions](interview-questions.md).
