---
title: "Edge AI Interview Questions and Answers (179 Questions)"
description: "179 Edge AI interview questions with answers and explanations, from beginner to advanced, organised by level. Free, from the Edge AI Mastery Path."
---

# Edge AI Interview Questions and Answers

**179 questions** with the correct answer and a short explanation, organised from beginner (Level 1) to advanced (Level 4). Every question comes from a lesson in the free Edge AI Mastery Path, so you can open the lesson to go deeper.

!!! tip "Practise, don't just read"
    Each lesson also ends with an interactive quiz, and the [final quiz](quiz.md) tests all four levels. Try to answer each question below before reading the answer.

## Level 1 · Entry

### [01 · What Is Edge AI](level-1/01-what-is-edge-ai.md)

**Q1.** What is edge AI?

**Answer:** Running ML inference on the device that owns the data rather than in the cloud

Models are usually trained elsewhere.

**Q2.** Which is a benefit of on-device inference?

**Answer:** Low latency, privacy, offline operation and no per-request cloud cost

Constraints are memory, power and compute.

**Q3.** What constrains models on a microcontroller?

**Answer:** Kilobytes of RAM, limited flash and compute, and a power budget

Design small.

**Q4.** Why is an always-on wake word model tiny?

**Answer:** It runs continuously on a battery, so it must be small and low power

Power dominates.

### [02 · The Edge AI Workflow](level-1/02-edge-ai-workflow.md)

**Q5.** What are the four pipeline stages?

**Answer:** Train, convert, quantize, deploy

Repeat for any model.

**Q6.** Where is the model usually trained?

**Answer:** On a workstation or in the cloud

Then shipped to the device.

**Q7.** What does conversion produce?

**Answer:** A .tflite flatbuffer consumed by interpreters

TFLite and TFLM read it.

**Q8.** Why prototype with a toy model first?

**Answer:** To master the whole pipeline cheaply before a real model

The sine model is the hello world.

### [03 · Training a Tiny Model](level-1/03-training-a-tiny-model.md)

**Q9.** What does the sine 'hello world' model learn?

**Answer:** A mapping from x to sin(x) from noisy samples

The official TFLM example.

**Q10.** Why add noise to the training samples?

**Answer:** To make a realistic regression problem and avoid an exact lookup

Generalization.

**Q11.** What does a tiny dense network look like?

**Answer:** A few small dense layers (e.g. 16 units each)

Small by design.

**Q12.** How do you check the trained model?

**Answer:** Compare predictions to true sin(x) on held-out points

Validation.

### [04 · Converting to TFLite/LiteRT](level-1/04-converting-to-tflite.md)

**Q13.** What is a .tflite file?

**Answer:** A FlatBuffer model for TensorFlow Lite runtimes

Compact and zero-copy.

**Q14.** Which tool converts a Keras model?

**Answer:** tf.lite.TFLiteConverter

from_keras_model.

**Q15.** Why inspect the converted model?

**Answer:** To verify input/output shapes, types and ops are what you expect

Use the interpreter's tensor details.

**Q16.** What can go wrong in conversion?

**Answer:** Unsupported ops or unexpected data types

Check supported op list.

### [05 · Quantization](level-1/05-quantization.md)

**Q17.** What does int8 quantization give?

**Answer:** About 4x smaller models and faster integer inference at slight precision cost

float32 is 4 bytes, int8 is 1.

**Q18.** A float range [-1.0, 1.0] maps to int8 with scale and zero-point. What's the scale?

```python
lo,hi=-1.0,1.0
scale=(hi-lo)/255
zp=round(-128-lo/scale)
print(round(scale,5), zp)
```

**Answer:** 0.00784 0

scale = range/255; zero_point = round(qmin - min/scale). Verified by running it.

**Q19.** What's post-training quantization?

**Answer:** Converting a trained float model to int8 using a representative dataset to calibrate ranges

Needs representative data.

**Q20.** Why supply a representative dataset?

**Answer:** So activation ranges are calibrated for realistic inputs

Bad calibration hurts accuracy.

### [06 · TFLite-Micro &amp; C Arrays](level-1/06-tflite-micro-c-arrays.md)

**Q21.** How does a model get onto a microcontroller without a filesystem?

**Answer:** As a C byte array compiled into the firmware

xxd -i.

**Q22.** What is TensorFlow Lite for Microcontrollers (TFLM)?

**Answer:** A stripped-down C++ interpreter for devices with kilobytes of RAM

No dynamic allocation.

**Q23.** What is the tensor arena?

**Answer:** A statically allocated buffer TFLM uses for tensors and scratch

Size it by trial.

**Q24.** Why register only the ops you use (OpResolver)?

**Answer:** To save flash by linking only needed kernels

MicroMutableOpResolver.

### [07 · Deploying to a Microcontroller](level-1/07-deploying-to-microcontroller.md)

**Q25.** Which board is the course's reference?

**Answer:** ESP32 (~520 KB RAM, 4 MB flash, 240 MHz, WiFi)

Cheap and capable.

**Q26.** What's the main difference moving from desktop to the MCU?

**Answer:** Memory and I/O limits, with near-identical inference code

Same bytes.

**Q27.** How do you verify the on-device result?

**Answer:** Compare outputs to the desktop interpreter for the same inputs

Parity testing.

**Q28.** What can fail only on hardware?

**Answer:** Arena too small, stack overflow, timing and sensor quirks

Measure on device.

### [08 · Sensors &amp; Feature Extraction](level-1/08-sensors-feature-extraction.md)

**Q29.** 100 Hz accelerometer, 2-second windows with 50% overlap. How many windows from a 10 s recording?

```python
fs=100; win=2*fs; step=win//2; n=10*fs
print((n-win)//step+1)
```

**Answer:** 9

Windows slide by 1 s: starts at 0..8 s. Verified by running it.

**Q30.** Why extract features from sensor streams?

**Answer:** Models need fixed-size inputs and features compress raw data into informative values

Windowing and statistics.

**Q31.** Which are common time-domain features?

**Answer:** Mean, RMS, variance, min/max and zero crossings

Cheap to compute.

**Q32.** Why must training and deployment preprocessing match exactly?

**Answer:** Any mismatch shifts inputs and degrades accuracy

Same code path.

### [09 · Evaluating &amp; Debugging Edge Models](level-1/09-evaluating-debugging.md)

**Q33.** Why can a model pass notebook tests and fail in the field?

**Answer:** Quantization, on-device preprocessing and real sensors each leak accuracy

Evaluate the deployed system.

**Q34.** What should you evaluate on device?

**Answer:** End-to-end accuracy with real sensor data

Field testing.

**Q35.** What does a confusion matrix reveal?

**Answer:** Which classes are confused, beyond overall accuracy

Targeted fixes.

**Q36.** How do you debug a quantized model?

**Answer:** Compare layer outputs against the float model to find where error enters

Layer-wise.

### [10 · Capstone — Gesture Recognition](level-1/10-capstone-gesture-recognition.md)

**Q37.** What does the gesture capstone classify?

**Answer:** 2-second accelerometer windows as circle, shake or still

Magic wand task.

**Q38.** Why generate labeled data in the project?

**Answer:** To practice the full pipeline reproducibly

Then collect real data.

**Q39.** What's the final verification step?

**Answer:** Check int8 model accuracy and on-device behavior

Close the loop.

**Q40.** What ties Level 1 together?

**Answer:** Data, features, training, quantization and deployment

End to end.

## Level 2 · Intermediate

### [Audio Wake Words &amp; Keyword Spotting](level-2/01-audio-wake-words.md)

**Q41.** What features do wake word models commonly use?

**Answer:** Log-mel spectrograms or MFCCs computed from short audio frames

Compact spectral features.

**Q42.** Why are false accepts and false rejects both costly?

**Answer:** False accepts wake the device (annoying, privacy); rejects frustrate users

Tune threshold.

**Q43.** What reduces always-on power?

**Answer:** A tiny first-stage detector that wakes bigger stages only when needed

Cascade.

**Q44.** How do you handle background noise?

**Answer:** Train with noisy/augmented data and a 'background' class

Dataset matters.

### [Image Classification on Microcontrollers](level-2/02-image-classification-mcu.md)

**Q45.** Why can't a ResNet run on a Cortex-M4 with 256 KB RAM?

**Answer:** Activations and weights exceed memory

Use MobileNet-style small nets.

**Q46.** What do depthwise-separable convolutions do?

**Answer:** Cut parameters and compute versus standard convolutions

MobileNet building block.

**Q47.** How do you account for peak RAM?

**Answer:** Sum the largest simultaneously live activation buffers (plus scratch)

Not total activations.

**Q48.** Why use small input resolutions (e.g. 96x96)?

**Answer:** Memory and latency scale with pixels

Trade accuracy.

### [Edge Impulse End-to-End Workflows](level-2/03-edge-impulse-workflows.md)

**Q49.** What does Edge Impulse provide?

**Answer:** A hosted workflow for data, DSP, training and deployment to devices

Fast prototyping.

**Q50.** What's an 'impulse'?

**Answer:** The pipeline from input window through processing block to learning block

Configurable.

**Q51.** When might you leave a platform and hand-build?

**Answer:** For custom models, unusual targets or full control and cost

Know the trade.

**Q52.** What does the EON compiler do?

**Answer:** Reduces RAM/flash versus a generic interpreter by compiling the model

Optimization.

### [Data Collection &amp; Dataset Design](level-2/04-data-collection-datasets.md)

**Q53.** What causes the gap between test set and real world?

**Answer:** Dataset differences like noise, lighting and devices

Collect representative data.

**Q54.** Why record from the target hardware?

**Answer:** Microphones and sensors differ and shape the data

Match deployment.

**Q55.** What is dataset leakage?

**Answer:** The same subject/session in both train and test, inflating accuracy

Split by subject/session.

**Q56.** How do you balance classes?

**Answer:** Collect more of the minority class and augment, rather than only reweight

Coverage.

### [Model Pruning &amp; Knowledge Distillation](level-2/05-pruning-distillation.md)

**Q57.** What is pruning?

**Answer:** Removing low-importance weights or structures to reduce size or compute

Sparsity.

**Q58.** What's knowledge distillation?

**Answer:** Training a small student to mimic a larger teacher's outputs

Soft targets.

**Q59.** Why unstructured sparsity often doesn't speed up MCU inference?

**Answer:** Kernels don't exploit scattered zeros; structured pruning does

Hardware support.

**Q60.** When prefer distillation?

**Answer:** When a small architecture trained alone underperforms

Student benefits from teacher.

### [Quantization-Aware Training](level-2/06-quantization-aware-training.md)

**Q61.** What is QAT?

**Answer:** Quantization-aware training: simulate quantization during training so weights adapt

Recovers accuracy.

**Q62.** When is QAT needed over PTQ?

**Answer:** When PTQ loses unacceptable accuracy (small nets, wide ranges)

More effort.

**Q63.** What does fake quantization do in training?

**Answer:** Rounds values in the forward pass but keeps float gradients

Straight-through estimator.

**Q64.** What must you still do after QAT?

**Answer:** Convert to a true int8 model and re-verify on device

Validate.

### [ESP32-CAM Vision Projects](level-2/07-esp32-cam-projects.md)

**Q65.** What constraints does the ESP32-CAM have?

**Answer:** Limited RAM (PSRAM helps), modest CPU, and tight power/heat

Cheap board.

**Q66.** Why is the camera frame size a problem?

**Answer:** Frame buffers consume large RAM; downscale or crop

Use grayscale 96x96.

**Q67.** What does PSRAM provide?

**Answer:** Extra external RAM for frame buffers and larger arenas

Slower than internal.

**Q68.** How do you save power on the ESP32-CAM?

**Answer:** Capture on trigger and sleep between frames

Duty cycle.

### [Person Detection (Visual Wake Words)](level-2/08-person-detection.md)

**Q69.** What is person detection as a task?

**Answer:** A binary visual wake word: person or not in the frame

Cheap and widely deployed.

**Q70.** Which model family is used for the standard example?

**Answer:** MobileNet-style small CNNs on 96x96 grayscale

Fits MCUs.

**Q71.** What privacy benefit does on-device detection offer?

**Answer:** Images never leave the device

Send only results.

**Q72.** What's a typical failure in the field?

**Answer:** Lighting and pose differences from the training data

Collect varied data.

### [Optimizing Memory &amp; Latency on MCUs](level-2/09-memory-latency-optimization.md)

**Q73.** What first when optimizing memory/latency?

**Answer:** Profile where RAM and time go, per layer

Measure.

**Q74.** Which reduces peak RAM?

**Answer:** Smaller activations (lower resolution, fewer channels) and in-place ops

Activation memory dominates.

**Q75.** What does CMSIS-NN provide?

**Answer:** Optimized int8 kernels for Arm Cortex-M

Faster than reference.

**Q76.** How do you choose between accuracy and latency?

**Answer:** Set a budget and pick the best model within it

Constraints first.

### [Project — Voice-Controlled Device](level-2/10-project-voice-controlled-device.md)

**Q77.** What does the voice-controlled device capstone combine?

**Answer:** Audio pipeline, keyword model, quantization and an on-device action

Offline commands.

**Q78.** How do you reduce false triggers?

**Answer:** Add a background/unknown class, thresholds and debouncing

Tune on real audio.

**Q79.** Why no cloud connection?

**Answer:** Privacy, latency and offline use

On-device.

**Q80.** What should you measure?

**Answer:** Accuracy, latency, power and false-trigger rate in realistic conditions

End to end.

## Level 3 · Advanced

### [Edge NPUs &amp; Accelerators (Coral Edge TPU)](level-3/01-npus-accelerators-coral.md)

**Q81.** What is an NPU?

**Answer:** A neural processing unit that accelerates tensor operations

Much faster than CPU for supported ops.

**Q82.** Why do Coral Edge TPU models need compiling?

**Answer:** Only supported int8 ops map to the TPU; the compiler partitions the graph

Unsupported ops fall back to CPU.

**Q83.** What's the cost of CPU fallbacks?

**Answer:** Data transfers and slower execution that erode the speedup

Check the compile report.

**Q84.** What input type does the Edge TPU require?

**Answer:** Fully int8-quantized models

Quantize first.

### [Vendor NPU Toolchains (i.MX, Ethos-U)](level-3/02-vendor-npu-toolchains.md)

**Q85.** Why do vendor NPUs need their own toolchains?

**Answer:** Each has its own compiler, supported ops and runtime

eIQ, Vela for Ethos-U.

**Q86.** What does Arm's Vela tool do?

**Answer:** Compiles TFLite models for the Ethos-U NPU

Optimizes memory movement.

**Q87.** What should you check before choosing an NPU?

**Answer:** Supported ops, toolchain maturity, memory and power

Fit the model.

**Q88.** Why benchmark on the real device?

**Answer:** Vendor numbers don't reflect your model and memory setup

Measure.

### [ONNX Runtime for Edge Inference](level-3/03-onnx-runtime.md)

**Q89.** What is ONNX?

**Answer:** An open model interchange format supported by many runtimes

PyTorch exports to it.

**Q90.** Why use ONNX Runtime at the edge?

**Answer:** Run PyTorch-origin models without converting to TFLite, with hardware execution providers

Flexible.

**Q91.** What are execution providers?

**Answer:** Backends that run the graph on specific hardware (CPU, NPU, GPU)

Select per target.

**Q92.** What's a downside of ONNX Runtime on tiny MCUs?

**Answer:** Larger footprint than TFLM

Use for Linux-class edge.

### [Streaming Inference Pipelines](level-3/04-streaming-inference.md)

**Q93.** Why can't a microphone stream be fed as a single input?

**Answer:** It's endless, so you window it with overlap

Streaming inference.

**Q94.** What's the benefit of streaming models with state?

**Answer:** They reuse computation across overlapping windows

Lower latency and power.

**Q95.** Why smooth predictions over time?

**Answer:** Reduce flicker from single noisy windows

Moving average or voting.

**Q96.** What buffer structure is common?

**Answer:** A ring buffer of recent samples

Fixed memory.

### [Object Detection on Edge Devices](level-3/05-object-detection-edge.md)

**Q97.** What does object detection output?

**Answer:** Boxes, class labels and confidence scores

More than classification.

**Q98.** Why is detection costlier on the edge?

**Answer:** Larger models and post-processing like NMS

Use light detectors (SSD-MobileNet, YOLO-nano).

**Q99.** What does NMS do?

**Answer:** Removes overlapping duplicate boxes keeping the best

Threshold on IoU.

**Q100.** How to lower detection compute?

**Answer:** Smaller input size, lighter backbone and frame skipping

Trade accuracy.

### [ML Compiler Stacks (TVM, IREE)](level-3/06-ml-compiler-stacks.md)

**Q101.** What does an ML compiler stack (TVM, TFLite) do?

**Answer:** Optimizes and generates code for a target from a model graph

Graph and operator optimizations.

**Q102.** What's operator fusion?

**Answer:** Combining adjacent ops into one kernel to cut memory traffic

Faster.

**Q103.** Why autotuning?

**Answer:** Search schedules for the best performance on specific hardware

Time-intensive.

**Q104.** What's a trade-off of compilers?

**Answer:** Peak speed versus portability and complexity

Choose pragmatically.

### [Model Security on Edge Devices](level-3/07-model-security.md)

**Q105.** Why protect a deployed model?

**Answer:** It's valuable IP and can be extracted or tampered with

Secure boot and encryption.

**Q106.** What's a model extraction attack?

**Answer:** Querying or dumping the model to reproduce it

Rate limit and protect flash.

**Q107.** What helps against model tampering?

**Answer:** Signed model files verified before loading

Secure update.

**Q108.** What's an adversarial example?

**Answer:** An input perturbed to cause a wrong prediction

Robustness testing.

### [Multi-Model Sensor Fusion](level-3/08-multi-model-sensor-fusion.md)

**Q109.** What is sensor fusion?

**Answer:** Combining multiple sensors' data for better inference

E.g. IMU plus microphone.

**Q110.** What's an architecture for multiple models on one MCU?

**Answer:** A scheduler sharing the arena and time, running models in sequence

Memory is the constraint.

**Q111.** Why synchronize sensor timestamps?

**Answer:** Fusing misaligned data corrupts features

Timebase.

**Q112.** What helps when one sensor fails?

**Answer:** Graceful degradation using remaining sensors

Robustness.

### [Benchmarking with MLPerf Tiny](level-3/09-benchmarking-mlperf-tiny.md)

**Q113.** What is MLPerf Tiny?

**Answer:** A benchmark suite for ultra-low-power ML inference

Standard tasks.

**Q114.** What does benchmarking report?

**Answer:** Latency, energy and accuracy under defined rules

Comparable results.

**Q115.** Why report accuracy with latency?

**Answer:** Fast but inaccurate isn't useful

Pareto trade-off.

**Q116.** What makes a benchmark fair?

**Answer:** Same task, dataset, accuracy target and measurement method

Rules matter.

### [Project: Accelerated Vision Node](level-3/10-project-accelerated-vision-node.md)

**Q117.** What does the accelerated vision node combine?

**Answer:** A camera, an NPU-compiled detector and streaming post-processing

End to end.

**Q118.** How do you validate speedup?

**Answer:** Measure end-to-end FPS and power against the CPU baseline

Real numbers.

**Q119.** What limits real-time performance besides the NPU?

**Answer:** Camera capture, pre/post-processing and memory bandwidth

Profile the pipeline.

**Q120.** What do you log in the field?

**Answer:** Latency, confidence distribution and errors

Telemetry for drift.

## Level 4 · Master

### [Production Edge AI Architecture](level-4/01-production-edge-ai-architecture.md)

**Q121.** What should a production edge AI architecture include?

**Answer:** Model lifecycle, OTA updates, telemetry, security and fallbacks

Operate fleets.

**Q122.** Why separate model from firmware updates?

**Answer:** Models change more often and updating them shouldn't need a full firmware release

Modular.

**Q123.** What's a safe fallback?

**Answer:** A known-good previous model or a safe default behavior

Resilience.

**Q124.** What should be monitored?

**Answer:** Inference health, drift and device status

Observability.

### [OTA Model Updates](level-4/02-ota-model-updates.md)

**Q125.** What's OTA model update best practice?

**Answer:** Signed, versioned updates with staged rollout and rollback

Safety.

**Q126.** Why staged rollouts?

**Answer:** Limit blast radius if a model misbehaves

Canary devices first.

**Q127.** How do you ensure update integrity?

**Answer:** Verify signatures and checksums before activating

Secure boot.

**Q128.** What if an update fails?

**Answer:** Automatic rollback to the previous version

A/B slots.

### [Fleet Monitoring &amp; Drift Detection](level-4/03-fleet-monitoring-drift.md)

**Q129.** What is data drift?

**Answer:** The input distribution changes over time, degrading the model

Monitor.

**Q130.** How can you detect drift on constrained devices?

**Answer:** Track simple statistics of inputs/outputs (e.g. confidence, feature means) and report them

Lightweight telemetry.

**Q131.** What do you do when drift is detected?

**Answer:** Collect samples, retrain and roll out an update

Close the loop.

**Q132.** Why privacy-aware telemetry?

**Answer:** Raw data may be sensitive; send aggregates or features

Minimize.

### [On-Device Learning](level-4/04-on-device-learning.md)

**Q133.** What is on-device learning?

**Answer:** Adapting a model locally with the device's own data

Personalization.

**Q134.** Why is it hard on MCUs?

**Answer:** Training needs more memory and compute than inference

Limited to last layers or small models.

**Q135.** What's a practical approach?

**Answer:** Fine-tune only the final layer or use few-shot embeddings

Lightweight.

**Q136.** What risk does on-device learning add?

**Answer:** Drift or poisoning from bad local data without validation

Guard with checks.

### [Federated Learning](level-4/05-federated-learning.md)

**Q137.** What is federated learning?

**Answer:** Training across devices by sharing model updates, not raw data

Privacy-friendly.

**Q138.** What does the server do in FedAvg?

**Answer:** Average client model updates into a new global model

Repeat rounds.

**Q139.** What are federated learning's challenges?

**Answer:** Communication cost, non-IID data and device heterogeneity

Real constraints.

**Q140.** Does federated learning guarantee privacy?

**Answer:** No: updates can leak information; add secure aggregation/differential privacy

Defense in depth.

### [Custom Kernels with CMSIS-NN](level-4/06-custom-kernels-cmsis-nn.md)

**Q141.** When write custom kernels?

**Answer:** When a bottleneck op isn't well served by existing optimized kernels

Profile first.

**Q142.** What does CMSIS-NN use to speed up int8?

**Answer:** SIMD instructions and optimized memory access on Cortex-M

DSP extensions.

**Q143.** How do you verify a custom kernel?

**Answer:** Compare against the reference kernel across many inputs

Bit-exact where possible.

**Q144.** What's the maintenance cost?

**Answer:** Keeping kernels correct across toolchains and hardware

Prefer vendor libs.

### [Ultra-Low-Power Inference](level-4/07-ultra-low-power-inference.md)

**Q145.** How do you minimize inference energy?

**Answer:** Reduce computation, use duty cycling and sleep between inferences

Energy = power x time.

**Q146.** What does duty cycling mean?

**Answer:** Running briefly then sleeping to cut average power

Wake on events.

**Q147.** What hardware helps always-on?

**Answer:** Low-power wake-up sensors and tiny co-processors

Cascade.

**Q148.** What should you measure?

**Answer:** Energy per inference and average current in real usage

Use a power analyzer.

### [Hardware/Software Co-Design](level-4/08-hardware-codesign.md)

**Q149.** What is hardware/software co-design?

**Answer:** Designing the model and hardware together to meet constraints

E.g. choose the NPU for the model.

**Q150.** Why consider memory bandwidth?

**Answer:** Data movement often costs more energy and time than compute

Locality.

**Q151.** What's quantization's hardware link?

**Answer:** Integer MAC units are smaller and more efficient than floating point

Design for int8.

**Q152.** How do you choose hardware?

**Answer:** Match model needs (ops, memory) and product constraints (power, cost)

Requirements first.

### [Privacy, Safety &amp; Compliance](level-4/09-privacy-safety-compliance.md)

**Q153.** What privacy benefit does edge AI provide?

**Answer:** Raw data can stay on the device

Minimize data collection.

**Q154.** What does safety-critical use require?

**Answer:** Verification, fail-safes and documented validation

Standards like IEC 61508.

**Q155.** What regulations may apply?

**Answer:** GDPR, product safety and sector rules depending on use

Know your market.

**Q156.** Why document model limitations?

**Answer:** Users and regulators need to know failure modes

Transparency.

### [Capstone: Production Edge AI System](level-4/10-capstone-production-edge-ai.md)

**Q157.** What does the capstone integrate?

**Answer:** Model, OTA, telemetry, security, power and compliance for a fleet

Production edge AI.

**Q158.** How do you prove readiness?

**Answer:** Field tests, monitoring, rollback drills and benchmarks

Operate it.

**Q159.** What's the final measure of success?

**Answer:** Reliable value to users within power, cost and privacy constraints

System thinking.

**Q160.** What documentation helps?

**Answer:** Architecture, runbooks, model cards and update procedures

Maintainability.

## Mixed questions across all levels

**Q161.** A sensor must run a keyword model 24/7 on a coin cell. What's the right approach?

**Answer:** A tiny always-on first-stage model with duty cycling, waking bigger stages only on a hit

Power dominates edge design.

**Q162.** Why quantize a model to int8 for an MCU?

**Answer:** About 4x smaller and faster on integer hardware, with a small accuracy cost

Often required to fit.

**Q163.** A float range [-2.0, 6.0] is quantized to int8 asymmetrically. What are scale and zero point?

```python
lo,hi=-2.0,6.0
scale=(hi-lo)/255
zp=round(-128-lo/scale)
print(round(scale,5), zp)
```

**Answer:** 0.03137 -64

zero_point = round(-128 - min/scale). Verified by running it.

**Q164.** Post-training quantization loses too much accuracy. What next?

**Answer:** Try quantization-aware training so weights adapt to int8 rounding

QAT simulates quantization in training.

**Q165.** A 1-second audio clip at 16 kHz is split into 30 ms frames with a 20 ms stride. How many frames?

```python
fs=16000; frame=480; stride=320; n=16000
print((n-frame)//stride+1)
```

**Answer:** 49

floor((16000-480)/320)+1. Verified by running it.

**Q166.** Training used librosa MFCCs but firmware computes them differently and accuracy collapses. Why?

**Answer:** Preprocessing mismatch shifts the input distribution; use the same implementation or verify parity

Test features end to end.

**Q167.** The tensor arena is too small and inference fails at AllocateTensors. What do you do?

**Answer:** Increase the arena (or reduce activation sizes) and measure arena_used_bytes

Size by measurement.

**Q168.** Your model needs ops not in the OpResolver. What happens and what's the fix?

**Answer:** AllocateTensors fails; register the needed ops (or change the architecture)

Only link the ops you use.

**Q169.** Test accuracy is 95% but the field accuracy is 70%. What's the most likely cause?

**Answer:** Dataset mismatch: noise, devices, lighting or subjects absent from training

Collect representative data.

**Q170.** Why split train and test by subject or session?

**Answer:** Same-subject leakage inflates accuracy

Generalization to new users.

**Q171.** Unstructured pruning removed 80% of weights but latency barely changed. Why?

**Answer:** Kernels don't exploit scattered zeros; structured pruning or smaller dense layers speed up inference

Hardware must exploit sparsity.

**Q172.** A Coral Edge TPU model runs slower than expected. What do you check first?

**Answer:** The compile report for ops falling back to the CPU

Unsupported ops split the graph.

**Q173.** Why benchmark on your own device instead of trusting vendor TOPS?

**Answer:** Real throughput depends on supported ops, memory bandwidth and your model

Measure.

**Q174.** A microphone stream is endless. How do you feed it to a fixed-size model?

**Answer:** Slide overlapping windows over a ring buffer and smooth the predictions

Streaming inference.

**Q175.** An object detector produces many overlapping boxes for one object. What post-processing do you apply?

**Answer:** Non-maximum suppression

Keep the highest-confidence box.

**Q176.** A model update bricks 2% of a pilot fleet. Which practices would have limited the damage?

**Answer:** Signed, staged (canary) rollouts with automatic rollback to the previous model

Safe OTA.

**Q177.** How can you detect drift on devices that can't upload raw data?

**Answer:** Report lightweight statistics such as confidence histograms and feature means

Privacy-preserving telemetry.

**Q178.** Federated learning sends model updates instead of data. Does it guarantee privacy?

**Answer:** No: updates can leak information, so add secure aggregation or differential privacy

Defense in depth.

**Q179.** Which is the best measure of an always-on model's efficiency?

**Answer:** Energy per inference and average current in real usage, alongside accuracy

Measure power.

## Practical coding questions

### Int8 quantization by hand

Implement quant_params (asymmetric int8: scale = range/255, zero_point = round(-128 - min/scale)), quantize (round, shift by zero point, clamp to int8) and dequantize. Verify the rounding error never exceeds half a quantization step and compare model sizes.

Starter code:

```
def quant_params(lo, hi):
    # asymmetric int8: scale = (hi-lo)/255, zero_point = round(-128 - lo/scale)
    return 1.0, 0

def quantize(x, scale, zp):
    # round(x/scale) + zp, clamped to [-128, 127]
    return 0

def dequantize(q, scale, zp):
    # (q - zp) * scale
    return 0.0

weights = [-0.92, -0.31, 0.0, 0.18, 0.55, 0.99]
scale, zp = quant_params(min(weights), max(weights))
print("scale=%.5f zero_point=%d" % (scale, zp))
qs = [quantize(w, scale, zp) for w in weights]
print(qs)
errs = [abs(w - dequantize(q, scale, zp)) for w, q in zip(weights, qs)]
print("max error=%.4f  (<= scale/2: %s)" % (max(errs), max(errs) <= scale / 2 + 1e-9))
print("size: float32=%d bytes, int8=%d bytes" % (4 * len(weights), len(weights)))
```

Expected output:

```text
scale=0.00749 zero_point=-5
[-128, -46, -5, 19, 68, 127]
max error=0.0032  (<= scale/2: True)
size: float32=24 bytes, int8=6 bytes
```

??? success "Show a solution"

    ```
    def quant_params(lo, hi):
        scale = (hi - lo) / 255
        zero_point = round(-128 - lo / scale)
        return scale, zero_point

    def quantize(x, scale, zp):
        q = round(x / scale) + zp
        return max(-128, min(127, q))

    def dequantize(q, scale, zp):
        return (q - zp) * scale

    weights = [-0.92, -0.31, 0.0, 0.18, 0.55, 0.99]
    scale, zp = quant_params(min(weights), max(weights))
    print("scale=%.5f zero_point=%d" % (scale, zp))
    qs = [quantize(w, scale, zp) for w in weights]
    print(qs)
    errs = [abs(w - dequantize(q, scale, zp)) for w, q in zip(weights, qs)]
    print("max error=%.4f  (<= scale/2: %s)" % (max(errs), max(errs) <= scale / 2 + 1e-9))
    print("size: float32=%d bytes, int8=%d bytes" % (4 * len(weights), len(weights)))
    ```

    Each real value is rounded to the nearest quantization step, so the error is at most scale/2 while the model shrinks 4x. TFLite does this per tensor or per channel using calibration data. Verified by running the solution.

### Windows and features for a gesture classifier

Implement windows() (overlapping full windows), features() (RMS to two decimals and zero-mean crossings) and classify() (rms &lt; 0.2 is still; at least 6 crossings is shake; otherwise slow-move). Run it on three synthetic windows.

Starter code:

```
import math

def windows(samples, size, step):
    # overlapping windows (only full-size windows)
    return []

def features(w):
    # return (rms rounded to 2 decimals, number of zero-mean crossings between neighbouring samples)
    return (0, 0)

def classify(w):
    # rms < 0.2 -> "still"; else crossings >= 6 -> "shake"; else "slow-move"
    return ""

still = [0.02, -0.03, 0.01, 0.0, 0.02, -0.01, 0.0, 0.03]
shake = [1.0, -1.1, 0.9, -1.0, 1.2, -0.9, 1.0, -1.1]
slow = [0.0, 0.4, 0.8, 1.0, 0.8, 0.4, 0.0, -0.4]
for name, s in [("still", still), ("shake", shake), ("slow", slow)]:
    print(name, features(s), classify(s))
print("windows:", len(windows(list(range(20)), 8, 4)))
```

Expected output:

```text
still (0.02, 6) still
shake (1.03, 7) shake
slow (0.59, 2) slow-move
windows: 4
```

??? success "Show a solution"

    ```
    import math

    def windows(samples, size, step):
        return [samples[i:i + size] for i in range(0, len(samples) - size + 1, step)]

    def features(w):
        n = len(w)
        mean = sum(w) / n
        rms = math.sqrt(sum(x * x for x in w) / n)
        crossings = sum(1 for a, b in zip(w, w[1:]) if (a - mean) * (b - mean) < 0)
        return round(rms, 2), crossings

    def classify(w):
        rms, zc = features(w)
        if rms < 0.2:
            return "still"
        return "shake" if zc >= 6 else "slow-move"

    still = [0.02, -0.03, 0.01, 0.0, 0.02, -0.01, 0.0, 0.03]
    shake = [1.0, -1.1, 0.9, -1.0, 1.2, -0.9, 1.0, -1.1]
    slow = [0.0, 0.4, 0.8, 1.0, 0.8, 0.4, 0.0, -0.4]
    for name, s in [("still", still), ("shake", shake), ("slow", slow)]:
        print(name, features(s), classify(s))
    print("windows:", len(windows(list(range(20)), 8, 4)))
    ```

    Windowing plus cheap time-domain features is the standard first stage of an edge sensor pipeline. A tiny rule or model on those features can run in microseconds. Verified by running the solution.

---

More free practice: the [Edge AI Mastery Path](index.md) curriculum, the [Interview Prep Pack](https://learnmasterypath.com/interview-pack/) and the [free resume ATS check](https://learnmasterypath.com/resume/).
