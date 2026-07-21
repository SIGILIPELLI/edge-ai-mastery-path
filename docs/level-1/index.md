# Level 1 · Entry <span class="level-badge">Foundations</span>

Goal: understand what edge AI is and master the complete TinyML workflow —
train a small model with Keras, convert it to TensorFlow Lite (LiteRT),
quantize it to int8, turn it into a C array for TFLite-Micro, and understand
exactly what changes when it runs on a microcontroller — finishing with a
complete gesture-recognition pipeline.

**You do not need to own any hardware for this level.** Every module runs on
a normal laptop with Python and TensorFlow: training, conversion,
quantization, and even the C++ inference code (which compiles as a desktop
program before it ever touches a board). Microcontroller deployment is
covered as the follow-on step in each module, with
[Wokwi simulator](https://wokwi.com/) notes where the ESP32 side can be
tried in the browser.

## Modules

1. [What Is Edge AI](01-what-is-edge-ai.md)
2. [The Edge AI Workflow](02-edge-ai-workflow.md)
3. [Training a Tiny Model](03-training-a-tiny-model.md)
4. [Converting to TFLite/LiteRT](04-converting-to-tflite.md)
5. [Quantization](05-quantization.md)
6. [TFLite-Micro & C Arrays](06-tflite-micro-c-arrays.md)
7. [Deploying to a Microcontroller](07-deploying-to-microcontroller.md)
8. [Sensors & Feature Extraction](08-sensors-feature-extraction.md)
9. [Evaluating & Debugging Edge Models](09-evaluating-debugging.md)
10. [Capstone — Gesture Recognition](10-capstone-gesture-recognition.md)

By the end of this level you'll be able to take a machine-learning idea from
a Keras training script to a quantized `.tflite` flatbuffer to a C array
compiled into microcontroller-style firmware — verifying at every step that
the model still produces the right answers — and you'll know precisely what
RAM, flash, latency, and accuracy each step costs.
