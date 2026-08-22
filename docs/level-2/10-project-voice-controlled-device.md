# Project — Voice-Controlled Device

This capstone combines every technique from Level 2 into one device: an
ESP32 that listens continuously for a small set of spoken commands (e.g.
"on", "off", "up", "down") and triggers an action, running entirely
on-device with no cloud connection. It uses Module 01's audio pipeline,
Module 04's dataset discipline, Modules 05-06's compression techniques, and
Module 09's optimization process — end to end, on real hardware.

## Project scope and success criteria

Build a keyword-spotting system that recognizes 4-6 short commands plus a
"background/unknown" class, running as a continuous streaming detector on
an ESP32 (no camera needed — this is audio-only, using an I2S MEMS
microphone such as the INMP441, a common ~$3 breakout).

Concrete success criteria to design against:
- **On-device latency**: detection fires within ~500ms of the command
  ending.
- **Flash budget**: quantized model + firmware fits comfortably under the
  ESP32's flash partition (target under 200 KB for the model itself).
- **RAM budget**: tensor arena + audio ring buffer + application code fits
  in the ESP32's usable SRAM without needing PSRAM.
- **False-accept rate**: fewer than roughly 1 false trigger per 10 minutes
  of ambient household audio that contains no command.

## Step 1: dataset (apply Module 04 directly)

Collect or source audio for each command word plus a background/unknown
class. The Google Speech Commands dataset is the standard public reference
if you want existing data to prototype against; for a device tuned to your
own environment, recording your own samples (per Module 04's "condition
diversity beats volume" principle) covering multiple speakers, distances
from the mic, and background noise levels will generalize far better to
your actual room than a generic public dataset alone.

```python
import numpy as np

# Apply Module 04's coverage check directly to this project's dataset.
def command_dataset_coverage(metadata):
    from collections import defaultdict
    coverage = defaultdict(lambda: defaultdict(int))
    for m in metadata:
        coverage[m["command"]][m["condition"]] += 1
    return {k: dict(v) for k, v in coverage.items()}

meta = [
    {"command": "on", "condition": "quiet_near"},
    {"command": "on", "condition": "quiet_far"},
    {"command": "off", "condition": "quiet_near"},
    {"command": "off", "condition": "noisy_near"},
    {"command": "unknown", "condition": "quiet_near"},
    {"command": "unknown", "condition": "noisy_near"},
]
print(command_dataset_coverage(meta))
# Reveals "on" has no noisy-condition samples -- a gap to fix before training.
```

Deliberately build the `unknown`/background class from hard negatives —
other household speech, TV audio, similar-sounding words ("on" vs. "one")
— exactly per Module 04's negative-class-design guidance.

## Step 2: feature extraction and model (apply Modules 01-02)

Reuse Module 01's `log_mel_spectrogram` pipeline unchanged — this project
is the direct continuation of that module's wake-word groundwork, just with
multiple target commands instead of one keyword. The model is the same
small CNN-over-spectrogram shape from Module 01/02, with the output layer
sized to `n_commands + 1` (the `+1` for background/unknown).

```python
# Described, not run -- needs a Keras/TF training environment.
# model = keras.Sequential([
#     keras.layers.Input(shape=(98, 40, 1)),
#     keras.layers.Conv2D(8, (10, 4), strides=(2, 2), activation="relu"),
#     keras.layers.DepthwiseConv2D(3, strides=2, activation="relu"),
#     keras.layers.Conv2D(16, 1, activation="relu"),
#     keras.layers.GlobalAveragePooling2D(),
#     keras.layers.Dense(n_commands + 1, activation="softmax"),
# ])
```

## Step 3: compress (apply Modules 05-06)

Run the optimization pass from Module 09 against the specific budget above:

1. Train the float model to a validation accuracy baseline.
2. If flash is over budget: apply structured channel pruning (Module 05),
   fine-tune, and re-measure.
3. Quantize to int8. Start with post-training quantization (Level 1 Module
   05); only escalate to quantization-aware training (Module 06) if PTQ's
   accuracy loss is unacceptable against your false-accept-rate target.
4. Re-run the size/accuracy table from Level 1 Module 05's format
   (float vs. pruned vs. quantized) so the tradeoff at each stage is
   documented, not assumed.

```python
def compression_stage_report(stage_name, size_kb, accuracy):
    print(f"{stage_name:20s}  size={size_kb:7.1f} KB   accuracy={accuracy:.3f}")

compression_stage_report("float baseline", 340.0, 0.94)
compression_stage_report("pruned 50%",     190.0, 0.925)
compression_stage_report("pruned+int8",     52.0, 0.918)
```

## Step 4: on-device streaming detector (apply Modules 01, 08, 09)

Combine the I2S microphone read loop, the sliding-window feature extraction,
and the debounce logic from Module 01's `StreamingDetector` (the same
hysteresis-style idea used for vision in Module 08's
`PersonPresenceTracker` applies directly to audio too).

```cpp
// Reviewed pattern, not run -- combines I2S mic capture with the TFLM
// inference call pattern from Level 1 Module 07 and Module 07's tensor
// preprocessing style.
#include <driver/i2s.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "command_model_data.h"

namespace {
  constexpr int kArenaSize = 40 * 1024;
  alignas(16) uint8_t tensor_arena[kArenaSize];
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  constexpr int kSampleRate = 16000;
  constexpr int kWindowSamples = kSampleRate * 1;   // 1 second, matches Module 01
  int16_t audio_ring_buffer[kWindowSamples];
  int ring_write_pos = 0;
}

void read_i2s_into_ring_buffer() {
  int16_t chunk[512];
  size_t bytes_read = 0;
  i2s_read(I2S_NUM_0, chunk, sizeof(chunk), &bytes_read, portMAX_DELAY);
  int n_samples = bytes_read / sizeof(int16_t);
  for (int i = 0; i < n_samples; i++) {
    audio_ring_buffer[ring_write_pos] = chunk[i];
    ring_write_pos = (ring_write_pos + 1) % kWindowSamples;
  }
}

void run_inference_on_current_window() {
  // Feature extraction (log-mel, per Module 01) happens here, writing
  // directly into `input->data.int8` with the same scale/zero_point
  // quantization pattern used throughout this course.
  interpreter->Invoke();

  int best_class = 0;
  int8_t best_score = output->data.int8[0];
  for (int i = 1; i < output->dims->data[1]; i++) {
    if (output->data.int8[i] > best_score) {
      best_score = output->data.int8[i];
      best_class = i;
    }
  }
  float confidence = (best_score - output->params.zero_point) * output->params.scale;
  Serial.printf("class=%d confidence=%.2f\n", best_class, confidence);
  // Feed best_class/confidence into a StreamingDetector-style debounce
  // (Module 01) before triggering any action, to avoid single-frame false triggers.
}
```

## Step 5: measure against the success criteria

Repeat Module 09's profiling discipline against this project's specific
numbers: `arena_used_bytes()` against the RAM budget, build-tool flash
percentage against the 200 KB target, `micros()`-measured inference time
against the 500ms detection-latency budget (inference itself should be a
small fraction of that budget — most of the 500ms is the 1-second sliding
window plus debounce delay), and a real false-accept count from at least 10
minutes of ambient recorded audio containing no commands.

## Stretch goals

1. **Add a second confirmation stage.** After the on-device model fires,
   require the *same* class to win two consecutive inference windows
   before triggering the action (adapt `StreamingDetector`'s
   `consecutive_needed` from Module 01) — measure how much this reduces
   your false-accept count and by how much it adds to detection latency.
2. **Apply quantization-aware training and compare.** Re-run the
   compression pipeline with QAT (Module 06) instead of plain PTQ, and
   build a side-by-side table showing whether the accuracy recovery is
   large enough to justify the extra training complexity for this specific
   model size.
3. **Add power-aware duty cycling.** Gate the full spectrogram+CNN pipeline
   behind a cheap energy-based voice-activity detector (a simple RMS
   threshold on raw ring-buffer samples, as discussed in Module 01), and
   estimate the resulting average power reduction versus always-on
   inference.
4. **Multi-command action mapping with confidence-based rejection.** Instead
   of always acting on the top class, add a minimum-confidence threshold
   below which the device treats the frame as "unknown" even if a command
   class scored highest — tune this threshold using a labeled validation
   set and report the resulting precision/recall tradeoff per command.
5. **Extend to wireless notification.** Have the ESP32 publish detected
   commands over WiFi (MQTT or a simple HTTP POST) to a local dashboard, and
   measure how much the added WiFi RAM usage affects your tensor arena
   budget — tying back directly to Level 1 Module 07's RAM-partitioning
   warning about WiFi buffers competing with the arena.
