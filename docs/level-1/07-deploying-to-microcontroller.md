# 07 · Deploying to a Microcontroller

The desktop program from Module 06 was the dress rehearsal; this module is
opening night. We take the same model bytes and near-identical code to an
**ESP32** — the course's reference board (~$5, 520 KB RAM, 4 MB flash,
240 MHz, WiFi) — using the Arduino toolchain. You'll see what actually
changes on-device (very little code, a lot of environment), how memory
constraints bite in practice, and how far you can get with the free
[Wokwi](https://wokwi.com/) browser simulator if you don't own hardware.

!!! info "No board? Keep reading anyway"
    This module works as a guided tour: every constraint discussed here
    (arena sizing, flash layout, latency measurement) is a concept you
    already met on the desktop — here you see its on-device face. Wokwi can
    simulate the ESP32 and its Arduino toolchain in the browser; TFLM
    projects compile there too, though large library builds can be slow.

## The Arduino + TFLM setup

1. Install the [Arduino IDE](https://www.arduino.cc/en/software), then add
   ESP32 support: *Settings → Additional boards manager URLs* →
   `https://espressif.github.io/arduino-esp32/package_esp32_index.json`,
   then install **esp32** in the Boards Manager.
2. Install a TFLM Arduino library. The maintained fork for ESP32 is
   **TFLite Micro for Espressif Chips** (`esp-tflite-micro`); Arduino's
   Library Manager also carries TFLM builds such as the
   Chirale_TensorFlowLite library for Arduino boards. Any of them provides
   the same headers you used in Module 06.
3. Create a sketch folder containing the `.ino` file below plus your
   `sine_model_data.h` / `sine_model_data.cc` from Module 06.

## The sketch: Module 06's code in Arduino clothes

```cpp
// sine_esp32.ino
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "sine_model_data.h"

namespace {
  constexpr int kArenaSize = 4 * 1024;
  alignas(16) uint8_t tensor_arena[kArenaSize];
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
}

void setup() {
  Serial.begin(115200);

  model = tflite::GetModel(g_sine_model);

  static tflite::MicroMutableOpResolver<1> resolver;
  resolver.AddFullyConnected();

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed");
    while (true) {}          // halt: nothing sensible to do without a model
  }
  input = interpreter->input(0);
  output = interpreter->output(0);
  Serial.printf("arena used: %u bytes\n",
                (unsigned) interpreter->arena_used_bytes());
}

void loop() {
  static float x = 0.0f;

  input->data.int8[0] = (int8_t) roundf(
      x / input->params.scale + input->params.zero_point);

  uint32_t t0 = micros();
  interpreter->Invoke();
  uint32_t dt = micros() - t0;

  float y = (output->data.int8[0] - output->params.zero_point)
            * output->params.scale;
  Serial.printf("x=%.3f  sin(x)~=%.3f  (%u us)\n", x, y, (unsigned) dt);

  x += 0.3f;
  if (x > 6.2832f) x = 0.0f;
  delay(200);
}
```

Compare with Module 06's `main.cc`: the model loading, resolver, arena,
quantize/invoke/dequantize logic are **line-for-line identical**. What
changed is packaging — `setup()`/`loop()` instead of `main()`, `Serial`
instead of `printf`, `static` objects so they outlive `setup()`. This is
the payoff of the desktop-first workflow: the hard parts were debugged
where you had a real debugger and instant rebuilds.

Select your board and port, click Upload, open the Serial Monitor at
115200 baud, and watch sine values stream out at ~15–40 µs per inference —
your first neural network running on a $5 chip.

## Where memory constraints actually bite

On the desktop you never thought about these; on-device they are the job:

- **Flash layout.** Your firmware + TFLM library + model array share flash.
  The Arduino IDE prints usage after each build ("Sketch uses 412,318 bytes
  (31%)…"). The TFLM library alone costs ~100–300 KB; each registered op
  adds more — which is *why* `MicroMutableOpResolver` exists instead of an
  everything-included resolver.
- **RAM partitioning.** The ESP32's ~320 KB of usable data RAM must hold
  the stack, WiFi buffers (if used, ~40 KB+), your app, *and* the tensor
  arena. The desktop trick of "start with a huge arena" fails here — a
  200 KB arena on top of WiFi may simply not link or will crash at boot.
  Size it with `arena_used_bytes()` + ~10% headroom.
- **The model array placement.** `const` puts the model in flash-mapped
  memory, read in place; forgetting `const` copies ~all of it into RAM at
  boot. On our 2.5 KB sine model, invisible; on a 300 KB vision model,
  fatal.
- **Latency is now real.** `micros()` around `Invoke()` is the on-device
  equivalent of a profiler. The sine model runs in tens of microseconds;
  the MNIST CNN from Module 05 takes tens of milliseconds — numbers you'll
  budget against sensor sampling rates in Module 08.

!!! warning "Boards smaller than the ESP32"
    On an Arduino Uno (2 KB RAM) this course's runtime doesn't fit at all —
    classic TinyML boards start around the Arduino Nano 33 BLE Sense
    (256 KB RAM). If your board has under ~64 KB of RAM, check the TFLM
    documentation's supported-platforms list before buying anything else.

## Trying it in Wokwi

[Wokwi](https://wokwi.com/) simulates ESP32 Arduino projects in the
browser, including third-party libraries via its `libraries.txt` project
file. Practical notes from the trenches:

- Start from an ESP32 template project, add your `.ino` and the model
  files, and add the TFLM library to `libraries.txt`.
- Builds of TFLM-sized libraries are slow in the free tier — minutes, not
  seconds. Patience, or trim to the desktop path for iteration and use
  Wokwi only for the final demo.
- Serial output works exactly like the real Serial Monitor, so the sketch
  above runs unmodified.
- Wokwi also simulates sensors (potentiometers, MPU6050 accelerometer) —
  which becomes genuinely useful in Module 08 and the capstone.

## Cheat sheet

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

## Exercise

1. (Board or Wokwi) Get the sketch running and record: arena bytes used,
   flash percentage from the build output, and average `Invoke()` time over
   100 runs.
2. Remove `const` from the model array (if your Module 06 file had it),
   rebuild, and compare the IDE's reported *dynamic memory* usage. Put it
   back. Extrapolate: what would this mistake cost with a 300 KB model?
3. Deliberately set `kArenaSize` to 1024. What happens, and — referring
   back to Module 06 — why is this failure mode considered a *feature* of
   TFLM's design?
4. Paper exercise if simulating: your next model needs a 180 KB arena and
   the product needs WiFi (~45 KB of RAM) plus 30 KB of application
   buffers. Draw the ESP32 RAM budget and decide: does it fit in ~320 KB of
   usable data RAM, and with how much headroom?
