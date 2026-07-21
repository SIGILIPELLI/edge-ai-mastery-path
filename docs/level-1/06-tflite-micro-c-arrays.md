# 06 · TFLite-Micro & C Arrays

Stage 4 begins. A microcontroller has no filesystem to load
`sine_model_int8.tflite` from — so the model must be **compiled into the
firmware as a C byte array**, and executed by **TFLite-Micro (TFLM)**, a
stripped-down C++ interpreter built for machines with kilobytes of RAM and
no OS. The best part for a hardware-less course: TFLM is portable C++, so
in this module you'll build and run a genuine microcontroller inference
program *on your own computer*. Same code, same runtime, same bytes — the
board can wait.

## From `.tflite` to C array with `xxd`

The classic tool is `xxd` (ships with vim on macOS/Linux):

```bash
xxd -i sine_model_int8.tflite > sine_model_data.cc
```

Which produces exactly this shape of file:

```c
unsigned char sine_model_int8_tflite[] = {
  0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, 0x14, 0x00, 0x20, 0x00,
  0x04, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x10, 0x00, 0x14, 0x00, 0x00, 0x00,
  /* ... ~2500 more bytes ... */
};
unsigned int sine_model_int8_tflite_len = 2568;
```

Those bytes *are* the flatbuffer — note `0x54 0x46 0x4c 0x33` = ASCII
`"TFL3"`, the format's magic marker, a few bytes in. No `xxd` on Windows?
Three lines of Python do the same:

```python
data = open("sine_model_int8.tflite", "rb").read()
with open("sine_model_data.cc", "w") as f:
    f.write("unsigned char g_sine_model[] = {\n  " +
            ",".join(f"0x{b:02x}" for b in data) +
            f"\n}};\nunsigned int g_sine_model_len = {len(data)};\n")
```

One production detail worth adopting now: real projects declare the array
`alignas(16) const unsigned char ...`. `const` keeps it in flash instead of
being copied to RAM at boot (on MCUs, a several-hundred-KB difference), and
16-byte alignment is required for the interpreter to read tensors in place.

## The TFLM runtime: three moving parts

TFLM inference always assembles the same three objects — meet them once and
every TFLM program you ever read becomes familiar:

```text
┌───────────────┐  ┌──────────────────────┐  ┌─────────────────────┐
│ model bytes    │  │ op resolver           │  │ tensor arena         │
│ (the C array,  │  │ the kernel functions  │  │ one static byte      │
│  lives in      │  │ this model needs —    │  │ buffer: ALL scratch  │
│  flash)        │  │ and only those        │  │ memory for inference │
└───────┬───────┘  └──────────┬───────────┘  └──────────┬──────────┘
        └──────────────► MicroInterpreter ◄─────────────┘
                        .AllocateTensors() then .Invoke()
```

- **Op resolver** — the flatbuffer names operations (`FULLY_CONNECTED`,
  `CONV_2D`, …); the resolver maps names to compiled kernel code. On
  desktops you'd link every op; on an MCU each op costs flash, so
  `MicroMutableOpResolver` lets you register *only* what your model uses.
  Forget one and `AllocateTensors` fails with "op not found" — the single
  most common TFLM error.
- **Tensor arena** — TFLM never calls `malloc`. You hand it one
  fixed-size byte array, and the interpreter plans every input, output, and
  intermediate tensor into it at `AllocateTensors()` time. Too small →
  allocation fails cleanly at init, not randomly at 2 a.m. in the field.
  You find the size empirically: start generous, query
  `arena_used_bytes()`, trim.
- **MicroInterpreter** — wires the three together and exposes the same
  set-input → `Invoke()` → read-output flow you used in Python in Module 04.

## A minimal, complete inference program

`main.cc` — this is real TFLM code, the same on a laptop and on an ESP32:

```cpp
#include <cstdio>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "sine_model_data.h"   // the C array from xxd

namespace {
  constexpr int kArenaSize = 4 * 1024;         // 4 KB is plenty here
  alignas(16) uint8_t tensor_arena[kArenaSize];
}

int main() {
  // 1. Map the model bytes
  const tflite::Model* model = tflite::GetModel(g_sine_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    printf("schema version mismatch\n");
    return 1;
  }

  // 2. Register ONLY the ops this model uses
  tflite::MicroMutableOpResolver<1> resolver;
  resolver.AddFullyConnected();

  // 3. Build the interpreter over the arena
  tflite::MicroInterpreter interpreter(model, resolver,
                                       tensor_arena, kArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("AllocateTensors failed (arena too small? missing op?)\n");
    return 1;
  }
  printf("arena used: %zu bytes\n", interpreter.arena_used_bytes());

  TfLiteTensor* input  = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);

  // 4. Quantize input, invoke, dequantize output — for x = pi/2
  const float x = 1.5708f;
  input->data.int8[0] = static_cast<int8_t>(
      x / input->params.scale + input->params.zero_point);

  interpreter.Invoke();

  float y = (output->data.int8[0] - output->params.zero_point)
            * output->params.scale;
  printf("sin(%.4f) ~= %.4f  (true: 1.0000)\n", x, y);
  return 0;
}
```

Every concept from Modules 04–05 reappears in C: tensor indices become
`input(0)`/`output(0)`, the Python `(scale, zero_point)` tuple becomes
`params.scale`/`params.zero_point`, and `invoke()` is `Invoke()`. Nothing
on a microcontroller will be conceptually new — only smaller.

## Building it on your desktop

TFLM lives in its own repository and builds for x86 out of the box:

```bash
git clone --depth 1 https://github.com/tensorflow/tflite-micro.git
cd tflite-micro

# Sanity check: build and run Google's own hello_world (also a sine model!)
make -f tensorflow/lite/micro/tools/make/Makefile hello_world
./gen/*/bin/hello_world
```

The easiest way to build *your* program is to piggyback on that example:
copy your `main.cc`, `sine_model_data.cc`, and `sine_model_data.h` over the
files in `tensorflow/lite/micro/examples/hello_world/`, and run the same
`make` target. You now have microcontroller inference code running with a
`printf` — verify the printed value matches Module 05's Python
`int8_predict(π/2)` to the same int8 step (they execute identical kernels
over identical bytes, so they should agree essentially exactly).

## Cheat sheet

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

## Exercise

1. Convert your `sine_model_int8.tflite` to a C array both ways (`xxd` and
   the Python snippet) and diff the byte counts. Find the `"TFL3"` magic in
   the hex.
2. Build and run the program above. Report `arena_used_bytes()`, then
   binary-search the smallest `kArenaSize` (to the nearest 256 bytes) that
   still allocates successfully.
3. Comment out `resolver.AddFullyConnected()` and rebuild. What exactly
   happens, and at which call? Why is failing *there* the best possible
   place?
4. Extend `main.cc` to sweep x from 0 to 2π in 20 steps, printing a crude
   ASCII plot of the predictions. Compare against Python's output for the
   same sweep — the pipeline's last verification before real hardware.
