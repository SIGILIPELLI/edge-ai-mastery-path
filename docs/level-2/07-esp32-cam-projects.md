# ESP32-CAM Vision Projects

The ESP32-CAM pairs the ESP32 you already know from Level 1 with an OV2640
camera module — roughly $8-10, no separate camera board required. It's the
default cheap hardware for on-device vision in the TinyML community, and it
comes with real constraints that don't show up on a training laptop: a
camera driver that eats RAM before your model even loads, no built-in
USB-serial chip (you need an external programmer), and a PSRAM chip that's
required for anything beyond the smallest frame sizes. This module covers
the board's quirks, the camera-to-tensor pipeline in C, and the code you
review (not run) for capturing and classifying frames on-device.

## Board quirks before you write any code

- **No onboard USB.** Programming requires an external FTDI/USB-to-serial
  adapter wired to the ESP32-CAM's UART pins, plus a jumper between `IO0`
  and `GND` during flashing (removed for normal boot). This trips up nearly
  everyone the first time.
- **PSRAM is mandatory for larger frames.** The OV2640 at anything above
  `QVGA` (320x240) needs the board's external PSRAM chip for frame buffers;
  the ESP32's internal SRAM alone (~320 KB usable) can't hold a `VGA` or
  larger JPEG/RGB buffer alongside your application and model arena.
  Confirm `psramFound()` returns true before requesting large frame sizes.
- **The flash LED is wired to a GPIO you'll likely want for something
  else.** GPIO4 drives the onboard white LED *and* is one of the SD-card
  pins — a common source of "why is my LED flickering" or "why won't SD
  init" confusion.
- **Camera init claims a meaningful RAM/flash budget before you've done
  anything.** The `esp32-camera` driver itself, plus its internal frame
  buffers, is worth accounting for in the same memory budget exercise from
  Module 02 and Level 1 Module 07 — it isn't free overhead you can ignore.

## The camera-to-tensor pipeline

A vision inference pipeline on ESP32-CAM has one extra stage that a
sine-wave or audio model doesn't: decoding and resizing a camera frame
into the exact tensor shape and format the model expects.

```cpp
// Reviewed for correctness; not executed in this environment (needs
// physical ESP32-CAM hardware + Arduino/ESP-IDF toolchain + esp32-camera lib).
#include "esp_camera.h"

// Typical AI-Thinker ESP32-CAM pin map (varies by board vendor -- verify
// against your specific board's silkscreen/datasheet before flashing).
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

bool init_camera() {
  camera_config_t config = {};
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_GRAYSCALE;   // matches Module 02's grayscale models
  config.frame_size = FRAMESIZE_96X96;         // small on purpose -- see Module 02/09
  config.fb_count = 1;                          // 2 needs PSRAM; 1 fits internal SRAM

  return esp_camera_init(&config) == ESP_OK;
}

// Capture one frame and hand back a pointer usable directly as a model input,
// since PIXFORMAT_GRAYSCALE + FRAMESIZE_96X96 already matches a 96x96x1 tensor.
camera_fb_t* capture_for_inference() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) return nullptr;
  // fb->buf is 96*96 = 9216 bytes of raw grayscale, row-major -- no decode
  // step needed because we requested GRAYSCALE directly from the sensor.
  return fb;   // caller must esp_camera_fb_return(fb) when done with it
}
```

Requesting `PIXFORMAT_GRAYSCALE` and a frame size that exactly matches the
model's input resolution (`FRAMESIZE_96X96`) sidesteps the resize/color-
convert mismatch problem from Module 02 entirely — the sensor does the
resizing in hardware, and there's no JPEG decode or color-space math to get
subtly wrong between training and deployment. This is the recommended
default for a first vision project; JPEG capture plus software decode and
resize is more flexible (arbitrary target resolutions, doesn't waste sensor
dynamic range at extreme downscales) but adds a real decode-time and
RAM cost worth avoiding until you need it.

## Wiring the captured frame into the TFLM inference call

```cpp
// Reviewed, not run -- combines the camera capture above with the TFLM
// pattern from Level 1 Module 07's sketch.
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "person_detect_model_data.h"

namespace {
  constexpr int kArenaSize = 96 * 1024;   // vision models need much more than the 4KB sine arena
  alignas(16) uint8_t tensor_arena[kArenaSize];
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
}

void run_one_inference() {
  camera_fb_t* fb = capture_for_inference();
  if (!fb) { Serial.println("capture failed"); return; }

  // Copy+quantize the raw grayscale bytes into the model's int8 input tensor.
  // Camera bytes are already uint8 0-255; the model expects int8, so the
  // conversion is the same affine mapping from Level 1 Module 05.
  for (int i = 0; i < 96 * 96; i++) {
    float pixel_float = fb->buf[i] / 255.0f;
    input->data.int8[i] = (int8_t) roundf(
        pixel_float / input->params.scale + input->params.zero_point);
  }
  esp_camera_fb_return(fb);   // release the frame buffer back to the driver

  uint32_t t0 = micros();
  interpreter->Invoke();
  uint32_t dt = micros() - t0;

  float person_score = (output->data.int8[0] - output->params.zero_point)
                        * output->params.scale;
  Serial.printf("person_score=%.3f  (%u us)\n", person_score, (unsigned) dt);
}
```

## Edge-AI tradeoffs

**Frame size vs. RAM budget.** Every step up in `FRAMESIZE_*` roughly
quadruples the frame buffer (dimensions scale in both axes); `96x96`
grayscale is 9 KB per frame and needs no PSRAM, while `VGA` grayscale is
already 300 KB and single-buffered `VGA` RGB888 exceeds internal SRAM
outright.

**Single vs. double frame buffering (`fb_count`).** Double-buffering
(`fb_count = 2`) lets the camera DMA into one buffer while your code reads
the other, improving throughput/reducing tearing — but doubles RAM cost and
requires PSRAM for anything but the smallest frame sizes.

**Sensor-side resize vs. software resize.** Capturing at the model's native
resolution directly (as above) is cheap and avoids training/inference
mismatch, but limits you to the sensor's fixed frame-size steps; software
resize from a larger capture is flexible but costs CPU time and RAM for the
intermediate buffer.

**Grayscale vs. color capture.** As in Module 02, grayscale cuts memory
3x and often costs little accuracy for presence/absence tasks — combined
with a small `FRAMESIZE`, this is what makes person-detection-class models
(Module 08) fit on an ESP32-CAM at all.

## Cheat sheet

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

## How It Actually Works

**Why requesting `PIXFORMAT_GRAYSCALE` at the model's exact resolution
moves work from software into the sensor's own analog/digital pipeline.**
The OV2640 sensor has an internal image signal processor (ISP) that
performs demosaicing (its raw photodiode array is actually a Bayer color
filter, RGB, even when you never see raw Bayer data), scaling, and
color-space conversion in hardware, on dedicated silicon, before handing
finished pixels over the parallel DVP bus (the `Y0`–`Y9`/`D0`–`D7` pins,
`VSYNC`/`HREF`/`PCLK` timing signals in `init_camera()`). Asking for
`FRAMESIZE_96X96` grayscale directly means the ISP performs the resize and
the RGB→gray luma conversion using its own fixed-function circuitry at
essentially zero additional CPU cost or code — versus capturing a larger
JPEG or RGB frame and running a software resize/color-convert on the
ESP32's general-purpose core, which costs cycles, RAM for an intermediate
buffer, and a second place a rounding/algorithm mismatch with training-time
preprocessing (Module 02) could creep in.

**Why frame buffer count interacts with DMA, not just "using more RAM."**
The camera driver streams pixel data into a frame buffer via DMA (direct
memory access) — the sensor's PCLK-clocked byte stream is written into
SRAM without CPU intervention, freeing the core to do other work while a
frame fills. With `fb_count = 1`, the single buffer being DMA-filled for
frame N+1 is the same buffer your code just finished reading for frame N,
which forces a strict "wait for full frame, read it, only then request
the next" sequence — the DMA engine cannot start writing a new frame into
a buffer your inference loop hasn't released yet. `fb_count = 2` lets DMA
fill buffer B while your code still reads buffer A, pipelining capture and
inference — real throughput gain, but it needs a second full-size buffer
existing simultaneously, which is exactly why it requires PSRAM once
frames grow past what fits twice in internal SRAM.

**Why the int8 conversion in `run_one_inference` reintroduces exactly
Module 05's affine mapping, byte for byte.** Camera pixels arrive as
`uint8` (0–255, unsigned); the model's quantized input tensor is `int8`
(-128–127, signed) with its own `(scale, zero_point)` learned during
calibration. Dividing by 255 first normalizes to a float in `[0,1]` — the
same range the training pipeline's images were normalized to before
quantization-aware calibration saw them — and then applying
`pixel_float/scale + zero_point` performs the identical quantize step
Level 1 Module 05's `int8_predict` applied to sine-wave inputs. Skipping
the `/255.0f` normalization (feeding raw 0–255 uint8 values straight into
the scale/zero_point formula) is a common on-device bug that produces
outputs that are numerically valid int8 values yet completely
uncorrelated with the model's trained expectations — because the affine
mapping now operates on a value range 255× larger than what the scale was
calibrated for.

## Exercise

1. Read through `init_camera()` and `capture_for_inference()` above and
   list every place a mismatch between training-time preprocessing (Module
   02) and this capture configuration could silently hurt accuracy —
   confirm whether this specific configuration avoids each one.
2. Using the memory-accounting method from Module 02, estimate total RAM
   needed for: one 96x96 grayscale frame buffer + a 96 KB tensor arena +
   a rough 20 KB of driver/stack overhead. Does it fit inside the ESP32's
   ~320 KB of usable internal SRAM without PSRAM?
3. If you own an ESP32-CAM: wire it to an FTDI adapter, flash the official
   `CameraWebServer` example sketch, and confirm you can view a live stream
   in a browser before attempting any custom inference code. Record the
   frame size and format you tested.
4. Paper exercise if you don't have hardware: given `FRAMESIZE_QVGA`
   (320x240) grayscale needs PSRAM, but `FRAMESIZE_96X96` doesn't, compute
   the byte sizes of both and explain the crossover in your own words.
