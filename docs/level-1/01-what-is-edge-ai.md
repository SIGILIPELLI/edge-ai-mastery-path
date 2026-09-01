# 01 · What Is Edge AI

Edge AI means running machine-learning **inference on the device that owns
the data** — a microcontroller inside a sensor, a camera, a wearable, a car —
instead of shipping the data to a cloud server and waiting for an answer.
The model was almost always *trained* in the cloud or on a workstation; what
moves to the edge is the *prediction* step. This module builds the mental
model for the whole course: why anyone bothers, what it costs, and what kind
of hardware we're actually talking about when we say "edge."

## Cloud inference vs. edge inference

The default way to use a trained model today is as a service call:

```text
Cloud inference
  sensor → network → cloud GPU runs model → network → device acts
  (two network hops on every single prediction)

Edge inference
  sensor → model runs on the device itself → device acts
  (no network in the loop at all)
```

Both run the *same kind* of model. The difference is where the multiply-adds
happen, and that one choice changes latency, privacy, power, cost, and
reliability all at once.

## Why push inference to the edge?

**Latency.** A round trip to a cloud API takes tens to hundreds of
milliseconds on a good network — and is unbounded on a bad one. A keyword
spotter running on-device answers in a few milliseconds, every time. For a
fall detector, an airbag-adjacent system, or a machine that must stop *now*,
the network is simply not allowed in the loop.

**Privacy.** An always-on microphone that streams raw audio to a server is a
surveillance device. The same microphone feeding an on-device wake-word model
never transmits audio at all — only, at most, the event "wake word heard."
Data that never leaves the device can't be intercepted, subpoenaed, or
breached in bulk.

**Power and bandwidth.** Radios are power-hungry. On many battery-powered
sensor nodes, transmitting a second of raw accelerometer or audio data costs
far more energy than running a small neural network over it. Sending the
*result* ("vibration signature abnormal") instead of the raw waveform can
stretch battery life from days to years, and shrinks bandwidth from
kilobytes-per-second to bytes-per-hour.

**Cost and scale.** Cloud inference is metered: every prediction from every
device, forever. On-device inference costs whatever the chip costs — once. A
fleet of 100,000 sensors each classifying data 10 times per second is ruinous
as API calls and free as local compute.

**Reliability.** No connectivity, no cloud. Devices in basements, on ships,
in vehicles, or in countries with patchy networks must keep working offline.

!!! info "So why isn't everything edge?"
    Because the cloud has unlimited memory and compute. A 70-billion-parameter
    language model does not fit in a microcontroller. Edge AI is a game of
    deciding which problems can be solved by a model small enough to live on
    the device — and shrinking models until they qualify.

## The constraint landscape

The word "edge" covers an enormous range of hardware. This course focuses on
the small end (TinyML), but you should know the whole ladder:

| Device class | Example | RAM | Storage | Typical model budget |
|---|---|---|---|---|
| Microcontroller (MCU) | Arduino Nano 33 BLE, STM32 | 32–512 **KB** | 256 KB–2 MB flash | 10–300 KB |
| Beefy MCU / DSP | ESP32, ESP32-S3 | 520 KB (+ PSRAM) | 4–16 MB flash | 100 KB–1 MB |
| Edge SBC | Raspberry Pi 4/5 | 2–8 **GB** | SD card | 5–100 MB |
| Edge + accelerator | Pi + Coral Edge TPU, Jetson | GBs + NPU | large | 10–500 MB |
| Smartphone | any recent phone | 4–12 GB + NPU | large | up to GBs |

Reading that table like an ML engineer, the microcontroller row should shock
you: **kilobytes** of RAM. For comparison, a single 224×224 RGB image as
float32 is ~600 KB — bigger than the *entire memory* of most MCUs. And
there's more:

- **No operating system** (or a tiny RTOS): no Linux, no Python, no files,
  no `pip install`. Your model and runtime are compiled into the firmware.
- **No floating-point luxury**: many MCUs have slow or no hardware
  floating-point, which is why integer (int8) models matter so much.
- **No dynamic memory to spare**: TFLite-Micro famously allocates one fixed
  "tensor arena" up front and never calls `malloc` during inference.
- **Milliwatt power budgets**: coin-cell devices budget microamps.

This is why edge AI is its own discipline rather than "ML, but smaller."

## What edge AI looks like in the real world

- **Wake words / keyword spotting** — "Hey Siri", "OK Google": a ~20–100 KB
  model listens continuously on a low-power chip; the big cloud model only
  wakes up after the on-device model fires. The archetypal TinyML success.
- **Predictive maintenance** — an accelerometer node bolted to a motor runs
  a tiny classifier over vibration spectra and flags bearing wear weeks
  before failure, sending only alerts, not waveforms.
- **Vision sensors / person detection** — a camera with a ~250 KB
  "visual wake words" model reports *person present: yes/no* instead of
  streaming video — a presence sensor with camera accuracy and switch-level
  privacy.
- **Gesture and activity recognition** — wearables classifying accelerometer
  windows into walk/run/sleep/fall entirely on the wrist.
- **Smart agriculture and industrial sensing** — acoustic detection of pests,
  anomaly detection on production lines, all offline.

Note the pattern in every example: the model's job is to **turn a torrent of
raw sensor data into a tiny, meaningful signal** as close to the sensor as
physically possible.

## You don't need hardware for this course

Everything in Level 1 runs on a normal computer. Python and TensorFlow
handle training, conversion, and quantization; the microcontroller inference
runtime (TFLite-Micro) is plain C++ that compiles happily on your laptop.
When a module reaches the "put it on a real board" step, we use the ESP32 as
the reference target and note where the free
[Wokwi simulator](https://wokwi.com/) can stand in for physical hardware.
If you later want real boards, the
[Embedded Systems Mastery Path](https://sigilipelli.github.io/embedded-mastery-path/)
covers the Arduino/ESP32 fundamentals.

## Cheat sheet

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

## Exercise

1. For each of the following, decide **edge, cloud, or hybrid** and write one
   sentence justifying it using latency/privacy/power/cost: (a) a doorbell
   that announces "package detected", (b) monthly sales forecasting for a
   retail chain, (c) a hearing aid that suppresses background noise,
   (d) a chatbot answering customer emails, (e) a vibration sensor on a
   remote wind turbine.
2. A 224×224 grayscale image stored as one byte per pixel needs how much
   RAM? Could it fit on an Arduino Nano 33 BLE (256 KB RAM) alongside a
   model? What about a 96×96 version — and what does that tell you about why
   TinyML vision models use tiny input resolutions?
3. Estimate the yearly cost difference between (a) 1,000 devices calling a
   cloud vision API once per minute at $1 per 1,000 calls, and (b) the same
   devices running the model locally. (This number is why this course
   exists.)
