---
description: "Image Classification on Microcontrollers — Running an image classifier on a microcontroller is a different sport from running one on a phone or a GPU. A…"
---

# Image Classification on Microcontrollers

Running an image classifier on a microcontroller is a different sport from
running one on a phone or a GPU. A Cortex-M4 at 80 MHz with 256 KB of RAM
cannot hold a ResNet's activations, let alone its weights. This module covers
the architecture choices, preprocessing, and memory accounting that make
image classification feasible in that budget — using MobileNet-style
depthwise-separable convolutions as the reference architecture and a
from-scratch numpy convolution to make the arithmetic concrete.

## Why depthwise-separable convolutions exist

A standard 3x3 convolution over `Cin` input channels producing `Cout` output
channels needs `3 * 3 * Cin * Cout` multiply-accumulates per output pixel.
For `Cin=32, Cout=64` that's 18,432 MACs per pixel — multiplied across every
pixel in the feature map, this is where all the compute (and energy) in a
CNN goes.

A **depthwise-separable** convolution splits this into two cheaper steps:

1. **Depthwise**: one 3x3 filter per input channel (no mixing across
   channels) — `3 * 3 * Cin` MACs per pixel.
2. **Pointwise**: a 1x1 convolution that mixes channels — `Cin * Cout` MACs
   per pixel.

Total: `9*Cin + Cin*Cout` vs `9*Cin*Cout` — for the numbers above, 2,336 vs
18,432 MACs, roughly **8x cheaper**. This is the single idea behind
MobileNet and is why virtually every microcontroller vision model is built
from depthwise-separable blocks instead of standard convolutions.

```python
import numpy as np

def depthwise_conv2d(x, kernels, stride=1):
    """x: (H, W, C). kernels: (C, kh, kw). Returns (H', W', C), 'valid' padding."""
    H, W, C = x.shape
    _, kh, kw = kernels.shape
    out_h = (H - kh) // stride + 1
    out_w = (W - kw) // stride + 1
    out = np.zeros((out_h, out_w, C))
    for c in range(C):
        for i in range(out_h):
            for j in range(out_w):
                patch = x[i*stride:i*stride+kh, j*stride:j*stride+kw, c]
                out[i, j, c] = np.sum(patch * kernels[c])
    return out

def pointwise_conv2d(x, weight):
    """x: (H, W, Cin). weight: (Cin, Cout). Returns (H, W, Cout) -- a 1x1 conv is a matmul."""
    H, W, Cin = x.shape
    Cout = weight.shape[1]
    return x.reshape(H * W, Cin).dot(weight).reshape(H, W, Cout)
```

## A minimal MCU-friendly classifier, layer by layer

A workable microcontroller image classifier for a small task (say, 96x96
grayscale, 2-3 classes) typically looks like:

```
Input (96, 96, 1)
-> Conv2D 3x3, 8 filters, stride 2      (standard conv is fine for the *first* layer -- Cin is only 1)
-> [Depthwise 3x3 + Pointwise 1x1 -> 16 channels], stride 2
-> [Depthwise 3x3 + Pointwise 1x1 -> 32 channels], stride 2
-> GlobalAveragePooling2D
-> Dense -> n_classes (softmax)
```

Note the pattern: use a regular convolution only where `Cin` is tiny (the
very first layer, `Cin=1` or `3`), and depthwise-separable everywhere the
channel count is larger, since that's where the savings compound.

```python
# Described, not run -- needs a Keras/TF training environment.
# model = keras.Sequential([
#     keras.layers.Input(shape=(96, 96, 1)),
#     keras.layers.Conv2D(8, 3, strides=2, activation="relu"),
#     keras.layers.DepthwiseConv2D(3, strides=2, activation="relu"),
#     keras.layers.Conv2D(16, 1, activation="relu"),          # pointwise
#     keras.layers.DepthwiseConv2D(3, strides=2, activation="relu"),
#     keras.layers.Conv2D(32, 1, activation="relu"),          # pointwise
#     keras.layers.GlobalAveragePooling2D(),
#     keras.layers.Dense(n_classes, activation="softmax"),
# ])
```

## Preprocessing that matches the camera, not the dataset

Training-set preprocessing (from a folder of JPEGs) and on-device
preprocessing (from a raw camera sensor buffer) must produce numerically
identical tensors, or accuracy silently degrades in the field. The two most
common mismatches:

- **Color format**: many MCU camera modules (like the OV2640 on ESP32-CAM)
  output **RGB565** or YUV422, not RGB888. Convert consistently on both
  sides, or convert to grayscale on both sides if the model doesn't need
  color.
- **Resize method**: OpenCV's bilinear resize during training and a nearest-
  neighbor crop-and-subsample on-device (cheaper, but different) produce
  different pixel values for the same image. Pick one method, or verify
  empirically that the discrepancy doesn't move predictions.

```python
def rgb565_to_grayscale(pixel_bytes):
    """Decode a buffer of RGB565 pixels (2 bytes each, little-endian) to
    single-channel grayscale, matching what firmware would do on-device."""
    raw = np.frombuffer(pixel_bytes, dtype="<u2")
    r = ((raw >> 11) & 0x1F) << 3
    g = ((raw >> 5) & 0x3F) << 2
    b = (raw & 0x1F) << 3
    # ITU-R BT.601 luma weights, integer-friendly
    gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
    return gray
```

## Memory accounting: will it even fit?

Before training anything, do the arithmetic that tells you whether a model
family is plausible for your target chip. Two numbers matter:

- **Flash (model size)**: the quantized `.tflite` file, loaded once.
- **RAM (arena size)**: the largest sum of simultaneously-live activation
  tensors during inference (TFLite Micro pre-allocates one fixed "tensor
  arena" for this).

```python
def estimate_activation_bytes(shape, dtype_bytes=1):
    """int8 activations: 1 byte/element. float32: 4."""
    return int(np.prod(shape)) * dtype_bytes

# Rough peak-RAM estimate for the toy network above at each stage:
stages = {
    "input_96x96x1":      (96, 96, 1),
    "after_conv1_48x48x8": (48, 48, 8),
    "after_dw1_24x24x8":   (24, 24, 8),
    "after_pw1_24x24x16":  (24, 24, 16),
}
for name, shape in stages.items():
    print(name, estimate_activation_bytes(shape), "bytes (int8)")
```

The tensor arena needs roughly the size of the *two largest adjacent*
tensors (input to a layer + its output), not the sum of all of them — TFLite
Micro reuses buffers once a tensor's consumers are done with it. A common
starting arena size for a 96x96 grayscale MobileNet-style classifier is
40-80 KB; measure it exactly with `interpreter.arena_used_bytes()` once you
have a real converted model, rather than trusting an estimate on a real
deployment.

## Edge-AI tradeoffs

**Input resolution vs. accuracy vs. RAM.** Halving image side length (say
96->48) cuts activation memory 4x and compute 4x, but usually costs several
accuracy points on any task with fine detail. Person/object presence
detection tolerates small inputs (96x96 or even 64x64) far better than
fine-grained classification (dog breeds) does.

**Grayscale vs. RGB.** Dropping color triples memory savings on the input
tensor and every early-layer activation, and many "is there a person/object"
style tasks lose little accuracy from it — but color-dependent tasks (ripe
vs. unripe fruit) obviously cannot.

**Standard conv vs. depthwise-separable.** Depthwise-separable saves compute
and often parameters, but has slightly less representational power per
layer — MobileNet compensates with more layers. On a small dataset, going
too deep can overfit before it under-computes.

**Global average pooling vs. Flatten+Dense.** GAP throws away spatial detail
but eliminates a huge Dense layer's worth of parameters (`H*W*C*n_classes`
weights) — on a microcontroller, this is almost always the right trade.

## Cheat sheet

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

## How It Actually Works

**Why the depthwise/pointwise split costs less without changing the
receptive field.** A standard `3×3` conv computes, per output pixel and
output channel, a weighted sum over a `3×3×Cin` volume — every output
channel mixes every input channel *and* spatial neighborhood at once,
which is why its cost multiplies `Cin × Cout`. Splitting it factors that
one big mixing operation into two smaller ones done in sequence: the
depthwise step only mixes spatially (each channel convolved independently,
no cross-channel mixing — cost scales with `Cin`, not `Cin×Cout`), and the
pointwise step only mixes channels (a `1×1` conv is literally a matmul
across the channel axis, no spatial extent — cost scales with `Cin×Cout`
but with a `1×1` kernel instead of `3×3`, so no factor of 9). The
receptive field after both steps still covers the same `3×3` spatial
neighborhood and touches every input channel — the network loses some
*joint* spatial-and-channel expressiveness per layer (it can't have a
weight that depends on both a specific spatial offset and a specific
channel pairing in one step), which MobileNet compensates for with more
layers, exactly as the module notes.

**Why global average pooling, not Flatten+Dense, is almost mandatory on an
MCU.** `Flatten()` followed by `Dense(n_classes)` on a `24×24×32` feature
map creates a weight matrix of `24×24×32×n_classes` parameters — for even
3 classes that's `24×24×32×3 = 55,296` weights, roughly 55 KB in int8,
often larger than every convolutional layer in the network combined.
`GlobalAveragePooling2D` instead computes one scalar per channel (the
mean over all `24×24` spatial positions), collapsing the feature map to a
`32`-length vector before the Dense layer, so the classifier head shrinks
to `32×n_classes` weights — a reduction of nearly three orders of
magnitude here. The tradeoff is real: GAP assumes *that a channel's
average activation over the whole image* is sufficient signal, discarding
exactly where in the image that activation occurred — fine for "is this
class present," much weaker for anything needing spatial layout (e.g.
counting or localization).

**Why RGB565 decoding must match bit-for-bit, not just visually.**
RGB565 packs a pixel into 16 bits: 5 bits red, 6 bits green, 5 bits blue —
chosen unevenly because the human eye is more sensitive to green. Shifting
`(raw >> 11) & 0x1F` extracts the 5-bit red field but leaves it in the
range 0–31, not 0–255; the `<< 3` left-shift is not decoration, it's
restoring the missing 3 low-order bits (approximately, by zero-padding)
so the value occupies the full 8-bit dynamic range a training-time RGB888
image would have used. Skip that shift, or use a different bit-width
assumption than the actual sensor's byte order (endianness — `<u2` here
is little-endian, matching common camera DMA output), and every decoded
pixel is systematically too dark or shifted by exactly the missing bits —
an error small enough to look "roughly right" on inspection but large
enough to shift every activation in the first conv layer.

## 🔀 Related lessons on other tracks

- [AI/ML — 05 · CNNs for Image Classification](https://sigilipelli.github.io/ai-ml-mastery-path/level-2/05-cnns-images/)

## Exercise

1. Implement `depthwise_conv2d` and `pointwise_conv2d` above, run them on a
   random `16x16x8` input, and confirm the output shapes match a standard
   `Conv2D(8->16, 3x3)`'s output shape. Then count and compare the actual
   multiply-accumulate operations performed by each path.
2. Using `estimate_activation_bytes`, build a table of activation sizes for
   an input pipeline of your choice at three resolutions (96x96, 64x64,
   48x48) and estimate how peak RAM changes.
3. Implement `rgb565_to_grayscale` and test it against a hand-computed
   example: pick an RGB565 value, decode it by hand into R/G/B, and confirm
   your function's output matches.
4. Describe why a model trained on `Flatten() -> Dense(256) -> Dense(n)` on
   a 24x24x32 feature map would likely fail to fit on a device with 128 KB
   flash, with the actual parameter-count arithmetic to back up your answer.
