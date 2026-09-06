# 05 · Quantization

Stage 3, and the most distinctively "edge" skill in the pipeline.
Quantization converts a model's numbers from 32-bit floats to 8-bit
integers: **4× smaller, faster on integer hardware, and required outright
by many microcontroller deployments** — at the cost of a little precision.
This module explains how a float becomes an int8, walks the two main
post-training quantization modes with the converter, and measures the real
size and accuracy impact on our sine model and on a model big enough for
the numbers to get interesting.

## How can an int8 replace a float32?

An int8 holds only the integers −128…127. The trick is an *affine mapping*
per tensor: store a float32 `scale` and an int8 `zero_point`, and encode

```text
real_value ≈ scale × (int8_value − zero_point)
```

Example: a weight tensor whose values span −2.0…+2.0 gets
`scale = 4.0/255 ≈ 0.0157`, `zero_point = 0`. Then the weight `0.83` is
stored as `round(0.83/0.0157) = 53`, and decodes back to `53 × 0.0157 =
0.832`. The rounding error (~half a scale step, here ~0.008) is
**quantization error** — small per weight, and neural networks, being
trained on noisy data, are remarkably tolerant of it.

For weights, the range is known (just look at them). For *activations* —
the values flowing between layers — the range depends on the input data,
which is why full-integer quantization needs to *see representative data*
(below).

## Mode 1: dynamic range quantization

The one-flag version — weights become int8, activations stay float at
runtime:

```python
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model("sine_model.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_dyn = converter.convert()
open("sine_model_dyn.tflite", "wb").write(tflite_dyn)
```

Size drops roughly 4× on the weights. Great for mobile CPUs; **not the one
for microcontrollers**, because compute still happens in float.

## Mode 2: full integer quantization (the MCU mode)

Everything — weights, activations, ideally inputs and outputs — becomes
int8, so inference runs entirely in integer arithmetic. To calibrate
activation ranges, you supply a **representative dataset**: a generator
yielding a few hundred typical inputs, which the converter runs through the
model while recording min/max at every tensor.

```python
import numpy as np

rng = np.random.default_rng(0)

def representative_data():
    for _ in range(200):
        x = rng.uniform(0, 2 * np.pi, size=(1, 1)).astype(np.float32)
        yield [x]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data
# Fail loudly if any op can't be integer-quantized:
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Make even the input/output tensors int8 (pure-integer device path):
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_int8 = converter.convert()
open("sine_model_int8.tflite", "wb").write(tflite_int8)
```

!!! warning "The representative dataset must look like real inputs"
    Calibrate with x in [0, 2π] and the model quantizes well for [0, 2π].
    Calibrate with the wrong range (say [−1, 1]) and activations outside it
    get clipped — accuracy craters *only on real data*, which is a
    maddening bug to find later. Use actual training samples whenever you
    have them.

## Running the int8 model: scale and zero-point in your code

With int8 input/output tensors, *you* now do the affine mapping at the
edges — exactly what your C code will do on the MCU:

```python
interp = tf.lite.Interpreter(model_path="sine_model_int8.tflite")
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]

in_scale, in_zp = inp["quantization"]     # e.g. (0.0246, -128)
out_scale, out_zp = out["quantization"]

def int8_predict(x_float):
    q = np.round(x_float / in_scale + in_zp).astype(np.int8)
    interp.set_tensor(inp["index"], q.reshape(1, 1))
    interp.invoke()
    q_out = interp.get_tensor(out["index"])[0, 0]
    return (int(q_out) - out_zp) * out_scale

print(int8_predict(np.pi / 2))    # ≈ 1.0 (within ~0.01)
```

## Measuring the damage: real numbers

Evaluate all three artifacts on the *same saved test set* from Module 03:

```python
data = np.load("sine_test_data.npz")
x_test, y_test = data["x_test"], data["y_test"]

keras_mae = np.mean(np.abs(
    model.predict(x_test.reshape(-1, 1), verbose=0).flatten() - y_test))
int8_mae = np.mean(np.abs(
    np.array([int8_predict(v) for v in x_test]) - y_test))
print(f"keras MAE={keras_mae:.4f}  int8 MAE={int8_mae:.4f}")
```

Typical results for this course's models:

| Model | Format | Size | Test metric |
|---|---|---|---|
| Sine MLP (321 params) | float32 `.tflite` | ~2.7 KB | MAE ≈ 0.085 |
| Sine MLP | full-int8 `.tflite` | ~2.5 KB* | MAE ≈ 0.086 |
| MNIST-subset CNN (~20K params) | float32 | ~84 KB | acc ≈ 97.8% |
| MNIST-subset CNN | full-int8 | ~24 KB | acc ≈ 97.6% |

*The sine model is so small that flatbuffer overhead (~2 KB of metadata)
hides the 4× weight shrink — 321 weights are only ~1.3 KB to begin with.
The CNN row shows the true story: **~3.5× smaller, 0.2 points of accuracy
lost.** That trade — huge shrink, negligible loss — is the norm for
post-training int8 quantization on well-behaved models, and it's why int8
is simply the default in TinyML.

## Cheat sheet

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

## How It Actually Works

**Why the affine mapping `real ≈ scale × (q − zero_point)` is the entire
mathematical content of int8 quantization.** An int8 spans 256 discrete
integers; a float32 weight tensor spans a continuous range of real
values. The affine map is a linear rescaling that stretches the 256
integer steps to cover exactly the tensor's observed value range: `scale`
sets how many real units one integer step represents, and `zero_point`
shifts the integer origin so that real zero (often meaningful — the
resting value of a ReLU'd activation, or a centered weight) lands exactly
on a representable integer rather than being approximated. Every other
piece of the quantization workflow — calibration, per-channel scales,
requantization after a matmul — is a variation on correctly computing or
applying this one formula; there is no additional mathematical machinery
hiding underneath.

**Why full-integer quantization needs a representative dataset but
weight quantization does not.** A weight tensor's values are fully known
the instant training finishes — `scale`/`zero_point` can be computed
directly from `min(weights)`/`max(weights)` with no additional data. An
activation tensor's range, by contrast, depends on what inputs flow
through the network, which isn't knowable from the weights alone; the
converter must actually run example inputs through the float model and
record the min/max seen at every intermediate tensor. This is precisely
why `representative_dataset` exists as a generator the converter calls
repeatedly during conversion, not a one-time argument: it's performing a
calibration pass, observing real activation statistics, before it can
compute the scale/zero-point pairs full-integer quantization requires for
every tensor in the graph, not just the weights.

**Why the sabotage exercise's miscalibrated range causes accuracy to
collapse only on real data, not on the representative set used to
calibrate.** If calibration observes inputs from `[0,1]` and computes a
scale/zero_point sized to that narrow range, then a real inference input
from, say, x=5 (well inside the true `[0,2π]` operating range but far
outside what calibration saw) produces an activation value that the
narrow scale cannot represent — it saturates at the int8 range's edge
(clipped to 127 or -128) rather than mapping proportionally. The model
was never wrong about *its own calibration data* — accuracy against
inputs drawn from `[0,1]` would look fine — the failure is entirely a
mismatch between the range calibration observed and the range production
inputs actually occupy, which is exactly why this class of bug is
"maddening to find later": every metric computed during development, if
computed against the same narrow calibration-adjacent data, looks
perfectly healthy.

## Exercise

1. Produce all three versions of the sine model (float, dynamic, full-int8)
   and build the size/MAE table from your own runs.
2. Print the input tensor's `(scale, zero_point)` and hand-compute the int8
   encoding of x = π. Verify against what your `int8_predict` sends.
3. Sabotage experiment: calibrate the representative dataset with
   `uniform(0, 1)` instead of `(0, 2π)` and re-measure MAE across the full
   range. Plot predictions vs. sin(x) — where exactly does it fail, and why?
4. Train a small CNN on MNIST (Keras built-in dataset; 2 conv layers,
   ~20K params, 3 epochs is fine), then full-int8 quantize it with 200
   training images as the representative set. Report float vs. int8: file
   size and test accuracy. You have now run the real pipeline on a real
   dataset.
