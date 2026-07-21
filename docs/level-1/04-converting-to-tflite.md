# 04 · Converting to TFLite/LiteRT

Stage 2 of the pipeline: turn the Keras model from Module 03 into a
**`.tflite` flatbuffer** — the deployment format that everything downstream
(the Python interpreter, Android/iOS runtimes, and TFLite-Micro on
microcontrollers) consumes. In this module you'll run the converter, poke
at the flatbuffer's insides, run inference on it with the TFLite
interpreter in Python, and — most importantly — verify its outputs match
the original Keras model.

!!! info "TFLite or LiteRT?"
    Google renamed TensorFlow Lite to **LiteRT** ("Lite Runtime") in late
    2024. The file format (`.tflite`), the converter, and the APIs are the
    same thing; newer docs say LiteRT and newer installs may ship the
    standalone `ai-edge-litert` package. This course says "TFLite" because
    the ecosystem (and every file extension) still does.

## Why a special format at all?

A `.keras` file is a training artifact: Python-flavored, holding layer
*objects*, optimizer state, and enough flexibility to keep training. A
deployment target needs none of that — it needs a frozen, ordered list of
math operations and their weights, readable without Python. That's a
**flatbuffer**: a binary you can memory-map and read in place, with zero
parsing/unpacking step. On a microcontroller the model bytes stay in flash
and the interpreter reads weights directly from them — this
zero-copy property is why the format was chosen.

```text
sine_model.keras  ──TFLiteConverter──►  sine_model.tflite
(layers, optimizer state,               (frozen op graph + weights,
 Python objects)                         self-contained flatbuffer)
```

## Running the converter

```python
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model("sine_model.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_bytes = converter.convert()

with open("sine_model.tflite", "wb") as f:
    f.write(tflite_bytes)

print(f"TFLite model: {len(tflite_bytes)} bytes")
```

That's the whole stage — `from_keras_model()` plus `convert()`. Expect
roughly **2–3 KB**: ~1.3 KB of float32 weights plus graph structure and
metadata. During conversion, TensorFlow ops are lowered to TFLite's smaller
op set (`FULLY_CONNECTED` with fused ReLU, here); Module 06 shows why that
op list matters on microcontrollers.

## Looking inside the flatbuffer

```python
interpreter = tf.lite.Interpreter(model_path="sine_model.tflite")
interpreter.allocate_tensors()

print(interpreter.get_input_details())
print(interpreter.get_output_details())
```

Output (trimmed to what matters):

```text
[{'name': 'serving_default_...', 'index': 0, 'shape': array([1, 1]),
  'dtype': <class 'numpy.float32'>, ...}]
[{'name': 'StatefulPartitionedCall...', 'index': 9, 'shape': array([1, 1]),
  'dtype': <class 'numpy.float32'>, ...}]
```

One float32 input of shape `[1, 1]`, one float32 output of shape `[1, 1]`.
The `index` values are positions in the interpreter's tensor table — you'll
use them for every read and write.

## Inference with the TFLite interpreter

The interpreter API is lower-level than Keras — you set the input tensor,
invoke, and read the output tensor. This is deliberate: it's the exact
shape of the C++ API you'll use on-device in Module 06, so learning it in
Python is free practice.

```python
import numpy as np

in_idx  = interpreter.get_input_details()[0]["index"]
out_idx = interpreter.get_output_details()[0]["index"]

def tflite_predict(x_scalar):
    interpreter.set_tensor(in_idx, np.array([[x_scalar]], dtype=np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(out_idx)[0, 0]

print(tflite_predict(np.pi / 2))   # ≈ 1.0
print(tflite_predict(np.pi))       # ≈ 0.0
```

Note the batch dimension: shape `[1, 1]` means one sample of one feature.
Type and shape must match `input_details` exactly — passing float64 or
shape `[1]` raises an error (a kindness, compared to what mismatches do on
a microcontroller).

## Verification: Keras vs. TFLite

The stage isn't done until the numbers say so. Run the *saved test set*
from Module 03 through both models:

```python
data = np.load("sine_test_data.npz")
x_test = data["x_test"]

keras_out  = model.predict(x_test.reshape(-1, 1), verbose=0).flatten()
tflite_out = np.array([tflite_predict(v) for v in x_test])

max_diff = np.max(np.abs(keras_out - tflite_out))
print(f"max |keras - tflite| = {max_diff:.2e}")
assert max_diff < 1e-5, "conversion changed the model!"
```

Typical result: `max diff ≈ 1e-7` — floating-point dust from op fusion and
reordering, nothing more. This float-to-float conversion should be
essentially lossless; if you ever see a real gap here, the converter
rewrote something (unsupported op, fallback path) and you want to know
*before* quantization muddies the water.

!!! tip "Batch the comparison in one call"
    The interpreter can take the whole test set at once if you resize:
    `interpreter.resize_tensor_input(in_idx, [200, 1])` then
    `allocate_tensors()` again. Handy for evaluation scripts; the
    one-at-a-time loop above mirrors how the MCU will run.

## Cheat sheet

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

## Exercise

1. Convert your Module 03 model and report: `.keras` file size, `.tflite`
   file size, and max |keras − tflite| difference over the saved test set.
2. Use `interpreter.get_tensor_details()` to list every tensor in the
   flatbuffer. Identify which entries are the weight matrices of your three
   Dense layers by their shapes ((1,16), (16,16), (16,1)).
3. Write a `SineModel` Python class wrapping the interpreter with a clean
   `.predict(x_array)` method that accepts a 1-D NumPy array of any length
   and handles the reshaping/looping internally. Keep it — you'll reuse it
   in Modules 05 and 09.
4. Stretch: convert with `converter.target_spec.supported_ops =
   [tf.lite.OpsSet.TFLITE_BUILTINS]` explicitly set, reconvert, and confirm
   the output bytes are identical — then read the docs for what
   `SELECT_TF_OPS` would mean and why it's unavailable on microcontrollers.
