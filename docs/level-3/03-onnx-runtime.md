# ONNX Runtime for Edge Inference

Every model this course has deployed so far has gone through TensorFlow
Lite. That's a reasonable default, but it's not the only inference
runtime, and it's not always the right one — a lot of production models
start life in PyTorch, and converting PyTorch straight to TFLite is
notoriously lossy. **ONNX** (Open Neural Network Exchange) is a
vendor-neutral model format that PyTorch, scikit-learn, and most other
training frameworks can export to directly, and **ONNX Runtime (ORT)** is
a small, dependency-light C++ inference engine (with Python, C, and Java
bindings) that runs those graphs — including on constrained devices, via
its `onnxruntime` pip package or a cross-compiled minimal build. This
module builds and runs a real ONNX graph and measures it, all of which was
actually executed while writing this lesson (not left as an exercise).

## Verified on this machine

Everything in this module ran, with real output, using:

```
pip install onnx onnxruntime
```

Both installed cleanly (`onnxruntime` is a self-contained CPU wheel with
no TensorFlow or PyTorch dependency, which is exactly why it's viable on
disk-constrained environments where those frameworks won't fit). Nothing
below is hypothetical.

## Building a tiny ONNX graph by hand

You'll usually get an `.onnx` file by exporting from PyTorch
(`torch.onnx.export`), but it's worth building one from the raw graph API
once, because it makes the format's structure — nodes, initializers
(weights), value infos (shapes) — concrete instead of a black box.

```python
import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np

# A tiny 2-layer MLP: input(1x8) -> Gemm -> ReLU -> Gemm -> Softmax
X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8])
Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])

rng = np.random.default_rng(0)
w1 = rng.standard_normal((8, 16)).astype(np.float32) * 0.1
b1 = np.zeros(16, dtype=np.float32)
w2 = rng.standard_normal((16, 4)).astype(np.float32) * 0.1
b2 = np.zeros(4, dtype=np.float32)

initializers = [
    numpy_helper.from_array(w1, name="w1"),
    numpy_helper.from_array(b1, name="b1"),
    numpy_helper.from_array(w2, name="w2"),
    numpy_helper.from_array(b2, name="b2"),
]
nodes = [
    helper.make_node("Gemm", ["input", "w1", "b1"], ["h1"]),
    helper.make_node("Relu", ["h1"], ["h1r"]),
    helper.make_node("Gemm", ["h1r", "w2", "b2"], ["h2"]),
    helper.make_node("Softmax", ["h2"], ["output"], axis=1),
]
graph = helper.make_graph(nodes, "tiny_mlp", [X], [Y], initializer=initializers)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

onnx.checker.check_model(model)   # validates shapes/types line up
onnx.save(model, "tiny_mlp.onnx")
print("model saved, checker passed")
```

Running this produced `model saved, checker passed` — `onnx.checker` is
worth calling on every graph you build or export, since it catches
mismatched tensor shapes before you ever hand the file to a runtime.

## Loading and running it with ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("tiny_mlp.onnx", providers=["CPUExecutionProvider"])
x = np.random.default_rng(0).standard_normal((1, 8)).astype(np.float32)
out = sess.run(None, {"input": x})
print("output:", out[0], "sum:", out[0].sum())
```

Actual output from this run:

```
output: [[0.24374843 0.2650039  0.23453632 0.25671133]] sum: 1.0
```

The softmax output summing to 1.0 confirms the graph executes correctly
end to end — no NaNs, no shape mismatches, a real probability
distribution out the other end.

## Execution providers: ORT's version of the delegate pattern

Module 02 covered TFLite delegates (eIQ's Neutron delegate, Ethos-U's
Vela-compiled custom op). ONNX Runtime has the same idea under a
different name: an **Execution Provider (EP)**. You pick a provider list
when creating the session, and ORT tries each one for each op in order,
falling back down the list for anything a provider doesn't claim —
structurally identical to the TFLite fallback behavior from Modules 01-02.

```python
import onnxruntime as ort
print(ort.get_available_providers())
```

On this machine that printed:

```
['CoreMLExecutionProvider', 'AzureExecutionProvider', 'CPUExecutionProvider']
```

`CPUExecutionProvider` ships in every ORT build and is the universal
fallback. On real edge targets you'd instead see hardware-specific
providers: `CoreMLExecutionProvider` (Apple Neural Engine, present here
because this is a Mac), `TensorrtExecutionProvider` / `CUDAExecutionProvider`
(NVIDIA Jetson boards), `ArmNNExecutionProvider` (Arm Ethos NPUs via the
Arm NN library), or `QNNExecutionProvider` (Qualcomm's Hexagon NPU). The
pattern from Modules 01-02 repeats exactly: request the accelerator
provider, and any op it can't handle silently runs on CPU — always check
`session.get_provider_options()` after creation to see what actually got
used, not just what you asked for.

## Benchmarking correctly

Same principle as Module 01's pipeline benchmark: separate the one-time
session-creation cost from the steady-state per-inference cost, since
`InferenceSession(...)` does graph optimization and provider negotiation
that you pay for exactly once.

```python
import time

n = 500
t0 = time.perf_counter()
for _ in range(n):
    sess.run(None, {"input": x})
t1 = time.perf_counter()
print(f"avg latency: {(t1 - t0) / n * 1000:.4f} ms")
```

Measured on this machine (Apple Silicon, CPU provider, tiny 2-layer MLP):

```
avg latency: 0.0059 ms
```

That number is only meaningful relative to itself — a 2-layer, 8x16x4 MLP
is about as small as a model gets, and this is a desktop CPU, not a
microcontroller. The point of running it live is to demonstrate the
*measurement methodology* (warm session, many repeated calls, average
outside the timing loop's own overhead), which transfers directly to
whatever real model and target hardware you actually deploy.

## Why ORT matters for edge specifically

| Reason | Detail |
|---|---|
| Framework-neutral input | PyTorch, scikit-learn (via `skl2onnx`), and most other trainers export to ONNX directly |
| Small footprint | the CPU-only Python wheel installed here pulled in only `numpy`, `protobuf`, `flatbuffers` — no TensorFlow, no PyTorch runtime dependency |
| A real "mobile/embedded" build variant | `onnxruntime` also ships a minimal-build config (`--minimal_build` at compile time) that strips the graph optimizer and op kernels down to only what your specific model needs, similar in spirit to TFLite Micro's static kernel selection |
| Same EP pattern as vendor NPU stacks | ArmNN, QNN, and TensorRT execution providers plug into the exact architecture demonstrated above with the CPU provider |

## Edge-AI tradeoffs

| Factor | TFLite / TFLite Micro | ONNX Runtime |
|---|---|---|
| Best-supported source framework | TensorFlow/Keras | PyTorch, scikit-learn, anything with an ONNX exporter |
| Microcontroller (bare-metal) support | mature (TFLite Micro, Level 1-2 of this course) | possible via minimal builds, less battle-tested on MCUs |
| Linux/apps-processor support | good | excellent, broad EP ecosystem |
| Model format | flatbuffer `.tflite` | protobuf-based `.onnx` |
| Quantization tooling | built into `tf.lite.TFLiteConverter` | separate `onnxruntime.quantization` module, similar int8 PTQ workflow |

## Exercise

Modify the `build_and_run` code above to add a third `Gemm` layer, re-run
`onnx.checker.check_model`, and confirm the output shape still matches what
`make_tensor_value_info` declares for `Y` — a mismatch here is one of the
most common export bugs when converting real PyTorch models, and catching
it with the checker before deployment costs nothing. Then print
`sess.get_provider_options()` after creating a session and compare it
against the provider list you requested, to build the habit of verifying
what actually ran rather than what you asked for.
