# Edge NPUs & Accelerators (Coral Edge TPU)

Levels 1-2 ran every model on a microcontroller's CPU: a Cortex-M core
executing CMSIS-NN kernels one multiply-accumulate at a time. That's fine
for a keyword spotter running at a few hundred kHz of audio, but it falls
apart the moment you want a camera running a real-time object detector at
30 fps. The fix isn't a faster CPU — it's a different kind of chip
entirely: a **Neural Processing Unit (NPU)**, purpose-built silicon that
does nothing but tensor math, and does it 10-100x more efficiently per
watt than a general-purpose core. This module covers what an NPU actually
is, using Google's Coral Edge TPU as the concrete example, and how the
compilation model changes once inference moves off the CPU.

## Why a separate chip beats a faster CPU

A CPU core spends most of its transistor budget on things that have
nothing to do with arithmetic: branch prediction, out-of-order execution,
cache coherency, instruction decoding. A convolution is enormous numbers
of the *same* operation (multiply, accumulate) repeated over a predictable
data access pattern. An NPU throws away everything a CPU needs for
general-purpose code and replaces it with a systolic array — a grid of
simple multiply-accumulate cells that pass partial sums to their neighbors
in lockstep, so data loaded once gets reused across many computations
without round-tripping to memory.

The Coral Edge TPU is a good teaching example because its numbers are
public and its constraints are typical of the whole NPU category:

| Property | Value |
|---|---|
| Peak throughput | 4 TOPS (int8) |
| Power draw | ~2W |
| Effective efficiency | ~2 TOPS/W |
| Supported precision | int8 only (no float) |
| Host interface | USB, PCIe, or M.2, depending on module |
| Model format | TensorFlow Lite, compiled ahead-of-time |

Compare that 2 TOPS/W to a Cortex-M7 running CMSIS-NN kernels, which lands
around 0.01-0.05 TOPS/W for int8 convolutions — the Edge TPU is roughly
two orders of magnitude more efficient *per operation*, at the cost of
being able to do only one kind of operation.

## The int8-only constraint and why it's non-negotiable

Every NPU in this class trades generality for efficiency, and the trade is
almost always the same: int8 arithmetic only. The systolic array's cells
are wired for 8-bit multiply-accumulate; there is no floating-point unit
on the chip at all. This means:

- Your model must be **fully integer-quantized** (Level 2 covered
  post-training quantization) before compilation — not "mostly int8 with
  a float fallback."
- Every op in the graph must have a **fixed, ahead-of-time-known integer
  scale**. Ops that don't (some custom layers, certain resize modes)
  simply cannot run on the accelerator.
- Any op the compiler can't map to the TPU's instruction set falls back to
  the host CPU, and that fallback is where most first-time deployments
  quietly lose all their speedup.

## Compiling a model for the Edge TPU

The Coral toolchain is a two-step, offline compilation pipeline: first
TFLite converts and fully quantizes the model (as in Level 2), then a
separate `edgetpu_compiler` binary maps the quantized graph onto Edge
TPU instructions and rewrites the ops it can't map into CPU-fallback nodes.

```python
"""
Conceptual walkthrough of the Coral compile step. This mirrors the real
edgetpu_compiler CLI's decision process; running it requires the Coral
compiler binary and hardware this environment doesn't have, so treat this
as an annotated model of what happens rather than executable code.
"""

def simulate_edgetpu_mapping(op_list, supported_ops):
    """op_list: sequence of (name, op_type) from a fully int8-quantized
    TFLite graph. supported_ops: the set of op types the Edge TPU
    compiler can map to hardware instructions (a fixed list per compiler
    version -- CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, and a limited
    set of others; things like custom ops or certain reshape patterns are
    excluded)."""
    mapped, fallback = [], []
    for name, op_type in op_list:
        if op_type in supported_ops:
            mapped.append(name)
        else:
            fallback.append(name)

    # In the real compiler, a single unsupported op doesn't just cost that
    # op -- it splits the graph into "segments" bounced between TPU and
    # CPU, and each hop pays a fixed transfer latency.
    segments = 1 if not fallback else 1 + len(fallback)
    return {
        "on_device_ops": mapped,
        "cpu_fallback_ops": fallback,
        "graph_segments": segments,
        "fully_mapped": len(fallback) == 0,
    }


supported = {"CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "ADD",
             "AVERAGE_POOL_2D", "RESHAPE"}
example_graph = [
    ("conv1", "CONV_2D"), ("dwconv1", "DEPTHWISE_CONV_2D"),
    ("resize1", "RESIZE_BILINEAR"),   # not in supported set
    ("conv2", "CONV_2D"), ("fc1", "FULLY_CONNECTED"),
]

result = simulate_edgetpu_mapping(example_graph, supported)
print(result)
# {'on_device_ops': ['conv1', 'dwconv1', 'conv2', 'fc1'],
#  'cpu_fallback_ops': ['resize1'], 'graph_segments': 2, 'fully_mapped': False}
```

That `graph_segments: 2` is the number that matters in practice. Every
segment boundary means a round trip across the USB/PCIe bus carrying
activation tensors, which at real frame rates can cost more time than the
fallback op itself would have taken on the CPU. The practical workflow is:
compile, read the compiler's op-mapping report, and if you see fallback
ops, restructure the model (swap `RESIZE_BILINEAR` for a supported resize
mode, move a custom layer to pre/post-processing on the host) rather than
accept the split.

## Measuring the speedup honestly

"NPU vs. CPU" comparisons are easy to get wrong by comparing the wrong
things — a common mistake is timing only the inference call and ignoring
the transfer cost of moving frames to and from the accelerator.

```python
import time

def benchmark_pipeline(preprocess_fn, transfer_fn, infer_fn, postprocess_fn,
                        frame, n_runs=100):
    """Times every stage separately so a 'fast' inference number doesn't
    hide a slow transfer stage -- a mistake common enough it's worth
    building the harness to prevent it structurally."""
    stages = {"preprocess": 0.0, "transfer": 0.0,
              "infer": 0.0, "postprocess": 0.0}
    for _ in range(n_runs):
        t0 = time.perf_counter()
        x = preprocess_fn(frame)
        t1 = time.perf_counter()
        x = transfer_fn(x)          # host -> accelerator memory copy
        t2 = time.perf_counter()
        y = infer_fn(x)
        t3 = time.perf_counter()
        _ = postprocess_fn(y)
        t4 = time.perf_counter()
        stages["preprocess"] += t1 - t0
        stages["transfer"] += t2 - t1
        stages["infer"] += t3 - t2
        stages["postprocess"] += t4 - t3
    return {k: (v / n_runs) * 1000 for k, v in stages.items()}  # ms/frame
```

On a USB-attached Edge TPU, `transfer` frequently costs as much as
`infer` for small models — the accelerator is so fast at the matmul that
the bottleneck moves entirely to getting bytes across the bus. This is
why Coral's PCIe and M.2 modules (which skip USB's protocol overhead)
show a bigger real-world win than the raw TOPS number suggests, and why
batching multiple frames per transfer (when latency budget allows) is a
standard optimization.

## Edge-AI tradeoffs

| Factor | CPU (CMSIS-NN) | NPU (Edge TPU class) |
|---|---|---|
| Peak efficiency | ~0.01-0.05 TOPS/W | ~1-4 TOPS/W |
| Precision flexibility | int8, int16, float32 | int8 only |
| Op coverage | anything you can write in C | fixed, compiler-defined set |
| Cold-start / setup cost | none | driver + runtime init, non-trivial |
| Best fit | small models, tight power budget, simple ops | vision models, high throughput, fixed op set |
| Failure mode when mismatched | just slow | silent CPU fallback, split-graph latency |

## Exercise

Take a TFLite int8 model you quantized in Level 2 (or reuse the keyword
spotter). List every op type in its graph (`interpreter.get_tensor_details()`
or inspect the `.tflite` with Netron), then check each one against Coral's
published supported-ops list for the current compiler version. Write down
which ops would fall back to CPU and estimate, using the `graph_segments`
idea above, how many host/accelerator boundary crossings your model would
incur — before you ever touch real hardware.
