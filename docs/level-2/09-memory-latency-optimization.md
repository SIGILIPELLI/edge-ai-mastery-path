# Optimizing Memory & Latency on MCUs

Every prior module in this course has touched memory or latency in passing —
arena sizing, quantization, depthwise-separable convolutions, frame rate
budgets. This module pulls those threads into one systematic optimization
process: how to profile where an embedded model's time and RAM actually go,
what levers move each bottleneck, and how to make an evidence-based
optimization decision instead of guessing.

## Profile before you optimize

The single biggest mistake in embedded optimization is guessing which layer
is slow or which buffer is large instead of measuring. TFLite Micro exposes
both directly.

```cpp
// Reviewed pattern, not run here -- combines with the interpreter setup
// from Level 1 Module 07 and Module 07's vision pipeline.
Serial.printf("arena used: %u / %u bytes\n",
              (unsigned) interpreter->arena_used_bytes(), kArenaSize);

// Per-operator timing: wrap Invoke() and, if your TFLM build exposes a
// profiler (tflite::MicroProfiler), attach it at interpreter construction
// to get a per-op breakdown instead of just a whole-network number.
uint32_t t0 = micros();
interpreter->Invoke();
uint32_t total_us = micros() - t0;
Serial.printf("total invoke: %u us\n", (unsigned) total_us);
```

Without per-op timing hardware/profiler support, a manual but effective
technique is the **layer-truncation trick**: build several `.tflite`
variants of the same model truncated after each successive layer (or, more
practically, log cumulative time by wrapping `Invoke()` on a sequence of
sub-models during desktop evaluation with the Python interpreter), and
attribute the *difference* in latency between successive truncations to the
layer added.

```python
import numpy as np

def estimate_layer_latency_from_deltas(cumulative_times_ms):
    """cumulative_times_ms: list of measured total-inference times for
    versions of the model truncated after layer 1, 1-2, 1-3, etc.
    Returns the estimated per-layer cost via successive differences."""
    times = np.asarray(cumulative_times_ms)
    per_layer = np.diff(times, prepend=0.0)
    return per_layer

cum_times = [0.8, 2.1, 2.4, 5.9, 6.3]   # ms, hypothetical 5-layer network
print(estimate_layer_latency_from_deltas(cum_times))
# e.g. [0.8, 1.3, 0.3, 3.5, 0.4] -- layer 4 is clearly the bottleneck
```

## Where MCU inference time actually goes

For convolutional networks, compute is heavily concentrated in a small
number of layers — usually the ones with the most channels or largest
spatial size, not the ones deepest in the network. Reuse the MAC-counting
method from Module 02 to predict this before profiling confirms it:

```python
def conv_macs(h, w, cin, cout, k=3, depthwise=False):
    """MACs for one conv layer's forward pass over an HxW feature map."""
    if depthwise:
        return h * w * cin * k * k          # depthwise: no cout mixing
    return h * w * cin * cout * k * k        # standard/pointwise-style conv

layers = [
    ("conv1 (std, 3x3)",        conv_macs(48, 48, 1, 8, k=3)),
    ("dw1 (depthwise, 3x3)",    conv_macs(24, 24, 8, 8, k=3, depthwise=True)),
    ("pw1 (pointwise, 1x1)",    conv_macs(24, 24, 8, 16, k=1)),
    ("dw2 (depthwise, 3x3)",    conv_macs(12, 12, 16, 16, k=3, depthwise=True)),
    ("pw2 (pointwise, 1x1)",    conv_macs(12, 12, 16, 32, k=1)),
]
total = sum(m for _, m in layers)
for name, m in layers:
    print(f"{name:25s} {m:>10,} MACs  ({m/total*100:5.1f}%)")
print(f"{'total':25s} {total:>10,} MACs")
```

Running this kind of table on your own architecture before training tells
you exactly which layer to target first — usually the largest early-stage
convolution (biggest spatial dimensions) or a wide pointwise conv (largest
channel product), matching what profiling later confirms.

## The optimization lever table

Every technique from this level maps to a specific bottleneck:

| Bottleneck | Lever | Covered in |
|---|---|---|
| Flash too large | int8 quantization | Level 1 Module 05 |
| Flash too large | structured pruning | Module 05 |
| Flash too large | smaller architecture / fewer channels | Module 02 |
| RAM (arena) too large | shrink input resolution | Module 02 |
| RAM (arena) too large | reduce peak adjacent-tensor size (fewer channels at the widest point) | Module 02 |
| RAM (arena) too large | `fb_count=1` instead of double-buffering | Module 07 |
| Latency too high | depthwise-separable convs instead of standard | Module 02 |
| Latency too high | lower input resolution (compute scales ~quadratically with side length) | Module 02 |
| Latency too high | reduce sampling/inference rate, add temporal smoothing | Module 08 |
| Accuracy lost to quantization | quantization-aware training | Module 06 |
| Accuracy lost to size reduction | knowledge distillation from a larger teacher | Module 05 |

## A worked optimization pass

Say profiling shows: arena = 110 KB (over an 80 KB target), latency = 60 ms
at 5fps sampling (fine), flash = 340 KB (over a 256 KB target). Two
independent budgets are blown, so treat them separately:

```python
def project_arena_after_resolution_change(current_arena_kb, old_side, new_side):
    """Rough scaling law: for the same architecture, most activation
    tensors scale with side_length^2 (spatial dims), so a resolution cut
    approximately scales the arena by the square of the ratio -- a useful
    first estimate to check against measurement, not a substitute for it."""
    ratio = (new_side / old_side) ** 2
    return current_arena_kb * ratio

projected = project_arena_after_resolution_change(110, old_side=96, new_side=64)
print(f"projected arena at 64x64: {projected:.1f} KB (target: 80 KB)")
```

`96 -> 64` projects roughly `110 * (64/96)**2 ≈ 49 KB` — comfortably under
budget, at the cost of whatever accuracy the resolution drop costs (measure
it, per Module 02's tradeoffs). For the flash overage, structured pruning
plus quantization (Modules 05-06, applied together) is the standard
combination — prune first, quantize last, since quantization is the final
lossy step you want applied to the smallest possible model.

## Edge-AI tradeoffs

**Optimize for the actual bottleneck, not the easiest lever.** Quantizing
harder when flash was never the constraint wastes engineering effort;
always profile (arena, flash, latency) before choosing which lever to pull.

**Global levers vs. targeted levers.** Shrinking input resolution improves
every downstream layer's cost simultaneously but touches accuracy broadly;
pruning specific layers (informed by the MAC-count table) is more surgical
but requires more analysis per layer.

**One-time cost vs. recurring cost.** Structured pruning and QAT cost
engineering time once; a permanently lower sampling rate or resolution
costs a little accuracy or responsiveness on every single inference for the
life of the product — evaluate which kind of cost your project can better
absorb.

**Measured vs. estimated numbers.** MAC-count tables and scaling-law
projections (like `project_arena_after_resolution_change`) are useful for
triage and prioritization, but only real profiling
(`arena_used_bytes()`, `micros()` around `Invoke()`, build-output flash
percentages) tells you whether an optimization actually worked.

## Cheat sheet

| Metric | How to measure | Typical target for a small MCU |
|---|---|---|
| Flash usage | build tool output ("Sketch uses…") | fits with room for OTA/other app code |
| RAM / arena | `interpreter->arena_used_bytes()` | fits alongside WiFi/app buffers, Level 1 Module 07 |
| Latency | `micros()`/`millis()` around `Invoke()` | matches your sampling-rate budget (Module 08) |
| Compute distribution | MAC-count table per layer, or truncation-delta profiling | identifies which layer to target first |
| Resolution scaling law | arena/compute scale ~ `(new_side/old_side)^2` | first-pass estimate before remeasuring |
| Combine techniques in order | prune/distill -> quantize (QAT or PTQ) last | smallest model gets the final lossy step |

## Exercise

1. Implement `conv_macs` and build a MAC-count table for an architecture of
   your own design (5+ layers, mixing standard and depthwise-separable
   convs). Identify the single largest contributor and propose one concrete
   change to reduce it.
2. Implement `estimate_layer_latency_from_deltas` on a synthetic cumulative-
   time list of your choosing and confirm it correctly attributes cost to
   the layer you intended to be the bottleneck.
3. Using `project_arena_after_resolution_change`, compute projected arena
   size for resolution changes `96->80`, `96->64`, and `96->48`, starting
   from a measured 110 KB. At what resolution does the projection cross
   below a 50 KB target?
4. Take a model design from an earlier module (Module 02's classifier or
   Module 08's person detector) and write a short optimization plan: which
   metric is over budget (assume numbers of your choosing), which lever(s)
   from the table you'd apply first, and what you'd re-measure afterward to
   confirm the fix worked.
