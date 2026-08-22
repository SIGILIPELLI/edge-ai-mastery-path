# Quantization-Aware Training

Level 1 Module 05 covered **post-training quantization (PTQ)**: train a
float32 model, then convert it to int8 afterward. That works well for most
models, but some — smaller networks, or ones with wide dynamic-range
activations — lose more accuracy than acceptable under PTQ. **Quantization-
aware training (QAT)** fixes this by simulating int8 rounding *during*
training, so the network's weights adapt to quantization error instead of
being surprised by it after the fact. This module builds the fake-
quantization math from scratch in numpy, shows exactly what QAT changes in
the training loop, and works through when the extra complexity earns its
keep.

## The core idea: fake quantization in the forward pass

QAT inserts "fake quant" nodes into the computational graph that, on the
forward pass, quantize a tensor to int8 precision and immediately
dequantize it back to float — so the *values* the network sees already
carry the rounding error int8 will eventually introduce, while the
*gradients* still flow through as if the operation were identity (this is
the **straight-through estimator**, since the true quantization operation
has zero gradient almost everywhere).

```python
import numpy as np

def fake_quantize(x, num_bits=8, sym=True):
    """Simulate int8 quantize-then-dequantize on a float32 array, returning
    a float32 array with quantization error baked in -- exactly what a
    fake-quant node produces on the forward pass."""
    qmin, qmax = -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1  # -128..127
    if sym:
        max_abs = np.max(np.abs(x)) + 1e-12
        scale = max_abs / qmax
        zero_point = 0
    else:
        x_min, x_max = x.min(), x.max()
        scale = (x_max - x_min) / (qmax - qmin) + 1e-12
        zero_point = int(round(qmin - x_min / scale))

    q = np.round(x / scale + zero_point)
    q = np.clip(q, qmin, qmax)
    dequantized = (q - zero_point) * scale
    return dequantized, scale, zero_point

rng = np.random.default_rng(0)
w = rng.normal(0, 1, size=(1000,))
w_fq, scale, zp = fake_quantize(w)
print("mean abs error from fake-quant round trip:", np.mean(np.abs(w - w_fq)))
print("scale:", scale, "zero_point:", zp)
```

## Straight-through estimator: why gradients still work

The rounding operation `np.round` has derivative 0 almost everywhere
(a staircase function) — if you used its true gradient, backpropagation
through a fake-quant node would kill essentially all gradient signal. The
**straight-through estimator (STE)** sidesteps this by defining the
backward pass as if the forward pass were the identity function (or,
in the clipped version, identity within `[qmin, qmax]` and zero outside):

```python
def fake_quantize_forward_backward(x, upstream_grad, num_bits=8):
    """Demonstrates STE explicitly: forward pass applies fake quantization;
    backward pass passes the gradient straight through, except where the
    input was clipped (those elements get zero gradient, same as a ReLU-like
    clip would)."""
    qmin, qmax = -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1
    max_abs = np.max(np.abs(x)) + 1e-12
    scale = max_abs / qmax

    q_unclipped = np.round(x / scale)
    clipped_mask = (q_unclipped >= qmin) & (q_unclipped <= qmax)

    q = np.clip(q_unclipped, qmin, qmax)
    forward_out = q * scale

    # STE: gradient passes through unchanged where NOT clipped, zero where clipped
    grad_to_x = upstream_grad * clipped_mask
    return forward_out, grad_to_x

x = rng.normal(0, 1, size=(10,))
upstream = np.ones_like(x)  # pretend d(loss)/d(fake_quant_output) = 1 everywhere
out, grad = fake_quantize_forward_backward(x, upstream)
print("forward output:", out)
print("gradient passed back:", grad)
```

This is exactly the mechanism TensorFlow's `tfmot.quantization.keras`
toolkit inserts automatically when you wrap a Keras model with
`quantize_model()` — every Conv/Dense layer's weights and activations get a
fake-quant node with STE gradients, and training proceeds with the normal
optimizer, just against a loss computed on quantization-corrupted values.

## What a QAT training step looks like (described, not run)

```python
# Described, not run -- needs tensorflow-model-optimization + TF training env.
# import tensorflow_model_optimization as tfmot
#
# base_model = keras.models.load_model("float_model.keras")
# quant_aware_model = tfmot.quantization.keras.quantize_model(base_model)
# quant_aware_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
#                            metrics=["accuracy"])
# quant_aware_model.fit(x_train, y_train, epochs=5, validation_data=(x_val, y_val))
#
# converter = tf.lite.TFLiteConverter.from_keras_model(quant_aware_model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# tflite_qat_model = converter.convert()
```

Note the pattern: QAT starts from an **already-trained float model** (rarely
from scratch), fine-tunes for a small number of additional epochs with
fake-quant nodes active, then goes through the *same* TFLite converter as
PTQ. The converter recognizes the fake-quant nodes' calibration ranges and
uses them directly, instead of needing a separate representative dataset
pass — QAT effectively bakes calibration into training.

## Measuring whether QAT is worth it

Since you already have a runnable `fake_quantize` above, you can directly
demonstrate what QAT training is optimizing against: run a small numpy
"model" (a linear layer) through ordinary gradient descent, once with
fake-quantized weights during training and once without, and compare final
loss after fake-quantizing both for evaluation.

```python
def linear_forward(x, w, use_fake_quant=False):
    w_used = fake_quantize(w)[0] if use_fake_quant else w
    return x @ w_used

def train_linear(x, y, epochs=200, lr=0.05, use_fake_quant=False, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.5, size=(x.shape[1], 1))
    for _ in range(epochs):
        w_used = fake_quantize(w)[0] if use_fake_quant else w
        pred = x @ w_used
        error = pred - y
        grad = x.T @ error / len(x)   # STE: gradient computed as if w_used == w
        w -= lr * grad
    return w

rng = np.random.default_rng(2)
x = rng.normal(0, 1, size=(200, 5))
true_w = rng.normal(0, 1, size=(5, 1))
y = x @ true_w + rng.normal(0, 0.05, size=(200, 1))

w_plain = train_linear(x, y, use_fake_quant=False)
w_qat = train_linear(x, y, use_fake_quant=True)

# Evaluate BOTH under fake-quantized weights, since that's the deployed condition
mse_plain_quantized = np.mean((x @ fake_quantize(w_plain)[0] - y) ** 2)
mse_qat_quantized = np.mean((x @ fake_quantize(w_qat)[0] - y) ** 2)
print(f"plain-trained, quantized at eval: MSE={mse_plain_quantized:.5f}")
print(f"QAT-trained, quantized at eval:   MSE={mse_qat_quantized:.5f}")
```

Run this and the QAT-trained weights typically show lower MSE once both are
evaluated under quantization — the training process nudged `w` toward
values that round more favorably, which is the entire point of QAT in one
minimal example.

## Edge-AI tradeoffs

**QAT accuracy recovery vs. training cost.** QAT typically closes most or
all of the gap PTQ leaves behind, but requires a working training pipeline,
labeled data, and extra fine-tuning epochs — overhead that plain PTQ (a
single conversion call) doesn't have.

**When QAT actually matters.** For models where PTQ already loses under
1% accuracy (common for well-behaved CNNs, as seen in Level 1 Module 05),
QAT's extra engineering cost usually isn't justified. QAT earns its keep on
smaller models, models with skip connections/attention (wider dynamic
range), or tasks where every fraction of a percent of accuracy matters
(medical, safety-adjacent).

**Per-tensor vs. per-channel quantization granularity.** Per-channel scales
(common for conv weights) fit each output channel's own range and generally
quantize much better than a single per-tensor scale for the whole layer —
this matters more, and interacts with, whichever of PTQ/QAT you choose.

**Symmetric vs. asymmetric quantization.** Symmetric (`zero_point = 0`,
used in `fake_quantize(sym=True)` above) is simpler and what most int8
weight quantization uses; asymmetric better fits activations after a ReLU
(which are never negative) by using the full int8 range on one side.

## Cheat sheet

| Concept | Summary |
|---|---|
| PTQ | quantize after training; fast, usually sufficient |
| QAT | fake-quantize during training; recovers accuracy PTQ loses |
| Fake quant node | quantize-then-dequantize in the forward pass |
| Straight-through estimator | backward pass ignores the rounding op, passes gradient through |
| Typical QAT recipe | start from trained float model, fine-tune a few epochs with fake-quant active |
| `tfmot.quantization.keras.quantize_model()` | TF's automatic QAT wrapper (described only — not installed here) |
| Symmetric quant | `zero_point=0`, used for weights |
| Asymmetric quant | nonzero `zero_point`, used for post-ReLU activations |
| Decision rule | try PTQ first; reach for QAT only if PTQ's accuracy loss is unacceptable |

## Exercise

1. Implement `fake_quantize` and run it on weight arrays of increasing
   spread (`np.random.normal(0, s)` for `s` in `[0.1, 1, 10]`). How does
   mean absolute round-trip error change with `s`, and why does that argue
   for per-channel rather than per-tensor scales when different channels
   have very different weight magnitudes?
2. Implement `fake_quantize_forward_backward` and confirm that elements of
   `x` outside `[-max_abs, max_abs]` after intentionally rescaling `scale`
   to be too small receive zero gradient, while in-range elements pass the
   upstream gradient through unchanged.
3. Run the `train_linear` comparison above with `epochs` set to `50, 200,
   1000`. Does the QAT-trained model's advantage over the plain-trained
   model (both evaluated quantized) grow, shrink, or stay roughly constant
   as training length increases? What does that suggest about how many
   fine-tuning epochs real QAT recipes need?
4. In prose, describe a scenario from your own interests (audio, vision, or
   sensor data) where you'd choose QAT over plain PTQ, and one where you
   would not — justify each using the accuracy-loss-vs-engineering-cost
   tradeoff from this module.
