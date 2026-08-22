# Model Pruning & Knowledge Distillation

Quantization (Level 1 Module 05) shrinks a model by changing *how* each
number is stored. Pruning and distillation shrink a model by changing *how
many* numbers there are in the first place — removing weights that
contribute little, or training a small "student" network to mimic a larger
"teacher." Both are complementary to quantization: a pruned, distilled
model still gets quantized afterward for its final int8 deployment size.
This module builds magnitude pruning and a basic soft-label distillation
loop with numpy, and measures the actual sparsity/size/accuracy tradeoffs.

## Magnitude pruning: the 80/20 of model compression

The simplest and most robust pruning method is **magnitude pruning**: after
training, zero out the weights with the smallest absolute value, on the
theory that a weight near zero contributes almost nothing to the network's
output. Neural networks tolerate this surprisingly well because they're
heavily overparameterized relative to what any single task needs.

```python
import numpy as np

def magnitude_prune(weights, sparsity):
    """Zero out the smallest-magnitude `sparsity` fraction of weights.
    weights: any-shape numpy array. sparsity: float in [0, 1)."""
    flat = np.abs(weights).flatten()
    if sparsity <= 0:
        return weights.copy()
    threshold = np.percentile(flat, sparsity * 100)
    mask = np.abs(weights) > threshold
    return weights * mask

# Demonstration on a random weight matrix
rng = np.random.default_rng(0)
w = rng.normal(0, 1, size=(64, 64))
w_pruned_50 = magnitude_prune(w, 0.50)
w_pruned_90 = magnitude_prune(w, 0.90)
print("actual sparsity @ 50% target:", np.mean(w_pruned_50 == 0))
print("actual sparsity @ 90% target:", np.mean(w_pruned_90 == 0))
```

## Structured vs. unstructured pruning — and why MCUs need the former

**Unstructured** pruning (what `magnitude_prune` does above) zeros
individual weights anywhere in a matrix — it maximizes achievable sparsity
for a given accuracy loss, but the result is a matrix full of scattered
zeros. Generic hardware (CPUs, GPUs, most microcontrollers) cannot skip
those zeros during a standard dense matmul without specialized sparse
matrix libraries, which most MCU toolchains don't have — so an unstructured-
pruned model is usually **not actually smaller or faster on-device**, only
smaller if you bother to encode it in a sparse format.

**Structured** pruning removes whole structural units — entire output
channels of a conv layer, entire neurons of a dense layer — so the resulting
model is a genuinely smaller dense network that any inference engine runs
as-is.

```python
def structured_prune_channels(weight, sparsity, axis=0):
    """Remove whole channels (rows along `axis`) ranked by their overall
    L2 norm, rather than individual elements. weight: e.g. (out_channels, in_channels).
    Returns the pruned weight AND the kept-channel indices (needed to also
    prune the corresponding bias and the next layer's input channels)."""
    norms = np.linalg.norm(weight, axis=tuple(a for a in range(weight.ndim) if a != axis))
    n_channels = weight.shape[axis]
    n_keep = max(1, int(round(n_channels * (1 - sparsity))))
    keep_idx = np.argsort(norms)[-n_keep:]
    keep_idx.sort()
    pruned = np.take(weight, keep_idx, axis=axis)
    return pruned, keep_idx

w_conv = rng.normal(0, 1, size=(32, 16, 3, 3))  # (out_ch, in_ch, kh, kw)
pruned_w, kept = structured_prune_channels(w_conv, sparsity=0.5, axis=0)
print("original shape:", w_conv.shape, "-> pruned shape:", pruned_w.shape)
```

Structured pruning genuinely shrinks the model (fewer channels means less
flash *and* less compute *and* less activation RAM), but costs more accuracy
per unit of sparsity than unstructured pruning does, since it removes
entire feature detectors rather than individually-unimportant connections.
This is why, for microcontroller deployment specifically, structured
pruning (or training a smaller architecture directly) is almost always
preferred over unstructured, despite unstructured's better sparsity/accuracy
curve on paper.

## Iterative pruning and fine-tuning

Pruning a large fraction of weights in one shot damages accuracy more than
necessary. The standard practice is **iterative pruning**: prune a modest
amount, fine-tune (continue training) to let the remaining weights
compensate, then prune again.

```python
def iterative_pruning_schedule(final_sparsity, n_steps):
    """A common schedule: cubic ramp-up, matching the widely-used
    'polynomial decay' pruning schedule -- most sparsity added early,
    tapering off, giving the model time to recover between each step."""
    steps = np.arange(n_steps + 1)
    frac = steps / n_steps
    return final_sparsity * (1 - (1 - frac) ** 3)

schedule = iterative_pruning_schedule(final_sparsity=0.8, n_steps=5)
for step, s in enumerate(schedule):
    print(f"step {step}: target sparsity = {s:.2f}")
```

Each step in a real pipeline would: apply `magnitude_prune` (or the
structured variant) at that step's target sparsity, run a handful of
fine-tuning epochs on the training set, then move to the next step —
described here in prose since it requires a Keras/TF training loop this
course's disk budget can't install.

## Knowledge distillation: training a small model to imitate a large one

Distillation trains a compact **student** network not just against the
hard ground-truth labels, but against the **soft probability distribution**
a larger, more accurate **teacher** network produces. The intuition: a
teacher's full softmax output ("70% cat, 25% dog, 5% fox") carries more
information than the one-hot label ("cat"), because the runner-up
probabilities encode which classes the teacher considers *similar* — this
is often called "dark knowledge."

```python
def softmax_with_temperature(logits, temperature=1.0):
    """Higher temperature -> softer, more spread-out probability distribution,
    revealing more of the relative ordering among non-target classes."""
    scaled = logits / temperature
    scaled -= scaled.max(axis=-1, keepdims=True)  # numerical stability
    exp = np.exp(scaled)
    return exp / exp.sum(axis=-1, keepdims=True)

def distillation_loss(student_logits, teacher_logits, true_labels, temperature=3.0, alpha=0.5):
    """Weighted sum of: (1) cross-entropy vs. the soft teacher distribution
    at raised temperature, and (2) standard cross-entropy vs. hard labels.
    alpha weights how much to trust the teacher vs. the ground truth."""
    soft_teacher = softmax_with_temperature(teacher_logits, temperature)
    soft_student = softmax_with_temperature(student_logits, temperature)
    # Cross-entropy between two distributions (soft targets)
    soft_loss = -np.sum(soft_teacher * np.log(soft_student + 1e-12), axis=-1).mean()

    hard_probs = softmax_with_temperature(student_logits, temperature=1.0)
    n = len(true_labels)
    hard_loss = -np.log(hard_probs[np.arange(n), true_labels] + 1e-12).mean()

    return alpha * (temperature ** 2) * soft_loss + (1 - alpha) * hard_loss
```

The `temperature ** 2` scaling factor is the standard correction from the
original distillation paper — it keeps the soft-loss gradient magnitude
comparable to the hard-loss gradient magnitude as temperature changes, so
`alpha` behaves as an actual mixing weight rather than something you have to
re-tune every time you change `temperature`.

```python
# Toy demonstration with fabricated logits for 3 classes, batch of 4
rng = np.random.default_rng(1)
teacher_logits = rng.normal(0, 3, size=(4, 3))   # confident teacher
student_logits = rng.normal(0, 1, size=(4, 3))   # less confident student
true_labels = np.array([0, 1, 2, 0])

loss = distillation_loss(student_logits, teacher_logits, true_labels, temperature=3.0, alpha=0.7)
print("distillation loss:", loss)
```

A full training pipeline would compute this loss's gradient with respect to
the student's weights every batch — exactly like standard supervised
training, just with a richer target signal. This course cannot run that
loop without TensorFlow/PyTorch installed, but the loss function itself
above is exact and runnable.

## Edge-AI tradeoffs

**Unstructured sparsity vs. actual on-device speedup.** High unstructured
sparsity (90%+) looks dramatic in a chart but delivers zero flash or latency
benefit on a standard TFLite Micro interpreter, which has no sparse-matmul
kernel — always confirm your target runtime can exploit sparsity before
counting on it.

**Structured pruning amount vs. accuracy floor.** Every removed channel is a
removed feature detector; past a task-dependent point (often 30-60% of
channels for small MCU models), accuracy degrades sharply rather than
gracefully — iterative pruning with fine-tuning pushes that floor lower but
doesn't eliminate it.

**Distillation temperature vs. information transferred.** Very low
temperature (near 1) makes the teacher's output nearly one-hot, discarding
the dark-knowledge signal; very high temperature over-flattens it toward
uniform, diluting the signal in noise. Values of 2-5 are typical starting
points.

**Pruning+distillation combined vs. engineering time.** Stacking iterative
pruning, distillation, and quantization together typically yields the
smallest, most accurate model — but each technique adds hyperparameters and
training runs, and for a simple task a smaller architecture trained fresh
may reach the same size/accuracy point with far less engineering effort.

## Cheat sheet

| Technique | What it removes | On-MCU benefit |
|---|---|---|
| Unstructured magnitude pruning | individual weights | flash only, and only with sparse storage/runtime support |
| Structured channel pruning | entire channels/neurons | flash + compute + RAM, works on any runtime |
| Iterative pruning | -- | recovers more accuracy than one-shot pruning at same sparsity |
| Knowledge distillation | -- | lets a small "from-scratch" architecture reach closer to teacher accuracy |
| `temperature` (distillation) | -- | higher = softer targets = more dark knowledge, typically 2-5 |
| `alpha` (distillation) | -- | weight on soft-teacher loss vs. hard-label loss |
| Order of operations | prune/distill first, quantize last | quantization further shrinks whatever comes out |

## Exercise

1. Implement `magnitude_prune` and, on a `(128, 128)` random Gaussian
   matrix, plot (or tabulate) achieved sparsity vs. the Frobenius-norm
   reconstruction error `||w - w_pruned||` at sparsity levels
   `[0.1, 0.3, 0.5, 0.7, 0.9, 0.95]`. At what point does error start growing
   noticeably faster?
2. Implement `structured_prune_channels` on a `(32, 16, 3, 3)` conv weight
   tensor and confirm the returned `keep_idx` correctly selects the
   highest-L2-norm channels by manually checking two or three.
3. Implement `distillation_loss` and `softmax_with_temperature`; verify that
   raising `temperature` from 1 to 10 visibly flattens
   `softmax_with_temperature(logits, temperature)` for a fixed logits array
   (print the resulting distributions and compare their entropy).
4. Using `iterative_pruning_schedule`, generate schedules for `n_steps=3`
   and `n_steps=10` at `final_sparsity=0.9` and compare how quickly each
   ramps up. Explain in a sentence why more steps generally preserves more
   accuracy for the same final sparsity.
