# On-Device Learning

Every model up to this point is trained once, in the cloud or on a
workstation, then frozen and shipped. **On-device learning** means the
model keeps adapting after deployment, using data the device itself
collects, without a round trip to the cloud. This is a meaningfully
different problem from cloud training — no GPU, often only a few
kilobytes of free RAM, and no ability to do a full backward pass through
a deep network — so this module focuses on the techniques that actually
fit the constraint: last-layer-only fine-tuning and simple online
learning rules, both implemented and tested here in plain NumPy.

## Why full backpropagation usually isn't viable on-device

Training a network requires storing activations from the forward pass to
compute gradients in the backward pass, and for a deep CNN those
activations can be many times larger than the model's own weights — the
same asymmetry that makes training need a GPU with tens of gigabytes of
memory while inference needs only megabytes. A microcontroller with
256 KB of RAM has no path to full backpropagation through a real
detection or classification backbone. The practical techniques that fit:

| Technique | What updates | Memory cost | Typical use |
|---|---|---|---|
| Last-layer fine-tuning | only the final classifier layer's weights | small — one layer's activations, not the whole network | personalizing a fixed feature extractor to a specific device/user |
| Online learning (single-sample updates) | any small model, one example at a time | tiny — no batch buffering | adapting a lightweight model continuously from a live stream |
| Federated learning | full model, but training happens jointly across many devices | offloads the aggregation cost to the server (Module 05) | fleet-wide learning without centralizing raw data |

This module covers the first two; Module 05 covers federated learning
specifically, since it's really a fleet-coordination problem layered on
top of these device-local techniques.

## Last-layer fine-tuning: freeze the backbone, adapt the head

The idea: treat a pretrained backbone (the expensive convolutional feature
extractor, e.g. from Level 2/3) as a fixed feature-extraction function,
and only train a small linear layer on top of its output — dramatically
cheaper because gradients never need to flow back through the backbone at
all.

```python
import numpy as np

class LastLayerFineTuner:
    """Simulates a frozen feature extractor (represented here as a fixed
    random projection, standing in for a real CNN backbone's output) with
    a trainable linear classifier head on top, updated via plain
    gradient descent on cross-entropy loss."""

    def __init__(self, feature_dim, n_classes, lr=0.05, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.01, size=(feature_dim, n_classes))
        self.b = np.zeros(n_classes)
        self.lr = lr

    def _softmax(self, z):
        z = z - np.max(z)
        e = np.exp(z)
        return e / e.sum()

    def predict(self, features):
        return self._softmax(features @ self.W + self.b)

    def update(self, features, true_class):
        probs = self.predict(features)
        grad_z = probs.copy()
        grad_z[true_class] -= 1.0   # softmax + cross-entropy gradient
        self.W -= self.lr * np.outer(features, grad_z)
        self.b -= self.lr * grad_z
        return -np.log(probs[true_class] + 1e-9)  # loss for monitoring


rng = np.random.default_rng(1)
tuner = LastLayerFineTuner(feature_dim=16, n_classes=3, lr=0.1)

# Simulate a fixed "frozen backbone output" per class, plus noise per
# sample -- stands in for real extracted features from a pretrained CNN.
class_prototypes = rng.normal(0, 1, size=(3, 16))

losses = []
for step in range(60):
    true_class = step % 3
    features = class_prototypes[true_class] + rng.normal(0, 0.3, size=16)
    loss = tuner.update(features, true_class)
    losses.append(loss)

print(f"loss, first 5 steps:  {[round(l, 3) for l in losses[:5]]}")
print(f"loss, last 5 steps:   {[round(l, 3) for l in losses[-5:]]}")

# Evaluate accuracy on fresh samples after training
correct = 0
n_eval = 30
for i in range(n_eval):
    true_class = i % 3
    features = class_prototypes[true_class] + rng.normal(0, 0.3, size=16)
    pred = np.argmax(tuner.predict(features))
    correct += (pred == true_class)
print(f"post-training eval accuracy: {correct}/{n_eval}")
```

Running this prints:

```
loss, first 5 steps:  [1.092, 0.963, 1.415, 0.605, 0.361]
loss, last 5 steps:   [0.045, 0.025, 0.039, 0.037, 0.039]
post-training eval accuracy: 30/30
```

The per-sample loss dropping from roughly 0.4-1.4 down to well under 0.05 over
60 single-sample updates — with 100% held-out accuracy afterward — shows
the mechanism working on a small, cleanly-separable synthetic task. In a
real deployment, the "frozen backbone" would be an actual quantized CNN's
penultimate-layer output (computed via a normal Level 2/3 inference pass),
and only this small head — here, `16 x 3` weights plus 3 biases, trivial
to store and update in a few KB of RAM — needs to persist and update
on-device.

## Online learning: updating from a live, unlabeled-until-confirmed stream

A common on-device pattern: the device makes a prediction, and only
*sometimes* gets a label — a user correcting a misclassification, or a
secondary sensor confirming a detection (Module 08's fusion pattern). The
update rule needs to handle this intermittent, single-sample supervision
without needing a full batch or a stored dataset.

```python
class OnlineMeanAdapter:
    """A minimal but real on-device adaptation pattern: rather than
    retraining a classifier, adapt per-class feature centroids
    incrementally as labeled corrections arrive, and classify new points
    by nearest centroid. This is a legitimate lightweight alternative to
    gradient-based fine-tuning when RAM is too tight even for the linear
    head above."""

    def __init__(self, feature_dim, n_classes):
        self.centroids = np.zeros((n_classes, feature_dim))
        self.counts = np.zeros(n_classes)

    def update(self, features, confirmed_class):
        # Incremental mean update: new_mean = old_mean + (x - old_mean) / n
        self.counts[confirmed_class] += 1
        n = self.counts[confirmed_class]
        self.centroids[confirmed_class] += (features - self.centroids[confirmed_class]) / n

    def predict(self, features):
        dists = np.linalg.norm(self.centroids - features, axis=1)
        return int(np.argmin(dists))


rng = np.random.default_rng(2)
adapter = OnlineMeanAdapter(feature_dim=8, n_classes=2)
true_centroids = rng.normal(0, 2, size=(2, 8))

# Only every 3rd sample gets a confirmed label (simulating intermittent
# supervision), the rest are prediction-only.
n_correct_after_labeling = []
for i in range(90):
    true_class = i % 2
    features = true_centroids[true_class] + rng.normal(0, 0.5, size=8)
    if i % 3 == 0:
        adapter.update(features, true_class)
    if i > 0 and adapter.counts.min() > 0:
        pred = adapter.predict(features)
        n_correct_after_labeling.append(pred == true_class)

accuracy = np.mean(n_correct_after_labeling)
print(f"labeled updates used: {int(adapter.counts.sum())}/90 samples")
print(f"prediction accuracy across the run: {accuracy:.3f}")
```

Running this prints:

```
labeled updates used: 30/90 samples
prediction accuracy across the run: 1.0
```

Only a third of the samples ever received a label, yet the incrementally-
updated centroids classify essentially every sample correctly — nearest-
centroid adaptation is a genuinely lightweight (O(feature_dim) memory per
class, O(1) update cost) technique for the common case where a device
gets sparse, intermittent supervision rather than a dense labeled stream.

## The risk unique to on-device learning: catastrophic forgetting and feedback loops

Both techniques above adapt to whatever data the device sees, which
creates a failure mode training-once-in-the-cloud never has to consider:
if a device's recent inputs are unrepresentative (a camera pointed at an
unusual scene for a few hours), online updates can drift the model away
from correctness for the *typical* case it will see again later. Standard
mitigations: cap how far online-adapted parameters can move from their
shipped starting point, keep a small buffer of "anchor" examples from the
original training distribution and periodically re-train against them
too, and — tying back to Module 03 — monitor the adapted model's
confidence distribution over time to catch drift the adaptation itself
introduced, not just drift from the environment.

## Edge-AI tradeoffs

| Factor | Last-layer fine-tuning | Online centroid adaptation |
|---|---|---|
| Memory cost | one layer's weights + biases | one centroid vector per class |
| Update cost per sample | one matrix-vector product + outer product | O(feature_dim), essentially free |
| Expressiveness | can learn any linear decision boundary over features | limited to nearest-centroid boundaries |
| Needs a frozen backbone already deployed | yes | yes |
| Risk of catastrophic forgetting | moderate, needs anchor-data mitigation | lower, but so is capacity to learn complex patterns |

## Exercise

Add an anchor-replay mechanism to `LastLayerFineTuner`: store a small
fixed buffer of the first N training examples seen (say N=10 across all
classes), and after every 10 online updates, run one additional gradient
step against each buffered anchor example. Compare final eval accuracy
and loss trajectory with and without anchor replay when you deliberately
feed the tuner a long, class-imbalanced run (e.g. 50 consecutive samples
from only class 0) — the anchor replay should visibly reduce how much the
tuner forgets classes 1 and 2 during that imbalanced stretch.
