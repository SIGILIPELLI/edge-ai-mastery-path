# Data Collection & Dataset Design

Edge AI models fail in the field far more often from bad data than from bad
architectures. A model trained on studio-quality recordings and clean lab
photos meets a noisy kitchen and a badly-lit hallway in production, and the
accuracy gap between "test set" and "real world" is almost always a dataset
design problem. This module covers the discipline of building a dataset
that actually represents deployment conditions: sampling strategy, labeling
consistency, splitting correctly, and measuring class balance and drift
before you ever start training.

## The core failure mode: test accuracy that lies

The single most common edge-AI dataset mistake is a **leaky split** —
training and test data drawn from the same recording session, same speaker,
same lighting, same background. A model can score 98% on such a test set
and 60% in deployment, because the test set never asked it to generalize
across the things that actually vary in the field (different rooms,
different people, different days, different hardware units).

```python
import numpy as np

def session_based_split(samples, session_ids, test_sessions):
    """Split by SESSION, not by random shuffle, so no session's audio/images
    leak between train and test. samples: list-like. session_ids: parallel
    array of session identifiers. test_sessions: set of session ids held out."""
    session_ids = np.asarray(session_ids)
    test_mask = np.isin(session_ids, list(test_sessions))
    train_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]
    return train_idx, test_idx

# Example: 5 recording sessions, hold out session 4 entirely for test.
sessions = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
train_idx, test_idx = session_based_split(range(len(sessions)), sessions, {4})
print("train:", train_idx, "test:", test_idx)
```

A random shuffle of the same 10 samples could easily put both halves of
session 4 on different sides of the split — the model "memorizes" that
session's background noise and looks great on a test point drawn from the
same recording. Splitting by session (or by speaker, by device unit, by day)
closes that leak.

## Measuring class balance and coverage

Before training, quantify two things: how balanced the classes are, and how
much *condition diversity* (lighting, background noise, distance from
sensor, orientation) each class actually covers.

```python
def class_balance_report(labels):
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    report = []
    for c, n in zip(classes, counts):
        report.append((c, n, n / total))
    imbalance = counts.max() / counts.min()
    return report, imbalance

def condition_coverage(metadata, condition_key):
    """metadata: list of dicts, one per sample, each with e.g.
    {'label': 'person', 'lighting': 'dim', 'distance_m': 2.5}.
    Returns per-label counts of each observed condition value."""
    from collections import defaultdict
    coverage = defaultdict(lambda: defaultdict(int))
    for m in metadata:
        coverage[m["label"]][m[condition_key]] += 1
    return {k: dict(v) for k, v in coverage.items()}

meta = [
    {"label": "person", "lighting": "bright"},
    {"label": "person", "lighting": "bright"},
    {"label": "person", "lighting": "dim"},
    {"label": "no_person", "lighting": "bright"},
    {"label": "no_person", "lighting": "bright"},
    {"label": "no_person", "lighting": "bright"},
]
print(condition_coverage(meta, "lighting"))
# {'person': {'bright': 2, 'dim': 1}, 'no_person': {'bright': 3}}
```

That toy example already reveals a real bug: the `no_person` class has zero
`dim` lighting examples. A model trained on it may learn "dim = person" by
accident, simply because dim conditions never appeared in the negative
class during training.

## Negative/background class design

For detection-style tasks (wake word, person detection, anomaly detection),
the negative class is usually *harder to design well* than the positive
class, because "everything that isn't the target" is enormous and easy to
under-sample. Two negative-class mistakes recur constantly:

- **Silence-only negatives** (for audio): a model trained against pure
  silence as its only negative learns to detect "any sound" rather than the
  specific keyword, and false-triggers on TV, traffic, or conversation.
- **Empty-background-only negatives** (for vision): a model trained only
  against an empty room learns "any change in the frame," not "the specific
  object," and false-triggers on shadows, pets, or camera noise.

The fix is the same in both domains: deliberately collect **hard
negatives** — near-miss audio (similar-sounding words, background speech)
or near-miss images (other objects, partial occlusions, similar-looking
non-targets) — and verify their presence with `condition_coverage`-style
checks before training.

## Data augmentation: multiplying coverage without multiplying collection

Augmentation is cheap coverage expansion, not a replacement for it — it can
simulate variation you already have *some* real examples of, but it cannot
invent variation you've never observed at all.

```python
def augment_audio_snr(clean_signal, noise_signal, target_snr_db, rng=None):
    """Mix clean_signal with noise_signal at a target signal-to-noise ratio,
    simulating a noisier deployment environment from clean recordings."""
    rng = rng or np.random.default_rng()
    sig_power = np.mean(clean_signal ** 2)
    noise_power = np.mean(noise_signal ** 2)
    target_noise_power = sig_power / (10 ** (target_snr_db / 10))
    scale = np.sqrt(target_noise_power / (noise_power + 1e-12))
    return clean_signal + scale * noise_signal

def augment_image_brightness(image, factor_range=(0.6, 1.4), rng=None):
    """image: float array in [0, 1]. Randomly scales brightness and clips."""
    rng = rng or np.random.default_rng()
    factor = rng.uniform(*factor_range)
    return np.clip(image * factor, 0.0, 1.0)
```

A dataset with 10 real "dim lighting" photos, brightness-augmented into 200
variants, still only encodes 10 distinct scenes — augmentation multiplies
apparent volume but not the number of independent real-world samples behind
it. Treat augmented counts and real-collection counts as separate numbers in
your coverage report.

## Labeling consistency

Multi-person or multi-session labeling drifts. A `condition_coverage`-style
audit is cheap for conditions you tagged deliberately, but labeling
*mistakes* (mislabeled class, inconsistent boundary decisions — "is this
still a wake word if it's whispered?") need spot-checking:

```python
def flag_label_disagreement(labels_a, labels_b, sample_ids):
    """Compare two independent labelings of the same samples (e.g. two
    annotators, or a re-label pass) and report disagreement rate."""
    labels_a, labels_b = np.asarray(labels_a), np.asarray(labels_b)
    disagree = labels_a != labels_b
    rate = disagree.mean()
    disagreeing_ids = np.asarray(sample_ids)[disagree]
    return rate, disagreeing_ids
```

A disagreement rate above a few percent between two labelers on the same
sample set usually means the labeling *instructions* are ambiguous, not that
one labeler is careless — fix the definition before collecting more data
against it.

## Edge-AI tradeoffs

**Collection volume vs. condition diversity.** 10,000 samples from one room
teach a model far less than 1,000 samples spread across ten rooms, ten
lighting conditions, and multiple hardware units — diversity of conditions
usually beats raw count for field robustness.

**Real hard negatives vs. synthetic augmentation.** Augmentation is nearly
free but can't manufacture a *genuinely novel* confusable case; a small
budget of real hard-negative collection (10-20 minutes of a housemate
talking near the wake-word mic) is often worth more than hours of synthetic
noise-mixing.

**Session-based splits vs. more training data.** Holding out entire
sessions/speakers/devices for testing means less data available to train
on, but produces a test accuracy number that actually predicts field
performance — a smaller number you can trust beats a larger number that
lies.

**Labeling speed vs. labeling consistency.** A single fast labeler is
internally consistent but may encode one person's blind spots system-wide;
cross-checking with `flag_label_disagreement` costs time but catches
definitional drift before it poisons the whole dataset.

## Cheat sheet

| Practice | Why |
|---|---|
| Split by session/speaker/device | prevents leakage that inflates test accuracy |
| `class_balance_report` | catches skewed class counts before training |
| `condition_coverage` per class | catches missing conditions (e.g. no dim negatives) |
| Collect real hard negatives | prevents the model learning the wrong signal |
| Track augmented vs. real counts separately | augmentation ≠ new information |
| `flag_label_disagreement` | catches ambiguous labeling instructions early |
| Re-check coverage after every collection round | datasets grow unevenly by default |

## Exercise

1. Implement `session_based_split` and, using 4 synthetic sessions of
   different sizes, show that a random shuffle split places samples from the
   same session on both sides while the session-based split never does.
2. Build a `metadata` list of at least 20 synthetic samples across 2 classes
   and 3 conditions (e.g. lighting or noise level), deliberately leaving one
   class/condition combination empty, then confirm `condition_coverage`
   surfaces the gap.
3. Implement `augment_audio_snr` on a synthetic sine-wave "clean" signal
   plus synthetic white-noise, and verify the achieved SNR (recompute it
   from the mixed signal) is close to your `target_snr_db` request.
4. Design (in prose) a hard-negative collection plan for a real deployment
   scenario of your choice (a wake word for a specific room, or a
   person-detector for a specific camera position) — list at least 5
   concrete hard-negative conditions you'd deliberately record.
