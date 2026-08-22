# Person Detection (Visual Wake Words)

Person detection — "is there a person in this frame, yes or no" — is the
canonical *visual wake word*: the vision equivalent of Module 01's audio
keyword spotting, and one of the most deployed edge-AI tasks in existence
(security cameras, smart doorbells, occupancy sensors, people counters). It
is deliberately a **binary classification**, not object detection with
bounding boxes — no localization, just presence — which is exactly what
makes it cheap enough to run continuously on a microcontroller. This module
covers the task's specific dataset and model design choices, and a
streaming decision layer analogous to Module 01's debounce logic.

## Why binary presence, not full object detection

Full object detection (bounding boxes, multiple classes, non-max
suppression) needs orders of magnitude more compute than classification —
architectures like YOLO or SSD are built for phone/edge-GPU tiers, not
microcontrollers with no hardware floating point acceleration. The
influential reference model for this space, Google's **Visual Wake
Words** (built on a MobileNetV1-style backbone, ~250KB quantized), reframes
the problem as pure classification: given a 96x96 (or similar)
frame, output one probability — "does a person appear anywhere in this
image" — using the same depthwise-separable-conv-plus-GlobalAveragePooling
architecture from Module 02, just with 2 output classes (`person`,
`no_person`) instead of N.

```python
# Described, not run -- Visual Wake Words-style architecture, extends Module 02's pattern.
# model = keras.Sequential([
#     keras.layers.Input(shape=(96, 96, 1)),
#     keras.layers.Conv2D(8, 3, strides=2, activation="relu"),
#     keras.layers.DepthwiseConv2D(3, strides=1, activation="relu"),
#     keras.layers.Conv2D(16, 1, activation="relu"),
#     keras.layers.DepthwiseConv2D(3, strides=2, activation="relu"),
#     keras.layers.Conv2D(32, 1, activation="relu"),
#     keras.layers.DepthwiseConv2D(3, strides=2, activation="relu"),
#     keras.layers.Conv2D(48, 1, activation="relu"),
#     keras.layers.GlobalAveragePooling2D(),
#     keras.layers.Dense(2, activation="softmax"),   # [no_person, person]
# ])
```

## Dataset design specific to presence detection

Person detection's dataset failure modes are a direct instance of Module
04's principles, with task-specific specifics worth calling out:

- **Scale variance is the dominant nuisance factor.** A person filling the
  frame and a person as a distant speck 15 pixels tall are wildly different
  in pixel statistics; your positive class must cover both, or the model
  learns "large blob in center" rather than "person."
- **Partial views count as positive.** A hand, a shoulder, a person mostly
  occluded by a doorframe — all still "person present" for most real
  deployments (a security camera should still fire on a partially-visible
  intruder). Decide this labeling policy explicitly and apply it
  consistently, or the training signal becomes ambiguous exactly at the
  hard cases that matter most.
- **Hard negatives are anything person-shaped.** Mannequins, large dogs,
  coat racks, framed photos of people, shadows shaped like a person — as
  discussed generally in Module 04, these near-miss negatives teach the
  model to look for actual distinguishing detail (limb structure, texture)
  rather than a crude silhouette heuristic.
- **Lighting and time-of-day coverage.** A camera-based deployment sees
  dawn, midday, dusk, and (if it has any night-vision/IR capability)
  infrared frames — each with a different pixel-intensity distribution that
  must appear in both classes, using the `condition_coverage` check from
  Module 04.

```python
import numpy as np

def scale_bucket_report(bounding_box_heights, frame_height, buckets=(0.1, 0.3, 0.6, 1.0)):
    """bounding_box_heights: array of person bbox heights in pixels (from your
    labeling tool, even though the final model has no box output -- box height
    is still useful metadata for checking scale coverage). Returns counts per
    relative-size bucket."""
    relative = np.asarray(bounding_box_heights) / frame_height
    counts = {}
    lower = 0.0
    for b in buckets:
        counts[f"{lower:.1f}-{b:.1f}"] = int(np.sum((relative > lower) & (relative <= b)))
        lower = b
    return counts

heights = [10, 15, 40, 96, 55, 20, 80, 12]  # pixel heights, out of a 96px-tall frame
print(scale_bucket_report(heights, frame_height=96))
```

A report showing zero samples in the `0.1-0.3` bucket (distant/small
persons) directly predicts the model will fail on distant subjects in
deployment — exactly the kind of gap `condition_coverage`-style auditing
catches before you burn training time.

## Confidence thresholding and temporal smoothing

Person detection deployed on a live camera feed needs the same debounce
logic as Module 01's wake-word detector, adapted for vision's typical
failure pattern: brief single-frame misclassifications from motion blur,
lighting flicker, or a person passing at the frame edge.

```python
class PersonPresenceTracker:
    """Tracks a smoothed 'person present' state across frames using an
    exponential moving average of the person-class probability, plus
    hysteresis (different thresholds to enter vs. exit the 'present' state)
    so the output doesn't flicker near the boundary."""
    def __init__(self, ema_alpha=0.3, enter_threshold=0.75, exit_threshold=0.45):
        self.ema_alpha = ema_alpha
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.smoothed_prob = 0.0
        self.present = False

    def update(self, person_prob):
        self.smoothed_prob = (self.ema_alpha * person_prob
                               + (1 - self.ema_alpha) * self.smoothed_prob)
        if not self.present and self.smoothed_prob >= self.enter_threshold:
            self.present = True
        elif self.present and self.smoothed_prob <= self.exit_threshold:
            self.present = False
        return self.present

tracker = PersonPresenceTracker()
probs = [0.1, 0.2, 0.8, 0.9, 0.6, 0.3, 0.2, 0.85, 0.9]
for p in probs:
    print(f"prob={p:.2f}  smoothed={tracker.smoothed_prob:.2f}  present={tracker.update(p)}")
```

The hysteresis gap (`enter_threshold=0.75` vs. `exit_threshold=0.45`) is
deliberate: without it, a probability oscillating around a single threshold
(say 0.6) flips the reported state on every small fluctuation. Requiring a
much lower probability to *exit* "present" than to *enter* it means a
person who briefly turns away or is partly occluded doesn't immediately
register as "gone."

## Edge-AI tradeoffs

**Frame rate vs. power.** Running inference on every camera frame at 30fps
is rarely necessary for presence detection — most deployments sample at
1-5fps and rely on temporal smoothing to bridge the gaps, cutting compute
(and battery drain) by 6-30x with negligible impact on detecting a
person who stays in frame for more than a second.

**Model capacity vs. false positive rate.** A model too small to learn
person-vs-mannequin distinctions will false-positive constantly in
retail/security settings full of person-shaped objects — Visual Wake
Words-class models (~250K params) exist at roughly the minimum capacity
found to handle this reliably; going much smaller trades real accuracy
for marginal RAM savings.

**Hysteresis width vs. responsiveness.** A wide gap between enter/exit
thresholds suppresses flicker very effectively but delays reporting "person
left" by however many frames it takes the EMA to decay — tune the gap
against how quickly your application needs to react.

**Binary presence vs. counting/localization.** If the actual product need
is "how many people" or "where in the frame," presence detection is the
wrong task entirely — that requires detection or counting models an order
of magnitude larger; don't retrofit a presence detector's output into a
counting feature.

## Cheat sheet

| Concern | Guidance |
|---|---|
| Task framing | binary classification, not detection — presence only |
| Reference architecture | Visual Wake Words: depthwise-separable CNN + GAP, ~250K params |
| Dominant dataset risk | scale variance — cover near AND far persons |
| Hard negatives | person-shaped non-persons (mannequins, photos, dogs) |
| Sampling rate | 1-5fps typical; rely on temporal smoothing, not raw fps |
| Smoothing | EMA + hysteresis (different enter/exit thresholds) |
| Failure to watch for | model keying on frame brightness/blob size, not features |
| When presence detection is the wrong tool | you need counting or localization |

## Exercise

1. Implement `scale_bucket_report` on a synthetic array of 30 bounding-box
   heights you construct with a deliberate gap in one bucket, and confirm
   the report surfaces the gap.
2. Implement `PersonPresenceTracker` and feed it a probability sequence that
   oscillates around 0.6 (e.g. `[0.55,0.65,0.58,0.62,0.57,0.63]*3`). Compare
   the number of state flips with hysteresis (`enter=0.75, exit=0.45`)
   against a naive single-threshold version at `0.6`. Quantify the flicker
   reduction.
3. In prose, design a hard-negative collection list (at least 6 items) for
   a person-detector meant for a front-porch camera, drawing on Module 04's
   negative-class-design principles applied to this specific task.
4. Given a target of 5fps sampling and a model that takes 45ms per
   inference, compute the maximum duty cycle (fraction of time actively
   computing vs. idle) and discuss what that means for battery-powered
   deployment versus a fixed sub-1-second detection latency requirement.
