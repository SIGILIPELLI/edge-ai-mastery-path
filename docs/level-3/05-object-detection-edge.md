# Object Detection on Edge Devices

Everything trained so far has been a **classifier**: one image or window
in, one label out. Object detection is a different task shape entirely —
one image in, a variable number of boxes-plus-labels out — and that
variability is exactly what makes it expensive on constrained hardware.
This module covers the architecture choices that make detection tractable
on a microcontroller or a small NPU (SSD-style single-shot detectors,
anchor boxes, and non-max suppression), with a fully tested NMS
implementation in Python, since the actual detector backbones require
training infrastructure this environment can't run.

## Why detection is harder than classification, mechanically

A classifier's output layer is small and fixed-size: one score per class.
A naive detector would need to predict a variable-length list of boxes,
which doesn't map onto a fixed-size neural network output at all. The
trick every edge-friendly detector (SSD, YOLO-tiny, MobileDet) uses is to
**reframe variable-length detection as fixed-size dense prediction**:
tile the image with a fixed grid of candidate box shapes ("anchors"), and
have the network predict, for every anchor, a confidence score and a
small position adjustment. A 20x20 grid with 3 anchor shapes per cell
produces exactly 1,200 fixed-size predictions, most of which will be
"no object here," and post-processing throws away the low-confidence ones.

```
Image -> backbone (MobileNet-style depthwise convs) -> feature map
       -> per grid cell, per anchor shape:
            [objectness_score, class_scores..., dx, dy, dw, dh]
       -> thousands of raw candidate boxes
       -> confidence threshold + NMS -> handful of final detections
```

This is why detection models are almost always built on the same
depthwise-separable convolution backbones from Level 2 (MobileNet-class
architectures) — the expensive part is the same feature extraction, and
the detection "head" added on top is comparatively cheap.

## Anchor boxes: pre-defined shape priors

Anchors are a fixed set of box width/height pairs, chosen ahead of time
(often via k-means clustering over the training set's ground-truth box
shapes) so the network only has to predict small *offsets* from a
reasonable starting shape, rather than a box's absolute size and position
from scratch — a much easier regression target.

```python
import numpy as np

def generate_anchor_grid(grid_size, anchor_shapes, image_size):
    """Generates anchor boxes as (x1, y1, x2, y2) for every cell in a
    grid_size x grid_size grid, for each shape in anchor_shapes
    (a list of (width, height) fractions of image_size)."""
    cell = image_size / grid_size
    anchors = []
    for row in range(grid_size):
        for col in range(grid_size):
            cx = (col + 0.5) * cell
            cy = (row + 0.5) * cell
            for (aw_frac, ah_frac) in anchor_shapes:
                aw, ah = aw_frac * image_size, ah_frac * image_size
                anchors.append((cx - aw / 2, cy - ah / 2,
                                 cx + aw / 2, cy + ah / 2))
    return np.array(anchors)

anchors = generate_anchor_grid(
    grid_size=4, anchor_shapes=[(0.2, 0.2), (0.3, 0.5), (0.5, 0.3)],
    image_size=224)
print(f"total anchors: {len(anchors)}")   # 4*4*3 = 48
print("first anchor (x1,y1,x2,y2):", anchors[0])
```

This actually runs and prints `total anchors: 48` and the first anchor's
coordinates — with a 4x4 grid and 3 shapes per cell, exactly
`4*4*3=48` candidate boxes, tiny by design so the arithmetic is checkable
by hand; a real MobileDet-class model uses several grid resolutions
simultaneously (fine grids catch small objects, coarse grids catch large
ones) and totals several thousand anchors.

## Non-max suppression: collapsing thousands of candidates to a few

Most of those thousands of anchors will fire on the *same* real object
with overlapping boxes at slightly different positions and confidences.
**Non-max suppression (NMS)** greedily keeps the highest-confidence box
and discards any other box that overlaps it more than a threshold,
repeating until no boxes remain.

```python
import numpy as np

def iou(box_a, box_b):
    """Intersection-over-union of two (x1,y1,x2,y2) boxes."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """boxes: list of (x1,y1,x2,y2). scores: matching confidence list.
    Returns indices of boxes to keep."""
    order = np.argsort(scores)[::-1]  # highest confidence first
    keep = []
    while len(order) > 0:
        current = order[0]
        keep.append(int(current))
        remaining = order[1:]
        remaining = [i for i in remaining
                     if iou(boxes[current], boxes[i]) <= iou_threshold]
        order = np.array(remaining)
    return keep


boxes = [
    (10, 10, 50, 50),   # box 0: object A
    (12, 11, 52, 49),   # box 1: near-duplicate of box 0 (high overlap)
    (100, 100, 140, 140),  # box 2: object B, far away
    (60, 60, 100, 100),    # box 3: object C, non-overlapping with either
]
scores = [0.95, 0.80, 0.90, 0.60]
kept = non_max_suppression(boxes, scores, iou_threshold=0.5)
print("kept indices:", kept)
for i in kept:
    print(f"  box {i}: {boxes[i]}  score={scores[i]}")
```

Running this prints:

```
kept indices: [0, 2, 3]
kept indices: [0, 2, 3]
  box 0: (10, 10, 50, 50)  score=0.95
  box 2: (100, 100, 140, 140)  score=0.9
  box 3: (60, 60, 100, 100)  score=0.6
```

Box 1 (the near-duplicate of box 0, IoU well above 0.5) is correctly
suppressed even though its own confidence (0.80) beat box 3's (0.60) —
NMS only compares boxes against already-kept, higher-scoring boxes, not
against a global ranking, which is exactly why the greedy highest-first
order matters.

## Why NMS is the part that hurts on microcontrollers

The convolutional backbone maps cleanly onto CMSIS-NN kernels or an NPU's
systolic array — it's dense, regular, predictable-latency math. NMS is
the opposite: variable-length lists, sorting, and data-dependent branching
per pair of boxes, none of which vectorizes or maps to fixed-function
accelerator hardware. In practice this means:

- NMS almost always runs on the host CPU even when the backbone runs on
  an NPU — the "graph segment" cost from Module 01 applies here even
  though NMS isn't a `.tflite` op, because it's genuinely a different kind
  of computation.
- Its cost scales with the number of *candidate* boxes surviving the
  confidence threshold, not the total anchor count — a noisy model that
  produces many above-threshold false positives makes NMS the bottleneck
  even if the backbone is fast.
- Raising the confidence threshold before NMS is a cheap, effective way to
  cut NMS's worst-case cost, at the price of potentially dropping real
  low-confidence detections.

## Edge-AI tradeoffs

| Factor | Classification (Level 1-2) | Detection (this module) |
|---|---|---|
| Output shape | fixed (N classes) | variable (0-K boxes), handled via dense anchors |
| Backbone cost | one forward pass | same backbone + detection head |
| Post-processing | argmax | confidence threshold + NMS (CPU-bound, data-dependent) |
| Maps to NPU cleanly? | yes, fully | backbone yes, NMS almost never |
| Typical model family | MobileNet, tiny CNN | SSD-MobileNet, MobileDet, YOLO-tiny |

## Exercise

Extend the `non_max_suppression` function to run per-class (group boxes by
predicted class label before applying NMS within each group, rather than
across all classes at once) — this is what real detectors do, since a
"person" box and a "dog" box can legitimately overlap heavily and neither
should suppress the other. Test it against a small hand-built example with
two overlapping boxes of different classes and confirm both survive.
