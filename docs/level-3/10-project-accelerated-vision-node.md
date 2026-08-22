# Project: Accelerated Vision Node

This project combines every module from Level 3 into one coherent design:
a camera-based edge node that detects objects using an NPU-accelerated
detector, streams frames through a bounded pipeline, verifies its own
model's integrity at boot, and reports latency using MLPerf-style
statistics rather than a single ad hoc timing. Nothing here trains a new
model — the goal is the system design and the glue code around a model,
which is where most real edge-AI engineering time actually goes.

## System overview

```
Camera sensor (30 fps)
   |
   v
[Bounded frame queue]  <-- Module 04: drop-oldest under backpressure
   |
   v
[Preprocess: resize + int8 quantize]
   |
   v
[ONNX Runtime session, NPU execution provider]  <-- Modules 01-03
   |  (falls back to CPU provider per-op if NPU can't claim it)
   v
[Raw detection candidates: boxes + scores, thousands of anchors]
   |
   v
[Confidence threshold + NMS]  <-- Module 05 (CPU-bound, by design)
   |
   v
[Final detections] --> application logic (alert / log / actuate)

At boot, before any of the above runs:
[Model file] -> [hash verification against signed manifest]  <-- Module 07
```

## Component 1: boot-time integrity check

Before the pipeline starts, the node verifies its own model file against a
signed manifest — a rollback-resistant version of Module 07's hash check,
so a compromised OTA update or tampered flash image is caught before a
single frame is processed.

```python
import hashlib
import json

def verify_and_load_manifest(model_bytes, manifest_json, installed_version):
    """manifest_json: {"sha256": "...", "version": int}, normally verified
    against a signature before trusting its contents at all -- signature
    verification itself is out of scope here (Module 07's exercise),
    this function assumes the manifest already passed that check."""
    manifest = json.loads(manifest_json)
    actual_hash = hashlib.sha256(model_bytes).hexdigest()

    if actual_hash != manifest["sha256"]:
        return {"ok": False, "reason": "hash mismatch -- possible tampering"}
    if manifest["version"] < installed_version:
        return {"ok": False, "reason": "rollback attempt -- older version rejected"}
    return {"ok": True, "reason": "verified", "version": manifest["version"]}


model_bytes = b"pretend .onnx model bytes"
good_manifest = json.dumps({
    "sha256": hashlib.sha256(model_bytes).hexdigest(), "version": 3,
})
result = verify_and_load_manifest(model_bytes, good_manifest, installed_version=2)
print(result)

rollback_manifest = json.dumps({
    "sha256": hashlib.sha256(model_bytes).hexdigest(), "version": 1,
})
result2 = verify_and_load_manifest(model_bytes, rollback_manifest, installed_version=2)
print(result2)
```

This runs as shown and prints `{'ok': True, ...}` for the valid, newer
manifest and `{'ok': False, 'reason': 'rollback attempt...'}` for the
older one — the node refuses to run a validly-signed but outdated model,
closing the rollback gap Module 07's exercise raised.

## Component 2: the bounded pipeline with per-stage timing

Reuses Module 04's `BoundedFrameQueue` and Module 09's warm-up-aware
benchmarking, applied per pipeline stage instead of to one function, so a
slowdown can be attributed to the actual stage causing it rather than
just "the pipeline got slower."

```python
from collections import deque
import time
import numpy as np

class BoundedFrameQueue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = deque()
        self.dropped = 0

    def produce(self, frame):
        if len(self.queue) >= self.capacity:
            self.queue.popleft()
            self.dropped += 1
        self.queue.append(frame)

    def consume(self):
        return self.queue.popleft() if self.queue else None


def run_pipeline_tick(queue, preprocess_fn, infer_fn, postprocess_fn, stage_times):
    frame = queue.consume()
    if frame is None:
        return None

    t0 = time.perf_counter()
    x = preprocess_fn(frame)
    t1 = time.perf_counter()
    raw_output = infer_fn(x)
    t2 = time.perf_counter()
    detections = postprocess_fn(raw_output)
    t3 = time.perf_counter()

    stage_times["preprocess"].append((t1 - t0) * 1000)
    stage_times["infer"].append((t2 - t1) * 1000)
    stage_times["postprocess"].append((t3 - t2) * 1000)
    return detections


# Simulated stand-ins for real preprocess/infer/postprocess -- the queue
# and timing harness structure is what this component demonstrates.
def fake_preprocess(frame):
    time.sleep(0.001)
    return frame

def fake_infer(x):
    time.sleep(0.015)   # NPU-accelerated backbone, fast
    return np.random.default_rng(x).random((100, 6))  # 100 candidate boxes

def fake_postprocess(raw_output):
    time.sleep(0.004)   # NMS, CPU-bound (Module 05)
    return raw_output[raw_output[:, 4] > 0.5]  # confidence threshold


queue = BoundedFrameQueue(capacity=3)
stage_times = {"preprocess": [], "infer": [], "postprocess": []}

for frame_id in range(20):
    queue.produce(frame_id)
    run_pipeline_tick(queue, fake_preprocess, fake_infer, fake_postprocess, stage_times)

for stage, times in stage_times.items():
    arr = np.array(times)
    print(f"{stage}: p50={np.percentile(arr,50):.2f}ms  p99={np.percentile(arr,99):.2f}ms")
print(f"frames dropped by queue: {queue.dropped}")
```

Running this (on this machine; `time.sleep` granularity and OS scheduling
noise mean exact numbers will vary run to run) printed:

```
preprocess: p50=1.26ms  p99=3.83ms
infer: p50=18.87ms  p99=46.22ms
postprocess: p50=5.04ms  p99=5.38ms
frames dropped by queue: 0
```

Per-stage breakdown immediately shows `infer` dominates total latency at
roughly 19ms per tick (with a noisy tail up to 46ms here, itself a small
reminder of Module 09's p50-vs-p99 lesson) — in a real deployment this is
where you'd know to
spend NPU/compiler-optimization effort (Modules 01, 06), while
`postprocess` (NMS, Module 05) is a distant second and not worth the same
investment. This attribution is only possible because timing is per-stage,
not a single end-to-end number.

## Component 3: fusing detection confidence with a secondary sensor

If the node also carries a PIR (passive infrared motion) sensor, Module
08's late-fusion pattern lets a borderline-confidence detection get
resolved by corroborating motion evidence rather than either accepted or
discarded on vision alone:

```python
def fuse_detection_with_motion(detection_confidence, motion_detected,
                                camera_weight=1.0, motion_weight=0.4):
    motion_score = 0.9 if motion_detected else 0.3
    weights_sum = camera_weight + motion_weight
    fused = (detection_confidence * camera_weight +
             motion_score * motion_weight) / weights_sum
    return fused

borderline_detection = 0.55  # camera alone: ambiguous
with_motion = fuse_detection_with_motion(borderline_detection, motion_detected=True)
without_motion = fuse_detection_with_motion(borderline_detection, motion_detected=False)
print(f"fused score with corroborating motion: {with_motion:.3f}")
print(f"fused score with no motion detected: {without_motion:.3f}")
```

This prints `fused score with corroborating motion: 0.650` and
`fused score with no motion detected: 0.479` — the same ambiguous 0.55
camera reading crosses a typical 0.6 decision threshold when corroborated
by motion and falls below it when not, which is the entire value
proposition of Module 08's fusion approach applied to a concrete
detection decision.

## What's deliberately out of scope

This design does not include model training, the actual NPU compiler
invocation (Modules 01-02's real CLI tools, which need physical hardware),
or a real camera driver — those are hardware- and vendor-specific enough
that they belong in a project repository tied to a specific board, not a
portable code sample. What transfers regardless of target hardware is the
architecture: verify-then-load, bound-then-drop, measure-per-stage,
fuse-when-ambiguous.

## Stretch goals

- Replace the fixed confidence/motion weights in Component 3 with the
  adaptive-weighting approach from Module 08 (degrade the camera's weight
  under a simulated low-light flag) and confirm the fusion output shifts
  appropriately.
- Add a fourth pipeline stage — a rolling MLPerf-style benchmark (Module
  09's `summarize_latency`) that runs every N frames and logs a warning if
  p99 `infer` latency drifts more than 20% above its first-hour baseline,
  as an early signal of thermal throttling or NPU driver degradation.
- Extend the boot-time manifest check to verify a second, independent hash
  covering the *inference code* itself (not just the model weights), so a
  tampered postprocessing/NMS implementation is caught by the same
  mechanism that catches a tampered model file.
