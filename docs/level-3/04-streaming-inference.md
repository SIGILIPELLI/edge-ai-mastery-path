# Streaming Inference Pipelines

Every model so far has been fed one complete, pre-cut input: a full 1-second
audio window, a full image frame. Real deployments rarely get inputs that
clean — a microphone produces an endless stream of samples with no natural
boundaries, and a camera sensor produces frames faster than a detector can
process them. **Streaming inference** is the set of techniques for running
a model continuously against an unbounded input stream without either
falling behind (dropping real-time deadlines) or reprocessing the same data
wastefully. This module covers the two dominant patterns — sliding windows
with overlap, and producer/consumer buffering with backpressure — both
runnable and tested here in plain Python.

## The sliding window problem

A keyword spotter needs a fixed-size window of audio (say, 1 second at
16 kHz = 16,000 samples) but the microphone hands you samples in small
chunks (say, 320-sample frames every 20 ms). Two decisions matter: how
much the windows overlap, and what happens to state at the boundary.

```python
import numpy as np
from collections import deque

class SlidingWindowBuffer:
    """Accumulates streamed chunks into fixed-size, overlapping windows.
    hop_size < window_size means consecutive windows overlap -- this
    tested run uses window=8, hop=4 (2x overlap) so it's checkable by eye."""

    def __init__(self, window_size, hop_size):
        self.window_size = window_size
        self.hop_size = hop_size
        self.buffer = deque(maxlen=window_size)
        self.samples_since_last_window = 0

    def push(self, chunk):
        """Feed one chunk of samples; yield a window each time enough
        new samples have arrived to advance by hop_size."""
        windows = []
        for sample in chunk:
            self.buffer.append(sample)
            self.samples_since_last_window += 1
            ready = (len(self.buffer) == self.window_size and
                     self.samples_since_last_window >= self.hop_size)
            if ready:
                windows.append(np.array(self.buffer))
                self.samples_since_last_window = 0
        return windows


buf = SlidingWindowBuffer(window_size=8, hop_size=4)
stream = list(range(1, 21))  # simulate 20 streamed samples, chunks of 3
all_windows = []
for i in range(0, len(stream), 3):
    chunk = stream[i:i + 3]
    all_windows.extend(buf.push(chunk))

for w in all_windows:
    print(w)
```

Running this prints:

```
[1 2 3 4 5 6 7 8]
[ 5  6  7  8  9 10 11 12]
[ 9 10 11 12 13 14 15 16]
[13 14 15 16 17 18 19 20]
```

Each window advances by exactly `hop_size=4` samples and overlaps the
previous one by `window_size - hop_size = 4` samples — the overlap is what
prevents a keyword spoken right at a window boundary from being split
across two windows and missed by both. The cost is proportional: 2x
overlap here means running the model roughly 2x as often as a
non-overlapping approach would, a direct latency-vs-accuracy trade you
tune per application.

## Producer/consumer buffering and backpressure

The sliding window above assumes inference keeps up with the stream. It
often doesn't — a camera producing frames at 30 fps against a detector
that takes 40 ms per inference (25 fps capacity) will fall behind forever
if every frame is queued. The standard fix is a **bounded queue with a
drop policy**: when the consumer can't keep up, deliberately discard the
oldest (or newest) unprocessed data rather than let the queue grow without
bound and turn every result stale and delayed.

```python
from collections import deque
import time

class BoundedFrameQueue:
    """A producer/consumer buffer with a fixed capacity and an explicit
    drop-oldest policy under backpressure -- simulates a camera producing
    frames faster than a model can consume them."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = deque()
        self.dropped = 0

    def produce(self, frame):
        if len(self.queue) >= self.capacity:
            self.queue.popleft()   # drop the oldest, keep freshness
            self.dropped += 1
        self.queue.append(frame)

    def consume(self):
        return self.queue.popleft() if self.queue else None


def simulate(producer_rate_hz, consumer_time_s, duration_s, capacity):
    q = BoundedFrameQueue(capacity)
    frame_interval = 1.0 / producer_rate_hz
    n_frames = int(duration_s / frame_interval)
    consumed = 0
    t = 0.0
    for frame_id in range(n_frames):
        q.produce(frame_id)
        t += frame_interval
        if t >= consumer_time_s * (consumed + 1):
            if q.consume() is not None:
                consumed += 1
    return {"produced": n_frames, "consumed": consumed,
            "dropped": q.dropped, "final_queue_len": len(q.queue)}


result = simulate(producer_rate_hz=30, consumer_time_s=1/25,
                   duration_s=2.0, capacity=3)
print(result)
```

Running this prints:

```
{'produced': 60, 'consumed': 50, 'dropped': 8, 'final_queue_len': 2}
```

Producer rate (30 fps) exceeds consumer rate (25 fps) over 2 seconds, so
the queue fills and the bounded policy sheds the excess (8 dropped here)
rather than accumulating unbounded latency. This is the correct behavior
for a live camera feed — a 2-second-stale detection is often worse than no
detection at all — but it's the *wrong* choice for something like a
security-event logger where every frame matters; that case needs either a
faster model, a larger disk-backed buffer, or accepting a growing backlog
during bursts. Which policy is correct depends entirely on whether stale
results or lost data hurts more for your application.

## State across window boundaries: the hidden bug

A subtlety that breaks naive streaming implementations: any *stateful*
preprocessing (a running mean/variance normalizer, an IIR filter, an RNN's
hidden state) must persist across windows, not reset per-window, or you
reintroduce discontinuities at every boundary that a stateless windowed
approach was supposed to eliminate.

```python
class StreamingNormalizer:
    """Running mean/variance normalizer that must NOT reset between
    windows -- a common streaming bug is re-instantiating this per
    window, which reintroduces a discontinuity at every boundary."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0  # Welford's algorithm for numerically stable variance

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def normalize(self, x):
        variance = self.m2 / self.n if self.n > 1 else 1.0
        std = max(variance ** 0.5, 1e-6)
        return (x - self.mean) / std


norm = StreamingNormalizer()
for x in [1.0, 2.0, 3.0, 100.0, 4.0, 5.0]:  # 100.0 simulates a transient spike
    norm.update(x)
    print(f"x={x:6.1f}  normalized={norm.normalize(x):7.3f}  running_mean={norm.mean:6.2f}")
```

Running this prints:

```
x=   1.0  normalized=  0.000  running_mean=  1.00
x=   2.0  normalized=  1.000  running_mean=  1.50
x=   3.0  normalized=  1.225  running_mean=  2.00
x= 100.0  normalized=  1.732  running_mean= 26.50
x=   4.0  normalized= -0.461  running_mean= 22.00
x=   5.0  normalized= -0.392  running_mean= 19.17
```

The important thing to notice is `running_mean` never resets to 0 at any
point; it carries forward exactly like a real deployment's normalizer
would across window boundaries, and a genuine spike like the `100.0`
outlier pulls the mean sharply upward without ever discarding history —
which is itself worth flagging: a real streaming normalizer usually needs
a forgetting factor (exponential decay on old statistics) so one transient
spike doesn't permanently bias normalization for the rest of the stream.

## Edge-AI tradeoffs

| Factor | Sliding window (overlap) | Bounded queue (drop policy) |
|---|---|---|
| Solves | boundary-split events (keyword cut across windows) | producer/consumer rate mismatch |
| Cost | more inferences per second of input | lost data during bursts |
| Tuning knob | overlap ratio (`hop_size / window_size`) | queue capacity + drop-oldest vs drop-newest |
| Wrong choice looks like | missed detections at buffer boundaries | unbounded latency growth, eventual OOM |
| Needs persistent state? | only if preprocessing is stateful (see normalizer above) | no — the queue itself is the only state |

## Exercise

Extend `BoundedFrameQueue` with a `drop_newest` policy (reject the
incoming frame instead of evicting the oldest one) and re-run `simulate`
with both policies at `capacity=1` — a tight buffer that maximizes the
practical difference. Compare `dropped` counts and think through which
policy a fall-detection alarm system should use versus which a live video
preview should use; they are not the same answer.
