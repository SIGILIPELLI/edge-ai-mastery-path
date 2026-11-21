---
title: "Final Quiz"
description: "Test what you have learned across all four levels of Edge AI Mastery Path with a 19-question final quiz."
---

# Final Quiz

19 questions drawn from all four levels of this course. You need **70% (14 out of 19)** to pass —
you can retake it as many times as you like. Your best result is saved in this browser.

<div id="mp-quiz-root"></div>

<script type="application/json" id="mp-quiz-data">
[
  {
    "question": "A sensor must run a keyword model 24/7 on a coin cell. What's the right approach?",
    "options": [
      "A tiny always-on first-stage model with duty cycling, waking bigger stages only on a hit",
      "A cloud call per second",
      "A large model at full clock",
      "Streaming raw audio"
    ],
    "correctIndex": 0,
    "explanation": "Power dominates edge design."
  },
  {
    "question": "Why quantize a model to int8 for an MCU?",
    "options": [
      "Needs a GPU",
      "More accurate",
      "Larger but faster",
      "About 4x smaller and faster on integer hardware, with a small accuracy cost"
    ],
    "correctIndex": 3,
    "explanation": "Often required to fit."
  },
  {
    "question": "A float range [-2.0, 6.0] is quantized to int8 asymmetrically. What are scale and zero point?\n```python\nlo,hi=-2.0,6.0\nscale=(hi-lo)/255\nzp=round(-128-lo/scale)\nprint(round(scale,5), zp)\n```\n",
    "options": [
      "0.03137 -64",
      "0.0625 -64",
      "0.03137 0",
      "0.03137 64"
    ],
    "correctIndex": 0,
    "explanation": "zero_point = round(-128 - min/scale). Verified by running it."
  },
  {
    "question": "Post-training quantization loses too much accuracy. What next?",
    "options": [
      "Give up",
      "Try quantization-aware training so weights adapt to int8 rounding",
      "Use float32",
      "Remove layers"
    ],
    "correctIndex": 1,
    "explanation": "QAT simulates quantization in training."
  },
  {
    "question": "A 1-second audio clip at 16 kHz is split into 30 ms frames with a 20 ms stride. How many frames?\n```python\nfs=16000; frame=480; stride=320; n=16000\nprint((n-frame)//stride+1)\n```\n",
    "options": [
      "50",
      "30",
      "49",
      "33"
    ],
    "correctIndex": 2,
    "explanation": "floor((16000-480)/320)+1. Verified by running it."
  },
  {
    "question": "Training used librosa MFCCs but firmware computes them differently and accuracy collapses. Why?",
    "options": [
      "Quantization only",
      "Too much RAM",
      "Bad luck",
      "Preprocessing mismatch shifts the input distribution; use the same implementation or verify parity"
    ],
    "correctIndex": 3,
    "explanation": "Test features end to end."
  },
  {
    "question": "The tensor arena is too small and inference fails at AllocateTensors. What do you do?",
    "options": [
      "Increase the arena (or reduce activation sizes) and measure arena_used_bytes",
      "Ignore",
      "Disable ops",
      "Reduce flash"
    ],
    "correctIndex": 0,
    "explanation": "Size by measurement."
  },
  {
    "question": "Your model needs ops not in the OpResolver. What happens and what's the fix?",
    "options": [
      "Slow",
      "Works",
      "Silent",
      "AllocateTensors fails; register the needed ops (or change the architecture)"
    ],
    "correctIndex": 3,
    "explanation": "Only link the ops you use."
  },
  {
    "question": "Test accuracy is 95% but the field accuracy is 70%. What's the most likely cause?",
    "options": [
      "Bad compiler",
      "Dataset mismatch: noise, devices, lighting or subjects absent from training",
      "Too many epochs",
      "Wrong loss"
    ],
    "correctIndex": 1,
    "explanation": "Collect representative data."
  },
  {
    "question": "Why split train and test by subject or session?",
    "options": [
      "Law",
      "Speed",
      "Style",
      "Same-subject leakage inflates accuracy"
    ],
    "correctIndex": 3,
    "explanation": "Generalization to new users."
  },
  {
    "question": "Unstructured pruning removed 80% of weights but latency barely changed. Why?",
    "options": [
      "Pruning is useless",
      "Bug",
      "Quantization",
      "Kernels don't exploit scattered zeros; structured pruning or smaller dense layers speed up inference"
    ],
    "correctIndex": 3,
    "explanation": "Hardware must exploit sparsity."
  },
  {
    "question": "A Coral Edge TPU model runs slower than expected. What do you check first?",
    "options": [
      "Fonts",
      "Wi-Fi",
      "The compile report for ops falling back to the CPU",
      "Battery"
    ],
    "correctIndex": 2,
    "explanation": "Unsupported ops split the graph."
  },
  {
    "question": "Why benchmark on your own device instead of trusting vendor TOPS?",
    "options": [
      "Cost",
      "Law",
      "Style",
      "Real throughput depends on supported ops, memory bandwidth and your model"
    ],
    "correctIndex": 3,
    "explanation": "Measure."
  },
  {
    "question": "A microphone stream is endless. How do you feed it to a fixed-size model?",
    "options": [
      "Resize audio",
      "One big input",
      "Slide overlapping windows over a ring buffer and smooth the predictions",
      "Wait"
    ],
    "correctIndex": 2,
    "explanation": "Streaming inference."
  },
  {
    "question": "An object detector produces many overlapping boxes for one object. What post-processing do you apply?",
    "options": [
      "Quantization",
      "Non-maximum suppression",
      "Dropout",
      "Pruning"
    ],
    "correctIndex": 1,
    "explanation": "Keep the highest-confidence box."
  },
  {
    "question": "A model update bricks 2% of a pilot fleet. Which practices would have limited the damage?",
    "options": [
      "Signed, staged (canary) rollouts with automatic rollback to the previous model",
      "All at once",
      "Bigger update",
      "Manual fixes"
    ],
    "correctIndex": 0,
    "explanation": "Safe OTA."
  },
  {
    "question": "How can you detect drift on devices that can't upload raw data?",
    "options": [
      "Upload everything",
      "Reboot",
      "Report lightweight statistics such as confidence histograms and feature means",
      "Nothing"
    ],
    "correctIndex": 2,
    "explanation": "Privacy-preserving telemetry."
  },
  {
    "question": "Federated learning sends model updates instead of data. Does it guarantee privacy?",
    "options": [
      "No: updates can leak information, so add secure aggregation or differential privacy",
      "Yes always",
      "Only on MCUs",
      "Never useful"
    ],
    "correctIndex": 0,
    "explanation": "Defense in depth."
  },
  {
    "question": "Which is the best measure of an always-on model's efficiency?",
    "options": [
      "File size",
      "Energy per inference and average current in real usage, alongside accuracy",
      "Parameter count",
      "Training loss"
    ],
    "correctIndex": 1,
    "explanation": "Measure power."
  }
]
</script>

## Coding challenges

Try these on your own computer, then compare your output. Nothing you write is sent anywhere.

### Int8 quantization by hand

Implement quant_params (asymmetric int8: scale = range/255, zero_point = round(-128 - min/scale)), quantize (round, shift by zero point, clamp to int8) and dequantize. Verify the rounding error never exceeds half a quantization step and compare model sizes.

**Starter code**

```python
def quant_params(lo, hi):
    # asymmetric int8: scale = (hi-lo)/255, zero_point = round(-128 - lo/scale)
    return 1.0, 0

def quantize(x, scale, zp):
    # round(x/scale) + zp, clamped to [-128, 127]
    return 0

def dequantize(q, scale, zp):
    # (q - zp) * scale
    return 0.0

weights = [-0.92, -0.31, 0.0, 0.18, 0.55, 0.99]
scale, zp = quant_params(min(weights), max(weights))
print("scale=%.5f zero_point=%d" % (scale, zp))
qs = [quantize(w, scale, zp) for w in weights]
print(qs)
errs = [abs(w - dequantize(q, scale, zp)) for w, q in zip(weights, qs)]
print("max error=%.4f  (<= scale/2: %s)" % (max(errs), max(errs) <= scale / 2 + 1e-9))
print("size: float32=%d bytes, int8=%d bytes" % (4 * len(weights), len(weights)))
```

**Expected output**

```text
scale=0.00749 zero_point=-5
[-128, -46, -5, 19, 68, 127]
max error=0.0032  (<= scale/2: True)
size: float32=24 bytes, int8=6 bytes
```

<details><summary>Show solution and explanation</summary>

<pre><code>def quant_params(lo, hi):
    scale = (hi - lo) / 255
    zero_point = round(-128 - lo / scale)
    return scale, zero_point

def quantize(x, scale, zp):
    q = round(x / scale) + zp
    return max(-128, min(127, q))

def dequantize(q, scale, zp):
    return (q - zp) * scale

weights = [-0.92, -0.31, 0.0, 0.18, 0.55, 0.99]
scale, zp = quant_params(min(weights), max(weights))
print(&quot;scale=%.5f zero_point=%d&quot; % (scale, zp))
qs = [quantize(w, scale, zp) for w in weights]
print(qs)
errs = [abs(w - dequantize(q, scale, zp)) for w, q in zip(weights, qs)]
print(&quot;max error=%.4f  (&lt;= scale/2: %s)&quot; % (max(errs), max(errs) &lt;= scale / 2 + 1e-9))
print(&quot;size: float32=%d bytes, int8=%d bytes&quot; % (4 * len(weights), len(weights)))
</code></pre>

<p>Each real value is rounded to the nearest quantization step, so the error is at most scale/2 while the model shrinks 4x. TFLite does this per tensor or per channel using calibration data. Verified by running the solution.</p>
</details>

### Windows and features for a gesture classifier

Implement windows() (overlapping full windows), features() (RMS to two decimals and zero-mean crossings) and classify() (rms < 0.2 is still; at least 6 crossings is shake; otherwise slow-move). Run it on three synthetic windows.

**Starter code**

```python
import math

def windows(samples, size, step):
    # overlapping windows (only full-size windows)
    return []

def features(w):
    # return (rms rounded to 2 decimals, number of zero-mean crossings between neighbouring samples)
    return (0, 0)

def classify(w):
    # rms < 0.2 -> "still"; else crossings >= 6 -> "shake"; else "slow-move"
    return ""

still = [0.02, -0.03, 0.01, 0.0, 0.02, -0.01, 0.0, 0.03]
shake = [1.0, -1.1, 0.9, -1.0, 1.2, -0.9, 1.0, -1.1]
slow = [0.0, 0.4, 0.8, 1.0, 0.8, 0.4, 0.0, -0.4]
for name, s in [("still", still), ("shake", shake), ("slow", slow)]:
    print(name, features(s), classify(s))
print("windows:", len(windows(list(range(20)), 8, 4)))
```

**Expected output**

```text
still (0.02, 6) still
shake (1.03, 7) shake
slow (0.59, 2) slow-move
windows: 4
```

<details><summary>Show solution and explanation</summary>

<pre><code>import math

def windows(samples, size, step):
    return [samples[i:i + size] for i in range(0, len(samples) - size + 1, step)]

def features(w):
    n = len(w)
    mean = sum(w) / n
    rms = math.sqrt(sum(x * x for x in w) / n)
    crossings = sum(1 for a, b in zip(w, w[1:]) if (a - mean) * (b - mean) &lt; 0)
    return round(rms, 2), crossings

def classify(w):
    rms, zc = features(w)
    if rms &lt; 0.2:
        return &quot;still&quot;
    return &quot;shake&quot; if zc &gt;= 6 else &quot;slow-move&quot;

still = [0.02, -0.03, 0.01, 0.0, 0.02, -0.01, 0.0, 0.03]
shake = [1.0, -1.1, 0.9, -1.0, 1.2, -0.9, 1.0, -1.1]
slow = [0.0, 0.4, 0.8, 1.0, 0.8, 0.4, 0.0, -0.4]
for name, s in [(&quot;still&quot;, still), (&quot;shake&quot;, shake), (&quot;slow&quot;, slow)]:
    print(name, features(s), classify(s))
print(&quot;windows:&quot;, len(windows(list(range(20)), 8, 4)))
</code></pre>

<p>Windowing plus cheap time-domain features is the standard first stage of an edge sensor pipeline. A tiny rule or model on those features can run in microseconds. Verified by running the solution.</p>
</details>
