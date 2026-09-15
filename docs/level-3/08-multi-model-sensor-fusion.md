---
description: "Multi-Model Sensor Fusion — Every module so far has run one model against one sensor stream. Many real edge devices carry several sensors at once — a…"
---

# Multi-Model Sensor Fusion

Every module so far has run one model against one sensor stream. Many
real edge devices carry several sensors at once — a camera plus a
microphone plus an IMU — and the interesting engineering problem isn't
running each model, it's **combining their outputs** into a single, more
reliable decision than any one sensor could produce alone. This module
covers the three levels at which fusion happens (early/feature, late/
decision, and hybrid), with tested Python implementations of the
statistical combination rules, since the full multi-sensor models
themselves need training data this environment doesn't have.

## Why fuse at all: complementary failure modes

The value of fusion isn't redundancy for its own sake — it's that
different sensors fail in different, often uncorrelated ways. A camera-
based person detector fails in the dark; a microphone-based detector fails
in a loud room; a millimeter-wave radar fails on largely stationary
objects. A system relying on any single sensor inherits that sensor's
blind spot completely; a fused system only fails when *multiple* sensors'
blind spots overlap simultaneously, which is a much rarer event if the
sensors were chosen for genuinely different failure conditions.

## Late fusion: combining independent model outputs

The simplest and most common pattern on constrained hardware: run each
sensor's model independently (each stays a normal Level 1-2 pipeline) and
combine only the final scores. This is attractive at the edge because each
model can be developed, quantized, and updated independently — no shared
architecture, no joint retraining required.

```python
import numpy as np

def late_fusion_weighted_average(scores, weights):
    """scores: dict of {sensor_name: probability_of_positive_class}.
    weights: dict of {sensor_name: reliability weight}, need not sum to 1."""
    total_weight = sum(weights.values())
    fused = sum(scores[k] * weights[k] for k in scores) / total_weight
    return fused

# A person-detection scenario: camera is confident but it's dim, mic
# picks up footsteps clearly, IMU (worn/attached) shows no motion at all
# (device is stationary -- weak evidence either way, so low weight).
scores = {"camera": 0.55, "microphone": 0.85, "imu": 0.50}
weights = {"camera": 1.0, "microphone": 1.5, "imu": 0.3}

fused_score = late_fusion_weighted_average(scores, weights)
print(f"individual scores: {scores}")
print(f"fused score: {fused_score:.3f}")
print(f"decision: {'person present' if fused_score > 0.6 else 'uncertain/absent'}")
```

Running this prints:

```
individual scores: {'camera': 0.55, 'microphone': 0.85, 'imu': 0.5}
fused score: 0.705
decision: person present
```

Neither the camera (0.55) nor the IMU (0.50) alone crosses the 0.6
decision threshold, but the microphone's strong, heavily-weighted signal
pulls the fused score past it — this is the practical payoff of fusion:
a borderline-uncertain camera reading gets resolved by corroborating
evidence rather than left as a coin flip.

## Setting the weights: reliability isn't static

Fixed weights (as above) are a reasonable starting point, but a more
robust design adjusts weights based on **known operating conditions** —
the camera's weight should drop in low light, the microphone's weight
should drop in high ambient noise, exactly because that's when each
sensor's individual failure mode is active.

```python
def adaptive_weights(base_weights, light_level, noise_level):
    """light_level, noise_level: 0.0 (poor conditions) to 1.0 (ideal).
    Degrades a sensor's weight toward zero as its operating conditions
    worsen, rather than trusting it equally regardless of context."""
    return {
        "camera": base_weights["camera"] * light_level,
        "microphone": base_weights["microphone"] * noise_level,
        "imu": base_weights["imu"],  # unaffected by light/sound
    }

base = {"camera": 1.0, "microphone": 1.5, "imu": 0.3}

good_conditions = adaptive_weights(base, light_level=0.9, noise_level=0.9)
dark_loud_conditions = adaptive_weights(base, light_level=0.1, noise_level=0.2)

print("good conditions weights:", good_conditions)
print("dark+loud conditions weights:", dark_loud_conditions)

fused_good = late_fusion_weighted_average(scores, good_conditions)
fused_bad = late_fusion_weighted_average(scores, dark_loud_conditions)
print(f"fused score, good conditions: {fused_good:.3f}")
print(f"fused score, dark+loud conditions: {fused_bad:.3f}")
```

Running this prints:

```
good conditions weights: {'camera': 0.9, 'microphone': 1.35, 'imu': 0.3}
dark+loud conditions weights: {'camera': 0.1, 'microphone': 0.3, 'imu': 0.3}
fused score, good conditions: 0.703
fused score, dark+loud conditions: 0.657
```

In dark, loud conditions, both the camera and microphone's weights shrink
sharply, so the IMU's uninformative 0.50 reading now makes up a much
larger share of the average — the fused score drops from 0.703 to 0.657,
noticeably closer to the "uncertain" IMU baseline even though it hasn't
crossed the 0.6 threshold in this particular example. Push conditions
further (near-zero light and noise weights) and it would cross; the point
is that known-bad sensor conditions correctly *reduce a sensor's influence*
on the final decision rather than only shrinking its raw score.

## Early (feature-level) fusion: the alternative, and why it's rarer at the edge

Early fusion concatenates raw features from multiple sensors *before* a
single shared model, rather than running separate models and combining
outputs. It can capture cross-sensor correlations late fusion structurally
cannot (e.g., "audio energy spikes at the exact same moment as vibration") —
but it requires one joint model trained on synchronized multi-sensor data,
which means every sensor's data must be resampled to a common clock, and
the joint model can't be updated for one sensor without retraining the
whole thing. On resource-constrained hardware, late fusion's independence
between models (each individually small enough to fit, individually
updatable, individually already-quantized from Level 2) usually wins
unless the cross-sensor correlation itself is the signal you need.

## Edge-AI tradeoffs

| Factor | Late fusion (combine outputs) | Early fusion (combine features) |
|---|---|---|
| Model count | one small model per sensor | one joint model, larger |
| Synchronization requirement | loose (per-model timing only) | strict (features must align in time) |
| Captures cross-sensor correlation | no, only independently | yes |
| Update one sensor's model independently | yes | no, requires full retrain |
| Memory/compute footprint | sum of independent small models | one larger joint model |
| Typical edge use | most multi-sensor edge products | specialized cases needing cross-modal correlation |

## How It Actually Works

**Why a weighted average is a maximum-likelihood combination under a
specific noise assumption, not an arbitrary heuristic.** If each sensor's
score is treated as a noisy, independent estimate of the true probability
with variance inversely proportional to its assigned weight, the
statistically optimal (minimum-variance) combination of several
independent noisy estimates is exactly the inverse-variance-weighted
average — a classical result from estimation theory. `weights` in
`late_fusion_weighted_average` is standing in for each sensor's inverse
noise variance: a sensor known to be reliable in current conditions gets a
higher weight because its errors are smaller, and the weighted average
mathematically discounts noisier sensors' contributions proportional to
how unreliable they are — which is precisely why the microphone's strong,
heavily-weighted 0.85 reading can pull an otherwise-borderline fused score
past the decision threshold in the worked example, rather than being
diluted evenly by three equally-trusted opinions.

**Why sensors with genuinely independent failure modes reduce the
system's overall failure probability multiplicatively, not additively.**
If a camera's failure probability in some condition is `p_cam` and a
microphone's is `p_mic`, and their failure causes are truly independent
(dark rooms don't make microphones fail; loud rooms don't make cameras
fail), the probability that *both* fail simultaneously on the same
event is `p_cam × p_mic` — multiplying two probabilities less than 1
always yields something smaller than either alone. This is the precise
mathematical content behind "a fused system only fails when multiple
sensors' blind spots overlap": it's not a vague synergy claim, it's
independence turning an OR of failure conditions (either sensor fails)
into an AND of failure conditions for the *fused* system to fail
outright, which only holds to the extent the chosen sensors' failure
modes are actually uncorrelated — correlated failures (e.g. camera and
IR-illuminated microphone both failing in the same power brownout)
don't get this multiplicative benefit.

**Why adaptive weighting must scale weight, not directly rewrite the raw
score, to correctly represent "no information."** `adaptive_weights`
multiplies a sensor's base weight by its condition quality (`light_level`,
`noise_level`) rather than, say, forcing a low-light camera's score toward
0.5 directly. This distinction matters mechanically in the weighted
average: shrinking a sensor's *weight* toward zero makes its score
contribute almost nothing to the sum regardless of what that score
happens to be, correctly modeling "this sensor's reading, whatever it
is, should barely count right now" — whereas overwriting the raw score
would conflate "the sensor is unreliable" with "the sensor is reporting
uncertainty," two different pieces of information the fusion formula
needs to keep separate to remain a valid weighted average at all.

## Exercise

Extend `adaptive_weights` to also degrade the microphone's weight under a
third condition — `wind_level` (0.0-1.0, where high wind noise saturates a
microphone the same way loud ambient noise does) — and confirm that
stacking two degraded conditions (loud *and* windy) pushes the fused score
further toward "uncertain" than either condition alone. This mirrors a
real design decision: whether degradation factors should combine
multiplicatively (compounding) or by taking the worst single factor.
