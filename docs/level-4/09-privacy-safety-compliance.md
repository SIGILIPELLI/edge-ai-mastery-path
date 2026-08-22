# Privacy, Safety & Compliance

Modules 04-05 mentioned, without fully addressing, that on-device
learning and federated learning don't provide a formal privacy guarantee
on their own. This module closes that gap and covers the regulatory and
safety-engineering side of production edge AI: **differential privacy**
as the actual mathematical technique that makes a privacy claim provable
rather than just "harder to reverse," and the compliance/safety-case
thinking (GDPR/CCPA-relevant data handling, functional-safety-style
failure analysis) that a shipped product built on Levels 1-3's models
needs before it reaches real users.

## Differential privacy: turning "hard to reverse" into a provable bound

Module 05 flagged that a federated model update can, under some
conditions, leak information about the data that produced it. Differential
privacy (DP) is the standard technique for closing that gap: add
carefully calibrated random noise to a value derived from data (a
gradient, an aggregate statistic) such that the *distribution* of possible
noisy outputs barely changes whether or not any single individual's data
was included — making it mathematically bounded how much one person's
data could have influenced what left the device.

```python
import numpy as np

def laplace_mechanism(true_value, sensitivity, epsilon, rng):
    """The canonical DP mechanism for a single scalar statistic:
    Laplace noise scaled by sensitivity/epsilon. Smaller epsilon means
    more noise and a stronger (more private) guarantee; sensitivity is
    the maximum amount one individual's data could change the true
    value (here, assumed known ahead of time for the statistic in
    question)."""
    scale = sensitivity / epsilon
    noise = rng.laplace(0, scale)
    return true_value + noise


def private_mean_release(values, value_range, epsilon, rng):
    """Releases a DP-protected mean of a local dataset -- e.g. what a
    federated client might report instead of raw data or an unprotected
    exact statistic. sensitivity of a mean over n bounded values is
    (max-min)/n, since one individual's value can shift the mean by at
    most that much."""
    n = len(values)
    true_mean = np.mean(values)
    sensitivity = (value_range[1] - value_range[0]) / n
    return laplace_mechanism(true_mean, sensitivity, epsilon, rng)


rng = np.random.default_rng(0)
local_values = rng.uniform(0, 100, size=50)   # e.g. 50 local sensor readings
true_mean = np.mean(local_values)

for epsilon in [0.1, 1.0, 10.0]:
    releases = [private_mean_release(local_values, (0, 100), epsilon, rng) for _ in range(1000)]
    error = np.std(releases)
    print(f"epsilon={epsilon:5.1f}  true_mean={true_mean:.2f}  "
          f"avg_released_mean={np.mean(releases):.2f}  std_of_noise={error:.2f}")
```

Running this prints:

```
epsilon=  0.1  true_mean=52.68  avg_released_mean=53.33  std_of_noise=28.02
epsilon=  1.0  true_mean=52.68  avg_released_mean=52.53  std_of_noise=2.93
epsilon= 10.0  true_mean=52.68  avg_released_mean=52.68  std_of_noise=0.28
```

Every epsilon value's released values *average* back to close to the true
mean over many releases (differential privacy protects individuals, not
the aggregate statistic itself), but the noise std at epsilon=0.1 (28.02)
is large enough that any single release is nearly useless as an estimate
of the true value — this is the concrete privacy/utility tradeoff DP makes
explicit and tunable, unlike federated learning's raw-data-never-leaves
property, which offers no dial to turn at all.

## Composability: why repeated queries erode the privacy budget

A single DP-protected release is bounded by its epsilon, but a device or
server answering many queries about overlapping data accumulates privacy
loss across all of them — a detail easy to overlook if each individual
release looks fine in isolation.

```python
def composed_epsilon(epsilon_per_query, n_queries):
    """Basic (non-advanced) composition: privacy loss adds linearly
    across independent queries against the same underlying data. Real
    systems use tighter 'advanced composition' bounds that grow more
    slowly, but the linear bound is the simple, always-valid baseline."""
    return epsilon_per_query * n_queries

for n in [1, 10, 100, 1000]:
    total = composed_epsilon(0.1, n)
    print(f"{n:4d} queries at epsilon=0.1 each -> total privacy loss (basic composition): {total:.1f}")
```

Running this prints:

```
   1 queries at epsilon=0.1 each -> total privacy loss (basic composition): 0.1
  10 queries at epsilon=0.1 each -> total privacy loss (basic composition): 1.0
 100 queries at epsilon=0.1 each -> total privacy loss (basic composition): 10.0
1000 queries at epsilon=0.1 each -> total privacy loss (basic composition): 100.0
```

Ten repeated queries against the same underlying data, each individually
labeled "epsilon=0.1" and each looking privacy-preserving in isolation,
compose to a total privacy loss of 1.0 — an order of magnitude worse than
any single query suggests. A total epsilon of 100 after 1,000 queries is
effectively no privacy protection at all, which is why real DP
systems track a fixed total privacy budget per individual and refuse
further queries once it's exhausted, rather than letting it grow
unbounded.

## Data minimization: the compliance-first design habit

Before any DP mechanism is needed, the simplest and most effective privacy
control is not collecting or transmitting more than necessary in the first
place — directly relevant to every telemetry design decision in Module 03.

```python
def redact_telemetry_payload(raw_payload: dict, allowed_fields: set) -> dict:
    """A minimal but real compliance control: strip any field not on an
    explicit allowlist before a payload leaves the device, rather than
    trusting every producer of telemetry to remember not to include
    sensitive fields. Fail-closed by construction -- an unrecognized new
    field added later gets dropped by default, not leaked by default."""
    return {k: v for k, v in raw_payload.items() if k in allowed_fields}

raw = {
    "device_id": "dev-0042", "confidence": 0.91, "detection_class": "person",
    "gps_lat": 37.7749, "gps_lon": -122.4194,   # precise location -- sensitive
    "raw_audio_snippet": b"...",                 # raw sensor data -- sensitive
}
allowed = {"device_id", "confidence", "detection_class"}
safe_payload = redact_telemetry_payload(raw, allowed)
print("raw payload keys:", sorted(raw.keys()))
print("redacted payload:", safe_payload)
```

Running this prints:

```
raw payload keys: ['confidence', 'detection_class', 'device_id', 'gps_lat', 'gps_lon', 'raw_audio_snippet']
redacted payload: {'device_id': 'dev-0042', 'confidence': 0.91, 'detection_class': 'person'}
```

An allowlist (fail-closed: only named fields survive) is a stronger
default than a denylist (fail-open: only named fields get stripped),
because a denylist silently leaks any new sensitive field a future code
change adds without updating the list — exactly the kind of gap a GDPR/
CCPA data-minimization review is meant to catch before it ships.

## Safety-case thinking: what to document before deployment

Regulatory frameworks relevant to physical-world AI systems (safety
standards in industrial and automotive contexts, emerging AI-specific
regulation) converge on the same basic structure: a documented, falsifiable
argument for why the system is safe enough for its intended use, not just
a claim that it works. For an edge AI product, the minimum viable safety
case draws directly on earlier Level 4 modules:

| Safety-case element | Where it's actually built |
|---|---|
| What happens when the model is wrong | Module 08's sensor fusion, confidence thresholds, human-in-the-loop fallback for high-stakes decisions |
| What happens when an update goes bad | Module 02's dual-slot rollback and staged rollout gates |
| How degradation is detected before it causes harm | Module 03's drift monitoring |
| What data is collected and why | this module's data minimization, documented against actual use |
| What happens if the model is tampered with | Level 3 Module 07's integrity verification |

The pattern across all five rows: a safety case is not a separate
deliverable bolted onto a finished product, it's a documentation pass over
engineering decisions that (if this course's earlier modules were
followed) already exist for other reasons.

## Edge-AI tradeoffs

| Factor | No formal privacy protection | Differential privacy applied |
|---|---|---|
| Guarantee strength | none — "probably fine" | mathematically bounded, tunable via epsilon |
| Utility of released data | full fidelity | degraded, proportional to privacy strength |
| Implementation cost | none | requires sensitivity analysis per statistic released |
| Query budget management | not applicable | must track cumulative epsilon per individual, enforce a cutoff |

## Exercise

Implement a `PrivacyBudgetTracker` class that maintains a running total
epsilon per device, raises an error on any query that would exceed a
configured maximum budget, and write a test that confirms it correctly
blocks further queries in a loop of otherwise-identical 0.1-epsilon
queries once the total would exceed a budget of 5.0 (this should happen
on roughly the 51st query). Then verify by hand that it does not block
the 50th.
