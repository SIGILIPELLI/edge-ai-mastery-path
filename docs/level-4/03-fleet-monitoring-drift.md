# Fleet Monitoring & Drift Detection

Module 01 named the failure mode that makes fleet monitoring necessary:
a model can get quietly worse with no crash, no exception, no log line
that says "I'm wrong now." **Drift** is the general term for this —
either the real-world data distribution shifting away from what the model
was trained on (*data drift*), or the model's actual accuracy degrading
even on similar-looking inputs (*concept drift*, e.g. because the thing
being predicted has genuinely changed). Since a fleet device usually has
no ground-truth labels available on-device (nobody's manually confirming
every detection), monitoring has to work from proxy signals alone. This
module builds and tests two of the standard proxy-signal detectors.

## Why you can't just check accuracy in production

Accuracy requires a ground-truth label, and by the time a fleet device
in someone's home or on a factory floor produces a prediction, there's
usually no labeled answer to compare it against — that's the entire
premise of *deploying* a trained model rather than continuing to run it
against a labeled test set. Fleet monitoring instead watches for
**proxy signals** that correlate with something being wrong, without
needing labels at all:

- The distribution of the model's own **input features** shifting away
  from the training distribution (data drift).
- The distribution of the model's **output confidences** shifting —
  a well-calibrated model that starts producing unusually many
  low-confidence or borderline predictions is a signal something changed.
- Simple **input statistics** (mean pixel brightness, audio RMS level)
  moving outside historically normal ranges — cheap enough to compute
  on-device continuously, unlike a full drift-detection model.

## Detecting data drift with a population statistics test

The Kolmogorov-Smirnov (KS) test is a standard, well-understood way to
ask "are these two samples plausibly drawn from the same distribution?"
without assuming any particular distribution shape — a good fit here
because you rarely know the true shape of a real sensor's feature
distribution.

```python
import numpy as np
from scipy import stats

def detect_drift_ks(baseline_sample, current_sample, alpha=0.05):
    """Two-sample KS test: compares the empirical distributions of a
    baseline (training-time) feature sample against a current (fleet
    telemetry) sample. A small p-value means the two are unlikely to come
    from the same distribution -- i.e., drift is likely."""
    statistic, p_value = stats.ks_2samp(baseline_sample, current_sample)
    drifted = p_value < alpha
    return {"ks_statistic": float(statistic), "p_value": float(p_value),
            "drift_detected": bool(drifted)}

rng = np.random.default_rng(7)
baseline = rng.normal(loc=0.0, scale=1.0, size=1000)          # training-time feature distribution

no_drift_sample = rng.normal(loc=0.0, scale=1.0, size=300)     # same distribution, new sample
drifted_sample = rng.normal(loc=0.8, scale=1.3, size=300)      # shifted mean + wider spread

print("no-drift case:", detect_drift_ks(baseline, no_drift_sample))
print("drifted case:", detect_drift_ks(baseline, drifted_sample))
```

Running this prints:

```
no-drift case: {'ks_statistic': 0.048, 'p_value': 0.6449197416275865, 'drift_detected': False}
drifted case: {'ks_statistic': 0.33166666666666667, 'p_value': 5.178516383339652e-23, 'drift_detected': True}
```

The KS statistic itself (max distance between the two empirical CDFs) is
useful as a magnitude signal even below the drift threshold — trending it
over time on a per-device or per-cohort basis catches *gradual* drift
building up before it crosses the hard significance threshold and gets
flagged as a discrete event.

## Detecting drift from confidence distributions alone (no feature access needed)

Feature-level drift detection requires shipping raw feature vectors off
the device — often not viable under bandwidth or privacy constraints
(Module 09). A cheaper, privacy-friendlier proxy is watching the
distribution of the model's own **output confidence scores**, which are
tiny (one float per inference) and already being computed anyway.

```python
import numpy as np

def confidence_drift_score(baseline_confidences, current_confidences,
                            low_confidence_threshold=0.6):
    """Compares the fraction of low-confidence predictions between a
    baseline period and a current period. A rising low-confidence
    fraction is a cheap, label-free proxy for the model encountering
    inputs unlike its training distribution."""
    baseline_low_frac = np.mean(np.array(baseline_confidences) < low_confidence_threshold)
    current_low_frac = np.mean(np.array(current_confidences) < low_confidence_threshold)
    relative_increase = (current_low_frac - baseline_low_frac) / max(baseline_low_frac, 1e-6)
    return {
        "baseline_low_confidence_rate": float(baseline_low_frac),
        "current_low_confidence_rate": float(current_low_frac),
        "relative_increase": float(relative_increase),
        "concerning": bool(relative_increase > 0.5 and current_low_frac > 0.1),
    }

rng = np.random.default_rng(3)
baseline_confidences = np.clip(rng.normal(0.85, 0.10, 500), 0, 1)   # normally confident
healthy_period = np.clip(rng.normal(0.83, 0.11, 200), 0, 1)          # still healthy
degraded_period = np.clip(rng.normal(0.62, 0.18, 200), 0, 1)         # confidence collapsing

print("healthy period:", confidence_drift_score(baseline_confidences, healthy_period))
print("degraded period:", confidence_drift_score(baseline_confidences, degraded_period))
```

Running this prints:

```
healthy period: {'baseline_low_confidence_rate': 0.008, 'current_low_confidence_rate': 0.025, 'relative_increase': 2.125, 'concerning': False}
degraded period: {'baseline_low_confidence_rate': 0.008, 'current_low_confidence_rate': 0.47, 'relative_increase': 57.75, 'concerning': True}
```

The healthy period shows a large *relative* increase (2.1x) purely
because the baseline low-confidence rate is tiny (0.8%) — a small
absolute change looks huge as a ratio. That's exactly why `concerning`
requires both a relative jump *and* an absolute floor
(`current_low_frac > 0.1`): the healthy period's 2.5% current rate never
clears that floor and correctly stays unflagged, while the degraded
period's 47% rate clears it easily. Relying on relative change alone
would have false-alarmed on the healthy period.

## Aggregating per-device signals into a fleet-level view

A single device's telemetry is noisy; the useful signal is usually in
aggregate, across a cohort, compared to the same cohort's own recent
history (not a single global baseline, since a factory-floor cohort and a
home-appliance cohort likely have legitimately different normal
distributions).

```python
def aggregate_fleet_drift(device_drift_scores: dict, alert_fraction=0.15):
    """device_drift_scores: {device_id: bool} indicating whether each
    device individually flagged drift this period. Fleet-level alerting
    fires on a *fraction* of devices flagging, not any single device --
    a single flaky device shouldn't page anyone, a fleet-wide pattern should."""
    total = len(device_drift_scores)
    flagged = sum(1 for v in device_drift_scores.values() if v)
    fraction = flagged / total if total else 0.0
    return {"devices_flagged": flagged, "total_devices": total,
            "fraction_flagged": fraction, "fleet_alert": fraction >= alert_fraction}

single_device_blip = {f"dev-{i}": (i == 7) for i in range(50)}
widespread_issue = {f"dev-{i}": (i % 4 == 0) for i in range(50)}
print("single flaky device:", aggregate_fleet_drift(single_device_blip))
print("widespread pattern:", aggregate_fleet_drift(widespread_issue))
```

Running this prints:

```
single flaky device: {'devices_flagged': 1, 'total_devices': 50, 'fraction_flagged': 0.02, 'fleet_alert': False}
widespread pattern: {'devices_flagged': 13, 'total_devices': 50, 'fraction_flagged': 0.26, 'fleet_alert': True}
```

One flaky device out of 50 (2%) stays well under the 15% alert threshold
and correctly doesn't page anyone; 13 devices flagging together (26%,
because every 4th device in this synthetic pattern shares the same
underlying issue) clears the threshold and correctly triggers
`fleet_alert: True` — the fraction-based gate does exactly what it's
meant to: ignore isolated noise, catch a pattern shared across the fleet.

## Edge-AI tradeoffs

| Signal | Data cost (bandwidth/privacy) | Sensitivity | Needs labels? |
|---|---|---|---|
| Raw feature KS test | high (ships feature vectors) | high, direct | no |
| Confidence distribution | very low (one float per inference) | moderate, indirect | no |
| Fleet-level aggregation | low (booleans/counts only) | catches fleet-wide patterns, misses single-device issues by design | no |
| True accuracy (ideal, rarely available) | requires labeled ground truth | highest | yes |

## Exercise

Run `confidence_drift_score` against a third scenario: a baseline with a
*higher* low-confidence rate to begin with (say 15%, simulating a model
that was never that confident to start) and a current period at 18%.
Check whether `concerning` fires, and reason about whether it should —
this stresses the interaction between the relative-increase term and the
absolute floor differently than either scenario above, and is a good way
to find thresholds that need tuning for your own model's typical
confidence distribution before trusting this gate in production.
