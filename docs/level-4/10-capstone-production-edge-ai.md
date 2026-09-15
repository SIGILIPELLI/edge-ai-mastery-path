---
description: "Capstone: Production Edge AI System — This capstone assembles every module of Level 4 into one coherent product design: a fleet of battery-powered…"
---

# Capstone: Production Edge AI System

This capstone assembles every module of Level 4 into one coherent product
design: a fleet of battery-powered environmental monitoring nodes running
an anomaly-detection model, managed and improved over years of deployment.
It's the fleet-scale, long-lived counterpart to Level 3's single-device
Accelerated Vision Node project — where that project was about one
device's runtime pipeline, this one is about the systems and processes
that keep a *fleet* of such devices correct, updated, and compliant for
the years they're actually deployed.

## System overview

```
                         +---------------------------+
                         |      Cloud backend         |
                         |                             |
                         |  Model registry + rollout   |<-- Module 02
                         |  gates (staged canary,      |
                         |  automatic rollback)        |
                         |                             |
                         |  Fleet telemetry aggregator |<-- Module 03
                         |  + drift dashboards         |
                         |                             |
                         |  Federated aggregation      |<-- Module 05
                         |  server (FedAvg rounds)      |
                         |                             |
                         |  DP-protected telemetry      |<-- Module 09
                         |  release + budget tracker    |
                         +--------------|--------------+
                                        | (intermittent uplink,
                                        |  redacted payloads only)
                                        v
      +--------------------------------------------------------+
      |                    Edge node (per device)                |
      |                                                            |
      |  Wake-on-event trigger (always-on comparator)  <- Module 07|
      |         |                                                  |
      |         v                                                  |
      |  Inference (quantized model, NPU/MCU per hardware rev)      |
      |         | (architecture chosen per Module 08 co-design)     |
      |         v                                                  |
      |  Local anomaly score + confidence                           |
      |         |                                                  |
      |         v                                                  |
      |  Local fine-tuning buffer (last-layer adaptation) <- Module 04|
      |         |                                                  |
      |         v                                                  |
      |  Redacted telemetry batch (DP-noised aggregates only)        |
      |                                                            |
      |  Dual-slot model store, boot-time hash+version check         |
      +--------------------------------------------------------+
```

## Design decision 1: fleet heterogeneity and staged rollout, combined

The fleet spans multiple hardware revisions (Module 01), so a model
release first has to pass an eligibility filter, then a staged rollout
gate, before reaching any device:

```python
import hashlib
from dataclasses import dataclass

@dataclass
class DeviceProfile:
    device_id: str
    npu_present: bool
    ram_kb: int
    current_model_version: int

def eligible_for_release(device, model_min_ram_kb, model_requires_npu, model_version):
    if device.current_model_version >= model_version:
        return False
    if device.ram_kb < model_min_ram_kb:
        return False
    if model_requires_npu and not device.npu_present:
        return False
    return True

def rollout_cohort(device_id, rollout_percentage):
    digest = hashlib.sha256(device_id.encode()).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket < rollout_percentage

def should_receive_update(device, model_min_ram_kb, model_requires_npu,
                           model_version, rollout_percentage):
    return (eligible_for_release(device, model_min_ram_kb, model_requires_npu, model_version)
            and rollout_cohort(device.device_id, rollout_percentage))

fleet = [
    # v2 hardware (every 3rd device) ships with both an NPU and more RAM;
    # v1 hardware (the rest) has neither.
    DeviceProfile(f"node-{i:04d}", npu_present=(i % 3 == 0),
                  ram_kb=512 if i % 3 == 0 else 256, current_model_version=5)
    for i in range(200)
]

for pct in [0.05, 0.25, 1.0]:
    receiving = sum(1 for d in fleet
                     if should_receive_update(d, 400, True, 6, pct))
    eligible_total = sum(1 for d in fleet
                          if eligible_for_release(d, 400, True, 6))
    print(f"rollout {pct*100:>5.1f}%: {receiving}/{eligible_total} eligible devices selected "
          f"({eligible_total}/{len(fleet)} total devices are eligible at all)")
```

Running this prints:

```
rollout   5.0%: 5/67 eligible devices selected (67/200 total devices are eligible at all)
rollout  25.0%: 14/67 eligible devices selected (67/200 total devices are eligible at all)
rollout 100.0%: 67/67 eligible devices selected (67/200 total devices are eligible at all)
```

133 of 200 devices (the v1 hardware majority: no NPU, only 256 KB RAM)
never become eligible at all for a model requiring an NPU and 400 KB of
RAM — the rollout percentage only ever applies within the 67-device
eligible subset, which is exactly the compounding effect Module 01's
per-hardware-revision compatibility tracking exists to make visible
rather than silently mask: a "25% rollout" here reaches only 14 devices
fleet-wide, not 50, because two-thirds of the fleet was never a candidate.

## Design decision 2: privacy-respecting telemetry that still enables drift detection

Module 03's drift detection needs aggregate statistics from the fleet;
Module 09's data minimization and differential privacy constrain what
those statistics can reveal about any individual device or its
environment. Combining them means releasing DP-noised *aggregates*, never
raw per-device readings:

```python
import numpy as np

def release_fleet_confidence_histogram(all_device_confidences, bins, epsilon, rng):
    """Each device's individual confidence readings never leave it in
    raw form -- only a DP-noised count per histogram bin, aggregated
    across the whole reporting fleet, crosses the cloud boundary. This
    is the fleet-level analogue of Module 09's private_mean_release,
    applied to the confidence-drift signal from Module 03."""
    counts, _ = np.histogram(all_device_confidences, bins=bins)
    sensitivity = 1  # one device's reading can move exactly one bin's count by 1
    noisy_counts = [c + rng.laplace(0, sensitivity / epsilon) for c in counts]
    return np.array(noisy_counts)

rng = np.random.default_rng(5)
fleet_confidences = np.clip(rng.normal(0.85, 0.12, 5000), 0, 1)  # 5000 readings, fleet-wide
bins = np.linspace(0, 1, 11)

true_counts, _ = np.histogram(fleet_confidences, bins=bins)
noisy_counts = release_fleet_confidence_histogram(fleet_confidences, bins, epsilon=1.0, rng=rng)

print("true bin counts: ", true_counts)
print("DP-released:     ", np.round(noisy_counts, 1))
```

Running this prints:

```
true bin counts:  [   0    0    0    0   11   82  418 1106 1627 1756]
DP-released:      [  -1.6    2.9   -3.8    0.4   11.6   83.9  418.0 1106.0 1626.8 1752.4]
```

For the large bins (hundreds to low thousands of counts), the DP noise is
small relative to the true count — bin 8's true 1627 releases as 1626.8,
essentially undisturbed. For the near-empty low-confidence bins, though,
the *same* absolute noise scale is large relative to the (near-zero) true
count, even producing a nonsensical negative released count for a bin
whose true count was 0. This is a real, important limitation, not a bug
to smooth over: uniform per-bin noise protects sparse bins far worse in
relative terms than dense ones, so any consumer of a DP-released histogram
(Module 03's drift detector included) must expect noisy or even
implausible values in sparsely-populated bins and should not treat a
single low-count bin's exact value as trustworthy — the fleet-wide dense
bins, not the sparse ones, are where this mechanism's signal is reliable.

## Design decision 3: the update loop across a device's lifetime

Putting the whole lifecycle together for one hypothetical node:

1. **Manufacture & provisioning**: device gets a signed initial model,
   hardware profile registered in the fleet database (Module 01).
2. **Normal operation**: wake-on-event triggers inference (Module 07),
   local last-layer adaptation runs on confirmed corrections (Module 04),
   redacted + DP-noised telemetry uploads periodically (Module 09).
3. **Fleet-wide learning**: the cloud periodically runs FedAvg rounds
   using participating devices' local model deltas, improving the shared
   base model without ever centralizing raw sensor data (Module 05).
4. **Model release**: a new base model passes the eligibility and staged
   rollout gates above, widening only if canary telemetry (Module 03's
   drift signals, Module 02's error-rate gate) stays within bounds.
5. **Ongoing monitoring**: fleet-level drift aggregation flags cohort-wide
   degradation before it's visible in any single device's behavior.
6. **Incident response**: if a rollout regresses, the dual-slot rollback
   (Module 02) reverts affected devices to their previous verified model
   without a re-download, and the safety-case documentation (Module 09)
   defines what happens operationally while the issue is investigated.

## What this capstone deliberately leaves unresolved

Real fleet systems need a device-identity and authentication scheme (how
does the cloud know a telemetry payload genuinely came from the device it
claims to), a concrete transport protocol (MQTT, CoAP, or a proprietary
scheme) for the intermittent uplink, and organizational processes (who
approves a rollout widening, who's on call for a fleet-wide alert) that
are product- and company-specific rather than technical decisions this
course can make generically. What this capstone demonstrates is that the
nine preceding modules aren't nine separate techniques — they compose
into one system, each module's output feeding a specific, identifiable
input of another.

## How It Actually Works

**Why eligibility filtering and rollout-percentage sampling must compose
in that specific order, not the reverse.** `should_receive_update` first
calls `eligible_for_release` (a hard, deterministic pass/fail on
hardware/version compatibility) and only then checks `rollout_cohort` (a
probabilistic hash-bucket membership test) on whatever survives. Reversing
this order — sampling a rollout cohort from the full fleet, then checking
eligibility within it — would make the *effective* rollout percentage
depend on how much of the sampled cohort happens to be eligible, which
varies fleet to fleet and release to release in a way nobody set
intentionally. Filtering first means "25% rollout" always means exactly
"25% of the devices this release could possibly run on," which is
precisely why the module surfaces the 14-of-67 (not 14-of-200) arithmetic
explicitly: the rollout percentage's denominator is the eligible
population, and hiding that distinction would make a "25% rollout"
silently mean something different depending on fleet composition on any
given day.

**Why uniform per-bin DP noise protects sparse and dense histogram bins
asymmetrically, and why that's an unavoidable property of the mechanism,
not a tuning miss.** `release_fleet_confidence_histogram` adds
i.i.d. Laplace noise with a *fixed* scale (`sensitivity/epsilon`) to every
bin, because each bin's count has the same sensitivity — one device's
reading can move any single bin's count by at most 1, regardless of that
bin's current size. The absolute noise magnitude is therefore identical
across bins by construction, but its impact *relative to the true count*
is not: a fixed noise standard deviation is negligible against a true
count in the thousands and can dwarf (or exceed, producing the observed
negative "count") a true count of 0. This is a direct, structural
consequence of using a sensitivity-based noise calibration uniformly
across bins of wildly different magnitude — the fix (used by more
sophisticated DP histogram releases) is bin-size-aware noise or
post-processing that clips to non-negative integers, not a larger
`epsilon`, which would weaken the privacy guarantee for every bin
including the already-reliable dense ones.

**Why this system's nine modules compose into a strict dependency chain
rather than nine independent features.** Each design decision in the
capstone's lifecycle only functions correctly because an earlier module's
output is exactly the input it needs: staged rollout gates (Module 02)
only work because eligibility filtering (Module 01) has already narrowed
the candidate pool to compatible hardware; drift detection (Module 03)
only has data to work with because telemetry was collected and minimized
according to Module 09's redaction rules; federated aggregation
(Module 05) only produces a trustworthy improved model because each
device's local update came from the last-layer adaptation mechanism of
Module 04, not raw uncoordinated retraining. This is precisely why the
module states the nine modules "compose into one system" rather than
listing nine separate techniques — removing any single stage doesn't
just weaken that stage's own guarantee, it invalidates an assumption a
later stage was built on (an ineligible device receiving an update it
can't run, an unredacted payload defeating the DP layer downstream, or
an un-adapted local model polluting the federated average).

## Stretch goals

- Extend the `should_receive_update` eligibility+rollout combination to
  log *why* each ineligible device was excluded (RAM, NPU, already
  up-to-date), and produce a summary report grouping the 200-device fleet
  by exclusion reason — the kind of report a real rollout dashboard would
  show an engineer deciding whether to proceed.
- Wire the DP-noised histogram release into Module 03's `detect_drift_ks`
  by reconstructing an approximate sample from the noisy bin counts
  (sample proportionally to each bin's noisy count) and running the KS
  test against that reconstruction instead of raw data — then measure how
  much detection sensitivity is lost compared to running the KS test on
  the true underlying data, as the concrete cost of the privacy layer.
- Add a `PrivacyBudgetTracker` (from Module 09's exercise) to the
  telemetry release loop above, and simulate what happens to a device's
  reporting capability once its epsilon budget for a reporting period is
  exhausted — should it stop reporting, report with less precision, or
  fall back to only reporting whether an alert threshold was crossed
  (a single bit, minimal privacy cost) instead of a full histogram?
