# Production Edge AI Architecture

Levels 1-3 built and optimized models for a single device. Level 4 shifts
the question: how do you run edge AI as a *product*, across a fleet of
potentially thousands of devices, over years, with updates, failures, and
monitoring that a single-device project never has to think about? This
module lays out the reference architecture the rest of Level 4 fills in —
OTA update delivery, fleet monitoring, on-device learning, privacy
compliance — so each later module has a concrete place in the whole system
rather than existing as an isolated technique.

## The reference architecture

```
                     +-------------------+
                     |   Cloud backend    |
                     |  - model registry  |
                     |  - OTA distribution|  <- Module 02
                     |  - fleet telemetry |  <- Module 03
                     |  - training/       |
                     |    fine-tuning     |  <- Modules 04-05
                     +---------|----------+
                               | (intermittent, often low-bandwidth)
                               v
      +------------------------------------------------+
      |                  Edge device                     |
      |  +------------+   +----------------+             |
      |  | Model store |->| Inference       |--> app      |
      |  | (versioned, |   | runtime         |    logic    |
      |  |  signed)    |   | (Level 3 stack) |             |
      |  +------------+   +----------------+             |
      |         ^                  |                       |
      |         |                  v                       |
      |  +------------+   +----------------+             |
      |  | Local learn |<--| Telemetry       |--> (batched|
      |  | / adapt     |   | + drift signals |   uplink)  |
      |  | (Module 04) |   | (Module 03)     |             |
      |  +------------+   +----------------+             |
      +------------------------------------------------+
```

The two arrows crossing the cloud/device boundary — OTA updates going
down, telemetry going up — are both **intermittent and bandwidth-
constrained** by assumption. Every design decision in this module and the
next several follows from taking that constraint seriously rather than
assuming the connectivity a typical cloud service can rely on.

## Why "just retrain in the cloud and push a new model" doesn't scale as-is

The naive fleet-update loop — collect data, retrain centrally, push a new
model to every device — has three problems that only show up past a
handful of devices:

1. **Bandwidth**: pushing a multi-megabyte model to 10,000 devices over
   cellular or constrained WiFi is expensive and slow, and doing it
   simultaneously can saturate shared network infrastructure (Module 02
   covers staged/canary rollouts specifically to avoid this).
2. **Heterogeneity**: a fleet accumulates hardware revisions over its
   lifetime — v1 boards with an older NPU, v2 boards with more RAM — and
   "one model for the fleet" stops being true the moment hardware
   diverges, which it always eventually does.
3. **Silent failure**: a model that regresses on a subset of devices (a
   specific sensor batch, a specific lighting condition, a specific
   firmware version) produces no exception, no crash log — just quietly
   worse decisions, which is exactly what Module 03's drift detection
   exists to catch.

## Modeling fleet heterogeneity as a first-class design input

A useful discipline before writing any fleet-management code: enumerate
the actual axes of variation your fleet has, because "the fleet" is never
actually homogeneous in a real deployment.

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DeviceProfile:
    device_id: str
    hardware_rev: str          # "v1", "v2", ...
    npu_present: bool
    ram_kb: int
    current_model_version: int
    firmware_version: str
    last_seen_unix: Optional[float] = None

def eligible_for_model(device: DeviceProfile, model_min_ram_kb: int,
                        model_requires_npu: bool, model_version: int) -> dict:
    """A device is eligible for a given model release only if its
    hardware profile can actually run it -- this check has to happen
    before an OTA push is even attempted (Module 02), not discovered
    after a failed update."""
    if device.current_model_version >= model_version:
        return {"eligible": False, "reason": "already up to date or newer"}
    if device.ram_kb < model_min_ram_kb:
        return {"eligible": False, "reason": "insufficient RAM for this model"}
    if model_requires_npu and not device.npu_present:
        return {"eligible": False, "reason": "model requires NPU, device has none"}
    return {"eligible": True, "reason": "compatible"}


fleet = [
    DeviceProfile("dev-001", "v1", npu_present=False, ram_kb=256, current_model_version=3, firmware_version="1.2.0"),
    DeviceProfile("dev-002", "v2", npu_present=True, ram_kb=512, current_model_version=3, firmware_version="1.4.0"),
    DeviceProfile("dev-003", "v2", npu_present=True, ram_kb=512, current_model_version=4, firmware_version="1.4.0"),
]

for d in fleet:
    result = eligible_for_model(d, model_min_ram_kb=400, model_requires_npu=True, model_version=4)
    print(f"{d.device_id}: {result}")
```

Running this prints:

```
dev-001: {'eligible': False, 'reason': 'insufficient RAM for this model'}
dev-002: {'eligible': True, 'reason': 'compatible'}
dev-003: {'eligible': False, 'reason': 'already up to date or newer'}
```

This 3-line eligibility table is a small example of a real production
requirement: a fleet's model registry has to track *per-hardware-revision*
compatibility, not just a single "latest model" pointer, or older-hardware
devices either fail an update or silently receive a model they can't
correctly run.

## Local-first vs. cloud-dependent inference

A design decision that ripples through everything else: does inference
work when the device is offline? A camera node that only detects objects
while connected to the cloud is not really an edge AI system — it's a
thin client with extra steps. The reference architecture above assumes
inference is always local (the whole reason Levels 1-3 exist), and the
cloud connection is used only for the slower-moving concerns: model
updates, telemetry, and periodic retraining data collection. This single
assumption is what makes the rest of Level 4 tractable — every later
module's designs (OTA, drift detection, federated learning) exist
precisely because the device must keep working correctly on its own
between connections.

## Edge-AI tradeoffs

| Factor | Single-device deployment (Levels 1-3) | Fleet production architecture (Level 4) |
|---|---|---|
| Model update mechanism | manual reflash | staged OTA with rollback (Module 02) |
| Failure visibility | you're watching it directly | requires telemetry + drift detection (Module 03) |
| Hardware assumption | one known board | heterogeneous fleet, versioned compatibility |
| Retraining loop | offline, one-off | ongoing, sometimes on-device (Modules 04-05) |
| Data handling | whatever's convenient during development | privacy/compliance constraints from day one (Module 09) |

## How It Actually Works

**Why "always local inference, cloud only for slow-moving concerns" is
the architectural decision that makes every other Level 4 module
tractable.** If inference itself depended on a live cloud connection, the
system's correctness would be coupled to network availability at the
same timescale as its core function — a dropped connection would mean
the device simply stops working, at exactly the moments (storms,
congested networks, remote installations) reliability matters most. By
keeping the entire Level 1-3 inference stack self-contained on-device and
routing only OTA updates and telemetry through the network, the system's
correctness becomes decoupled from connectivity at a much slower
timescale: a device can go offline for hours or days and keep making
correct local decisions the whole time, only falling behind on updates
and monitoring — a degraded-but-functioning state, not a failed one. This
single property is what allows Module 02's staged rollouts, Module 03's
drift detection, and Module 04's on-device learning to all treat network
access as optional and intermittent rather than a hard dependency.

**Why hardware eligibility must be checked against the device's actual
profile before an OTA push, not discovered from a failed boot
afterward.** A model compiled or quantized with a specific tensor arena
size assumption, or one that requires an NPU delegate to hit its latency
target, is not a "smaller/bigger" variant of a model that runs anywhere —
it is a binary artifact tied to specific hardware capabilities
(available RAM for the arena, presence of specific accelerator hardware).
Attempting to load such a model on an incompatible device fails at
`AllocateTensors()` or delegate-creation time (Level 1 Module 06, Level 3
Module 02) — a failure that, without the `eligible_for_model` check
performed centrally before the push, would have to be detected and
recovered from independently by every incompatible device in the fleet,
turning one preventable mismatch into thousands of individual on-device
failure-and-rollback events instead of zero pushes to ineligible devices
in the first place.

**Why version comparison ("already up to date or newer") must be a
first-class field, not inferred from hash equality.** Two different model
builds can be functionally equivalent-or-better yet produce completely
different hash values (different training runs, different but equally
valid quantization outcomes) — hash equality only proves "identical
bytes," never "this device doesn't need this update" or "this update is
older than what's installed." Tracking `current_model_version` as an
explicit monotonically increasing integer, checked before any hash or
compatibility work is even attempted, is what lets a fleet safely skip
redundant pushes to already-current devices and — as the exercise's
semver-ordering extension makes concrete — correctly reject a
numerically-smaller version string even when naive string comparison
would get the ordering of `"1.2.0"` vs `"1.10.0"` wrong (since
lexicographic string comparison treats `"1.10.0"` as less than `"1.2.0"`,
comparing the character `'1'` against `'2'` at the second position).

## Exercise

Extend `DeviceProfile` and `eligible_for_model` with a third eligibility
axis: `min_firmware_version` (a model release may require a firmware
feature — say, a new sensor driver — introduced only in a specific
firmware version). Write a version-comparison helper that correctly
orders semver-style strings like `"1.2.0"` and `"1.10.0"` (a naive string
comparison gets this wrong), and add a test case in the fleet list above
that fails eligibility purely on firmware version while passing every
other check.
