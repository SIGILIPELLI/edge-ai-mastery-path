# OTA Model Updates

Module 01 sketched a fleet architecture with a model registry pushing
updates down to devices. This module gets concrete about the mechanism:
how a model update is delivered, verified, and — critically — rolled back
if it turns out to be bad, without ever leaving a device in a state where
it can't run *any* model. Push-based OTA for ML models is the same
problem as firmware OTA, but with a twist regular firmware doesn't have:
you frequently need to A/B test a model change's *accuracy*, not just
confirm the binary booted.

## The core invariant: never brick the model slot

The single non-negotiable rule of any OTA system: a failed update must
never leave the device with no working model. The standard technique
(shared with firmware OTA generally) is the **A/B (dual-slot) pattern** —
two model storage slots, one active, one staging — so a failed or bad
update can always fall back to the previous, known-good slot.

```python
from dataclasses import dataclass
from enum import Enum
import hashlib

class SlotState(Enum):
    EMPTY = "empty"
    STAGED = "staged"       # written, not yet verified
    VERIFIED = "verified"   # hash-checked, ready to activate
    ACTIVE = "active"       # currently running
    FAILED = "failed"       # verification or boot failed

@dataclass
class ModelSlot:
    version: int = 0
    state: SlotState = SlotState.EMPTY
    sha256: str = ""

class DualSlotUpdater:
    """Models the A/B slot pattern: stage into the inactive slot, verify,
    then flip which slot is 'active' -- the previous active slot becomes
    the fallback and is never overwritten until the new one proves out."""

    def __init__(self):
        self.slots = {"A": ModelSlot(version=1, state=SlotState.ACTIVE, sha256="abc123"),
                      "B": ModelSlot()}
        self.active_slot = "A"

    def _inactive_slot(self):
        return "B" if self.active_slot == "A" else "A"

    def stage_update(self, model_bytes, expected_version):
        target = self._inactive_slot()
        actual_hash = hashlib.sha256(model_bytes).hexdigest()
        self.slots[target] = ModelSlot(version=expected_version,
                                        state=SlotState.STAGED, sha256=actual_hash)
        return target

    def verify_and_activate(self, target_slot, expected_hash):
        slot = self.slots[target_slot]
        if slot.sha256 != expected_hash:
            slot.state = SlotState.FAILED
            return {"activated": False, "reason": "hash mismatch, staged update rejected"}
        slot.state = SlotState.VERIFIED
        # Boot attempt would happen here in a real system; simulate success.
        slot.state = SlotState.ACTIVE
        previous_active = self.active_slot
        self.slots[previous_active].state = SlotState.VERIFIED  # kept as fallback
        self.active_slot = target_slot
        return {"activated": True, "new_active": target_slot,
                "fallback_available": previous_active}


updater = DualSlotUpdater()
new_model = b"pretend new model bytes v2"
expected_hash = hashlib.sha256(new_model).hexdigest()

staged_slot = updater.stage_update(new_model, expected_version=2)
print(f"staged into slot: {staged_slot}")
result = updater.verify_and_activate(staged_slot, expected_hash)
print(result)
print(f"active slot is now: {updater.active_slot}, "
      f"fallback slot A state: {updater.slots['A'].state}")
```

Running this prints:

```
staged into slot: B
{'activated': True, 'new_active': 'B', 'fallback_available': 'A'}
active slot is now: B, fallback slot A state: SlotState.VERIFIED
```

Slot A never gets erased — it sits in `VERIFIED` state as an immediately
available rollback target. This is the mechanical guarantee that makes
"never brick the model slot" actually true: rollback isn't a re-download,
it's a pointer flip back to a slot that's already fully present on-device.

## Staged (canary) rollouts across the fleet

Pushing a new model to 100% of a fleet simultaneously means a bad model
(one that regressed accuracy on a device population the training data
didn't represent well) reaches every device before anyone notices. The
standard mitigation is a **staged rollout**: push to a small percentage
first, watch telemetry (Module 03), then widen.

```python
import hashlib

def rollout_cohort(device_id: str, rollout_percentage: float) -> bool:
    """Deterministically assigns a device to the rollout cohort based on
    a hash of its ID, so the same device is consistently in or out across
    repeated checks (rather than re-rolling dice each time, which would
    make a device flicker between cohorts)."""
    digest = hashlib.sha256(device_id.encode()).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF  # -> [0.0, 1.0)
    return bucket < rollout_percentage

device_ids = [f"dev-{i:04d}" for i in range(2000)]
for pct in [0.01, 0.10, 0.50, 1.0]:
    included = sum(1 for d in device_ids if rollout_cohort(d, pct))
    print(f"rollout {pct*100:>5.1f}%: {included}/{len(device_ids)} devices included "
          f"({included/len(device_ids)*100:.1f}% actual)")
```

Running this prints:

```
rollout   1.0%: 24/2000 devices included (1.2% actual)
rollout  10.0%: 226/2000 devices included (11.3% actual)
rollout  50.0%: 1033/2000 devices included (51.6% actual)
rollout 100.0%: 2000/2000 devices included (100.0% actual)
```

The hash-bucket approach also guarantees **monotonic inclusion**: every
device in the 1% cohort is also in the 10% cohort, which is also in the
50% cohort — widening a rollout never removes a device that already
received the update, only adds more, which keeps the rollout's semantics
simple to reason about and matches what real staged-rollout systems (app
stores, cloud feature flags) do internally.

## Deciding when to widen or roll back

A rollout stage needs an explicit, automatic gate — not a person eyeballing
a dashboard — comparing the new cohort's telemetry against the previous
model's baseline, tying directly into Module 03's drift/health metrics.

```python
def rollout_gate(baseline_error_rate, canary_error_rate, min_sample_size,
                  canary_sample_size, max_relative_regression=0.10):
    """A simple automatic gate: reject widening if the canary cohort's
    error rate is more than max_relative_regression worse than baseline,
    but only once enough samples exist to trust the comparison."""
    if canary_sample_size < min_sample_size:
        return {"decision": "hold", "reason": "insufficient canary sample size"}
    relative_change = (canary_error_rate - baseline_error_rate) / baseline_error_rate
    if relative_change > max_relative_regression:
        return {"decision": "rollback", "reason": f"error rate regressed {relative_change*100:.1f}%"}
    return {"decision": "widen", "reason": "canary within acceptable bounds"}

print(rollout_gate(baseline_error_rate=0.02, canary_error_rate=0.021,
                    min_sample_size=100, canary_sample_size=250))
print(rollout_gate(baseline_error_rate=0.02, canary_error_rate=0.035,
                    min_sample_size=100, canary_sample_size=250))
print(rollout_gate(baseline_error_rate=0.02, canary_error_rate=0.05,
                    min_sample_size=100, canary_sample_size=40))
```

Running this prints:

```
{'decision': 'widen', 'reason': 'canary within acceptable bounds'}
{'decision': 'rollback', 'reason': 'error rate regressed 75.0%'}
{'decision': 'hold', 'reason': 'insufficient canary sample size'}
```

The third case matters as much as the other two: a canary error rate of
5% looks alarming, but with only 40 samples it's statistically
indistinguishable from noise — the gate correctly refuses to act on it
either way, rather than triggering a false rollback on too little data.

## Edge-AI tradeoffs

| Factor | Single-slot ("overwrite in place") | Dual-slot (A/B) OTA |
|---|---|---|
| Storage cost | one model's worth of flash | two models' worth of flash |
| Failed-update recovery | requires re-download, device may be bricked meanwhile | instant pointer-flip rollback, no re-download |
| Implementation complexity | low | moderate (slot state machine) |
| Suitable for | severely flash-constrained MCUs where 2x storage isn't affordable | anything that can spare the flash — the default choice |

## Exercise

Extend `rollout_gate` to also check a **secondary metric** — average
inference latency, not just error rate — and reject widening if either
metric regresses beyond its own threshold, independently. Then construct
a test case where error rate looks fine but latency regressed badly
(e.g. a model that's more accurate but runs 3x slower on canary devices)
and confirm your extended gate catches it, since a gate that only watches
accuracy would ship a regression the fleet architecture in Module 01
explicitly cares about.
