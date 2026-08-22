# Model Security on Edge Devices

Every previous module assumed the model file, once deployed, is trusted
and safe. Edge deployment breaks that assumption in a way cloud inference
never has to deal with: **the model file physically ships to a device an
attacker can hold in their hands.** Unlike a cloud API where the weights
never leave your servers, a `.tflite` file flashed onto a consumer device
can be extracted, read, and reverse-engineered by anyone with a firmware
dumper and patience. This module covers the three practical attack
surfaces — model extraction, adversarial inputs, and tampering — and what
defenses actually change the economics for an attacker rather than just
adding friction.

## Attack surface 1: model extraction

Flash memory on most microcontrollers and SoCs is readable with commodity
tools (a JTAG probe, a flash programmer, sometimes just a serial bootloader
with debug access left enabled). Once an attacker has the raw
`.tflite`/`.onnx` bytes, they have your architecture, your quantization
scheme, and your trained weights — everything needed to run your model
themselves, fine-tune it, or study it for adversarial weaknesses.

```python
"""
Demonstrates why 'just don't leave debug ports open' is necessary but not
sufficient: even a well-protected device's model file, once extracted by
any means, has no cryptographic protection unless you added it. This
inspects a real .tflite's raw bytes to show the model's structure is
plainly visible with no protection applied -- illustrative of the
extraction risk, not an attack tool.
"""
import struct

def inspect_flatbuffer_header(tflite_bytes):
    """A .tflite file is a flatbuffer; the first 4 bytes are a root table
    offset, followed shortly by a file-identifier string. No secrecy is
    applied to any of this by default."""
    root_offset = struct.unpack("<I", tflite_bytes[0:4])[0]
    identifier = tflite_bytes[4:8]
    return {"root_table_offset": root_offset, "file_identifier": identifier}

# Any .tflite file demonstrates this -- the format is fully documented and
# requires no key or credential to parse.
print("A .tflite file's structure is openly documented (flatbuffers schema); "
      "extraction requires no cryptographic bypass, only physical/firmware access.")
```

The practical defense is encrypting the model file at rest and decrypting
it into RAM only at load time, using a key held in the chip's secure
element or a hardware-backed key store (Arm TrustZone, a discrete secure
element, or a vendor's OTP-fused key). This raises the bar from "dump
flash" to "extract the key from tamper-resistant hardware," which is a
categorically harder attack, not just a slower one.

## Attack surface 2: adversarial inputs

Even a model an attacker can't extract can be attacked purely through its
inputs — small, deliberately crafted perturbations that cause a
misclassification while looking unchanged (or unremarkable) to a human or
sensor. This matters more at the edge than in the cloud because edge
models are frequently the *only* line of defense (a camera-based access
control gate, a wake-word detector guarding an action), with no
server-side second opinion to catch anomalies.

```python
"""
A minimal, deliberately simplified demonstration of adversarial
perturbation on a toy linear classifier -- not a real vision attack
(which needs gradient access to a real CNN this environment can't train),
but the same core idea: a tiny, bounded perturbation in the direction of
the model's own gradient flips a decision.
"""
import numpy as np

def toy_linear_classifier(x, w, b):
    return 1 if np.dot(w, x) + b > 0 else 0

def fgsm_style_perturbation(x, w, epsilon):
    """Fast-Gradient-Sign-Method-style step: for a linear model the
    gradient of the decision w.r.t. input is just w itself, so the
    perturbation direction is sign(w)."""
    return x + epsilon * np.sign(w)

rng = np.random.default_rng(0)
w = rng.standard_normal(5)
b = -0.2
x = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

original_pred = toy_linear_classifier(x, w, b)
x_adv = fgsm_style_perturbation(x, w, epsilon=0.3)
adv_pred = toy_linear_classifier(x_adv, w, b)

print(f"original input: {x}, prediction: {original_pred}")
print(f"perturbed input: {x_adv}, prediction: {adv_pred}")
print(f"max perturbation magnitude: {np.max(np.abs(x_adv - x)):.2f}")
```

Running this prints:

```
original input: [0.1 0.1 0.1 0.1 0.1], prediction: 0
perturbed input: [ 0.4 -0.2  0.4  0.4 -0.2], prediction: 1
max perturbation magnitude: 0.30
```

A bounded perturbation (each element moved by at most 0.3) flips the
decision entirely. Real CNN attacks use the same principle against actual
gradients computed via backpropagation, and are far less visually obvious
than this toy example. Edge-specific defenses that actually help:

- **Input sanity bounds** — reject inputs outside the sensor's known
  physical range (a temperature sensor reporting 300°C on a device rated
  for -20 to 60°C is a signal, not a valid reading), catching a class of
  attacks before they ever reach the model.
- **Adversarial training** — training on a mix of clean and perturbed
  examples so the model's decision boundary is less brittle; a training-time
  fix, not a deployment-time one, so it must be planned into Level 1's
  training pipeline, not bolted on after.
- Treating a model's raw confidence score with suspicion — many
  adversarial examples produce a confidently wrong answer rather than an
  uncertain one, so "the model was very sure" is not, by itself, evidence
  the answer is correct.

## Attack surface 3: tampering and integrity

Between "an attacker extracts your model" and "an attacker attacks your
model with inputs" sits a third case: an attacker **replaces** your model
file with a different one (a backdoored version, an intentionally
degraded one, or someone else's model entirely) via a compromised OTA
update or physical flash access.

```python
import hashlib

def verify_model_integrity(model_bytes, expected_sha256):
    """The minimum viable defense: never load a model without checking
    its hash against a value obtained through a trusted channel (signed
    at build time, verified against a certificate chain at boot)."""
    actual = hashlib.sha256(model_bytes).hexdigest()
    return actual == expected_sha256, actual

model_bytes = b"pretend this is a .tflite file's raw bytes"
expected = hashlib.sha256(model_bytes).hexdigest()  # normally shipped separately, signed
tampered_bytes = model_bytes + b"\x00"  # simulates one byte of tampering

ok, actual = verify_model_integrity(model_bytes, expected)
print(f"untampered check: {ok}")
ok2, actual2 = verify_model_integrity(tampered_bytes, expected)
print(f"tampered check: {ok2}")
```

This prints `untampered check: True` then `tampered check: False`,
correctly catching the modification. A hash check alone only proves the
file matches *some* expected value — it's only a security control if the
expected hash itself arrives through a channel the attacker can't also
tamper with (a signed manifest verified with a public key baked into the
device's boot ROM, not a hash fetched over the same unauthenticated
channel as the model file).

## Edge-AI tradeoffs

| Attack | Defense | Cost |
|---|---|---|
| Model extraction (flash dump) | encrypt-at-rest + secure-element key | needs hardware secure element or TrustZone; not free on cheap MCUs |
| Adversarial inputs | input sanity bounds, adversarial training | training-time cost + can reduce clean accuracy slightly |
| Tampering / malicious OTA update | signed manifest + hash verification at boot | needs a boot ROM/bootloader that enforces signature checks |
| All three | defense in depth (all of the above together) | full stack cost; still not "unbreakable," only "expensive to break" |

## Exercise

Take the `verify_model_integrity` function and extend it to also check a
**version number** embedded in a small unencrypted header before the model
bytes, rejecting any file claiming an older version than the device's
currently-installed one — this is the standard defense against a
**rollback attack**, where an attacker reflashes a known-vulnerable but
validly-signed older model version instead of forging a new one. Write a
test that rejects a validly-hashed but older-versioned file.
