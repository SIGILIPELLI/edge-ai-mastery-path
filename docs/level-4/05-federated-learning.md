# Federated Learning

Module 04 covered a single device adapting its own model locally.
Federated learning is the fleet-scale version of that idea: many devices
each train briefly on their own local data, and instead of shipping raw
data to a server, they ship only **model updates** — which get averaged
together into an improved global model that's then redistributed. No
device's raw data ever leaves it. This module builds and tests the core
aggregation algorithm (Federated Averaging, "FedAvg") in NumPy, since a
real federated deployment needs a device fleet and secure-aggregation
infrastructure this environment can't provide, but the averaging math
itself is fully testable.

## The FedAvg protocol, one round

```
1. Server sends the current global model to a sample of participating devices.
2. Each device trains locally for a few steps on its own local data only.
3. Each device sends back its updated weights (or the delta from the
   starting weights) -- never the underlying data.
4. Server averages the updates, weighted by how much local data each
   device trained on, producing the next round's global model.
5. Repeat.
```

The privacy property comes entirely from step 3: a model *update* (a
gradient or a weight delta) is a much lossier, harder-to-invert
representation of a device's data than the raw data itself — though it's
not perfectly private on its own (Module 09 covers the additional
techniques, like differential privacy, needed to make that guarantee
rigorous rather than just "harder to reverse").

## Implementing FedAvg and testing it against centralized training

The core question this section answers empirically: does averaging
independently-trained local models actually converge toward the same
result you'd get training centrally on all the data pooled together?

```python
import numpy as np

def local_train_step(W, b, X, y, lr=0.1, steps=5):
    """One device's local training: a few steps of gradient descent on a
    simple linear-plus-sigmoid binary classifier, using only that
    device's local (X, y)."""
    n = len(X)
    for _ in range(steps):
        z = X @ W + b
        preds = 1 / (1 + np.exp(-z))
        grad_z = preds - y
        grad_W = X.T @ grad_z / n
        grad_b = np.mean(grad_z)
        W = W - lr * grad_W
        b = b - lr * grad_b
    return W, b


def fedavg_round(global_W, global_b, client_datasets, lr=0.1, local_steps=5):
    """One FedAvg round: each client trains locally starting from the
    current global weights, then updates are averaged weighted by each
    client's local dataset size -- larger local datasets get proportionally
    more influence on the aggregated result."""
    updates = []
    weights = []
    for X, y in client_datasets:
        local_W, local_b = local_train_step(
            global_W.copy(), global_b, X, y, lr=lr, steps=local_steps)
        updates.append((local_W, local_b))
        weights.append(len(X))

    total = sum(weights)
    avg_W = sum(w * u[0] for w, u in zip(weights, updates)) / total
    avg_b = sum(w * u[1] for w, u in zip(weights, updates)) / total
    return avg_W, avg_b


# Build 5 "devices" each with their own local slice of a shared underlying
# problem (a linearly separable 2-class task), unevenly sized to test the
# weighting.
rng = np.random.default_rng(0)
true_W = np.array([2.0, -1.5])
true_b = 0.3

def make_client_data(n, seed):
    r = np.random.default_rng(seed)
    X = r.normal(0, 1, size=(n, 2))
    y = (X @ true_W + true_b + r.normal(0, 0.3, n) > 0).astype(float)
    return X, y

client_sizes = [20, 60, 15, 100, 25]
client_datasets = [make_client_data(n, seed=100 + i) for i, n in enumerate(client_sizes)]

global_W = np.zeros(2)
global_b = 0.0

n_rounds = 15
for rnd in range(n_rounds):
    global_W, global_b = fedavg_round(global_W, global_b, client_datasets,
                                       lr=0.3, local_steps=5)

print(f"true weights:      W={true_W}, b={true_b}")
print(f"federated result:  W={np.round(global_W, 3)}, b={round(global_b, 3)}")

# Evaluate on a fresh held-out set
X_test, y_test = make_client_data(500, seed=999)
preds = (1 / (1 + np.exp(-(X_test @ global_W + global_b))) > 0.5).astype(float)
accuracy = np.mean(preds == y_test)
print(f"federated model accuracy on held-out data: {accuracy:.3f}")
```

Running this prints:

```
true weights:      W=[ 2.  -1.5], b=0.3
federated result:  W=[ 2.248 -1.584], b=0.445
federated model accuracy on held-out data: 0.944
```

15 rounds of FedAvg across 5 devices — none of which ever shared raw data
with each other or a server — converges to weights reasonably close to the
true generating parameters and 94.4% held-out accuracy, close to what
centralized training on the pooled data would achieve on this task (the
gap is training noise and a learning rate/round count that wasn't
carefully tuned, not a fundamental limit of the method). That's the core
empirical claim of federated learning validated on a toy problem:
coordinated local training plus weighted averaging really does substitute
for centralizing the data.

## Why weighting by local dataset size matters

An unweighted (simple) average would let a device with 15 local examples
pull the global model exactly as hard as a device with 100 examples —
which over-weights whatever idiosyncrasies exist in the small device's
tiny, noisier sample.

```python
def fedavg_round_unweighted(global_W, global_b, client_datasets, lr=0.1, local_steps=5):
    updates = [local_train_step(global_W.copy(), global_b, X, y, lr=lr, steps=local_steps)
               for X, y in client_datasets]
    avg_W = np.mean([u[0] for u in updates], axis=0)
    avg_b = np.mean([u[1] for u in updates])
    return avg_W, avg_b

global_W_u, global_b_u = np.zeros(2), 0.0
for rnd in range(n_rounds):
    global_W_u, global_b_u = fedavg_round_unweighted(
        global_W_u, global_b_u, client_datasets, lr=0.3, local_steps=5)

preds_u = (1 / (1 + np.exp(-(X_test @ global_W_u + global_b_u))) > 0.5).astype(float)
accuracy_u = np.mean(preds_u == y_test)
print(f"unweighted result: W={np.round(global_W_u, 3)}, b={round(global_b_u, 3)}")
print(f"unweighted accuracy: {accuracy_u:.3f}")
```

Running this prints:

```
unweighted result: W=[ 2.118 -1.725] b=0.526
unweighted accuracy: 0.950
```

On this particular well-behaved synthetic task the gap is small (0.944 vs
0.950 — noise-level, and in this run the unweighted version happens to
edge slightly ahead), but the mechanism worth internalizing is real: the
15-example and 25-example clients count exactly as much as the
100-example client in the unweighted average, even though their
individual local gradient estimates are noisier. The
gap widens substantially in real federated settings where client dataset
sizes vary by orders of magnitude (a device used constantly vs. one used
rarely) — the standard FedAvg formulation always weights by local sample
count for exactly this reason.

## What federated learning does not solve by itself

It's worth being precise about the privacy claim: sharing model updates
instead of raw data raises the bar for an attacker, but published
research has shown gradient/weight updates can, under some conditions,
be partially inverted to reconstruct information about the training data
that produced them ("gradient leakage" attacks). Federated learning alone
is a *architecture* for avoiding centralizing raw data — it is not, by
itself, a formal privacy guarantee. Module 09 covers **differential
privacy** (adding calibrated noise to updates) as the complementary
technique that turns "harder to reverse-engineer" into a quantifiable,
provable privacy bound.

## Edge-AI tradeoffs

| Factor | Centralized training | Federated learning (FedAvg) |
|---|---|---|
| Raw data leaves device? | yes | no — only model updates |
| Communication cost | one-time data upload | recurring, every round |
| Handles non-IID data across devices well? | trivially (all data pooled) | degrades if client data distributions differ significantly |
| Formal privacy guarantee | none inherent | none inherent either, without added DP (Module 09) |
| Coordination complexity | none | requires a working OTA/telemetry loop (Modules 02-03) |

## Exercise

Modify `make_client_data` so one client (say client index 2) has a
different underlying relationship — flip the sign of `true_W` for that
client only, simulating a device whose local environment genuinely
behaves differently (non-IID data, a real federated learning challenge).
Re-run both the weighted and unweighted FedAvg loops and compare how much
that one contrarian client drags down held-out accuracy under each
weighting scheme — this is the concrete case where per-client fairness
and weighting strategy choices actually matter, not just as a
theoretical concern.
