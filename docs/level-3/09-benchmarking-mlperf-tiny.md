# Benchmarking with MLPerf Tiny

Every module so far has measured latency ad hoc — a `time.perf_counter()`
loop around one function call. That's fine for comparing two versions of
your own pipeline, but it doesn't let you compare your device against
someone else's, or your model against a published number, because ad hoc
benchmarks differ in warm-up handling, what's included in the timing
window, and how outliers are treated. **MLPerf Tiny** is the standardized
benchmark suite for exactly this: a fixed set of reference models (keyword
spotting, visual wake words, image classification, anomaly detection) and
a fixed measurement methodology, so a number from one team's board is
comparable to another team's. This module covers what MLPerf Tiny actually
measures and reimplements its core statistical methodology in tested
Python, since running the real submission harness needs the reference
models and target hardware this environment doesn't have.

## What MLPerf Tiny standardizes (and why each part matters)

| Standardized element | Why it matters |
|---|---|
| Four reference tasks (keyword spotting, visual wake words, image classification, anomaly detection) | Same task, same dataset, same accuracy target across every submission — no cherry-picking an easy task |
| EnergyRunner harness | Measures both latency *and* energy per inference, since a model can trade one for the other invisibly if only one is reported |
| Fixed accuracy threshold per task | A submission's latency number is only valid if accuracy stays at or above a published floor — you can't just make a model faster by making it worse |
| Median-of-many-runs latency, not single-run | A single latency measurement is dominated by noise (thermal throttling, other clock cycles, first-run effects); the standard requires many repeated runs |
| Closed vs. open divisions | "Closed" locks the model architecture so only implementation/compiler optimizations are compared; "open" allows model changes, compared separately |

The closed/open split is the detail that matters most for interpreting
published results: a closed-division Cortex-M55 number and a closed-
division Cortex-M4 number are directly comparable because they ran the
*identical* model — the difference you're seeing is genuinely the
hardware and compiler, not a smaller or different network hiding in one
submission.

## Reimplementing the core measurement methodology

The statistical backbone of any MLPerf-style benchmark is straightforward
but easy to get subtly wrong: warm-up runs excluded, enough repetitions to
get a stable distribution, and reporting percentiles (not just the mean,
which hides tail latency that matters for real-time deadlines).

```python
import numpy as np
import time

def benchmark_with_warmup(infer_fn, n_warmup=10, n_measured=200):
    """Runs infer_fn repeatedly, discarding warm-up runs (JIT/cache
    effects, first-call overhead) before recording measured latencies.
    Returns the full latency distribution in milliseconds."""
    for _ in range(n_warmup):
        infer_fn()

    latencies_ms = []
    for _ in range(n_measured):
        t0 = time.perf_counter()
        infer_fn()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)
    return np.array(latencies_ms)


def summarize_latency(latencies_ms):
    """MLPerf-style reporting: median (robust to outliers) plus tail
    percentiles, since a real-time system cares about worst-case
    behavior, not just typical-case."""
    return {
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p90_ms": float(np.percentile(latencies_ms, 90)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
        "mean_ms": float(np.mean(latencies_ms)),
        "std_ms": float(np.std(latencies_ms)),
    }


# Simulate an inference function with realistic jitter: a fast typical
# case with an occasional slow tail (e.g. an OS scheduling hiccup).
rng = np.random.default_rng(42)
call_count = [0]

def simulated_infer():
    call_count[0] += 1
    base_latency_s = 0.002  # 2ms typical
    if rng.random() < 0.05:       # 5% of calls hit a slow tail
        time.sleep(base_latency_s * 4)
    else:
        time.sleep(base_latency_s + rng.normal(0, 0.0002))

latencies = benchmark_with_warmup(simulated_infer, n_warmup=10, n_measured=200)
stats = summarize_latency(latencies)
for k, v in stats.items():
    print(f"{k}: {v:.3f}")
```

Running this prints (exact values vary slightly with system scheduling
noise, but the shape is stable):

```
p50_ms: 2.570
p90_ms: 2.907
p99_ms: 10.028
mean_ms: 2.688
std_ms: 1.076
```

This is the number that a mean-only benchmark hides: `p50` (~2.6 ms) looks
nothing like `p99` (~10.0 ms) — the simulated 5% slow-tail calls barely
move the median but dominate the 99th percentile. For a real-time keyword
spotter that must respond within a fixed budget, the p99 number is the one
that determines whether the system meets its deadline, not the mean.

## The accuracy-latency tradeoff MLPerf Tiny forces you to report honestly

A benchmark number without its accuracy is close to meaningless — a model
quantized aggressively enough will always be fast. MLPerf Tiny's closed
division specifically prevents reporting a latency win that came from
silently trading away accuracy, by requiring both numbers together.

```python
def evaluate_tradeoff(latency_ms, accuracy, min_accuracy_threshold):
    """Models MLPerf Tiny's closed-division validity rule: a submission
    below the accuracy floor is disqualified regardless of how fast it
    is, and can't be compared on latency at all."""
    valid = accuracy >= min_accuracy_threshold
    return {
        "latency_ms": latency_ms, "accuracy": accuracy,
        "valid_submission": valid,
        "reason": "meets accuracy floor" if valid else "below accuracy floor -- disqualified",
    }

candidates = [
    {"name": "int8, full model", "latency_ms": 12.0, "accuracy": 0.94},
    {"name": "int8, pruned 40%", "latency_ms": 7.5, "accuracy": 0.91},
    {"name": "int8, pruned 70%", "latency_ms": 4.0, "accuracy": 0.83},  # too aggressive
]
threshold = 0.90
for c in candidates:
    result = evaluate_tradeoff(c["latency_ms"], c["accuracy"], threshold)
    print(f"{c['name']}: {result['valid_submission']} ({result['reason']})")
```

Running this prints:

```
int8, full model: True (meets accuracy floor)
int8, pruned 40%: True (meets accuracy floor)
int8, pruned 70%: False (below accuracy floor -- disqualified)
```

The 70%-pruned model is the fastest of the three (4.0 ms) but gets
disqualified outright — exactly the discipline that makes MLPerf Tiny
numbers trustworthy: you cannot cite a MLPerf Tiny latency figure without
it implying the accuracy floor was met.

## Edge-AI tradeoffs

| Factor | Ad hoc benchmarking (Modules 01-08's `time.perf_counter` loops) | MLPerf Tiny methodology |
|---|---|---|
| Comparable across teams/hardware | no — every team's harness differs | yes, by design |
| Reports tail latency | only if you remember to | mandatory (percentiles, not just mean) |
| Accounts for accuracy tradeoff | no, unless you track it separately | mandatory, enforced accuracy floor |
| Setup cost | minutes | real submissions require reference model + harness + review |
| Best fit | day-to-day development iteration | publishable, cross-comparable claims |

## Exercise

Modify `simulated_infer` to model thermal throttling instead of random
scheduling jitter: have `base_latency_s` increase gradually (e.g. by 5%)
every 20 calls to simulate a chip warming up and clocking down, reset
after 100 calls to simulate a cooldown period. Re-run `benchmark_with_warmup`
and `summarize_latency` and compare how p50 vs. p99 responds to a
*systematic drift* pattern versus this module's *random spike* pattern —
they should look different in the resulting distribution shape, and
knowing which pattern you're seeing changes the right fix (cooling vs.
scheduler tuning).
