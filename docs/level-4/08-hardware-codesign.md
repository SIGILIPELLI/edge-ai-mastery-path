# Hardware/Software Co-Design

Every module so far assumed the hardware — the MCU, the NPU, the sensor —
was a fixed given, and the job was to fit a model onto it. **Hardware/
software co-design** inverts part of that relationship: for a product
built at real volume, the model's architecture and the chip's capabilities
get chosen *together*, because a small change in one can unlock a large
win in the other. This module covers how that joint decision actually
gets made in practice — architecture search under hardware-aware cost
models, and the classic build-vs-buy silicon tradeoff — using a tested
Python cost model, since evaluating real ASIC/FPGA tapeouts needs
fabrication runs this environment obviously can't do.

## Why "pick the best model, then pick a chip for it" leaves performance on the table

Level 2's approach — design a good architecture, quantize it, deploy —
implicitly optimizes for accuracy first and treats latency/power as
constraints checked after the fact. Co-design instead makes hardware cost
part of the search objective from the start: two architectures with
identical accuracy can have wildly different costs on a specific target,
because operation types map onto hardware unevenly (Module 01's anchor
example: depthwise convolutions are cheap in FLOPs but can be memory-
bandwidth-bound on some NPUs, while regular convolutions are the reverse).
Searching for a model *architecture and target hardware together* finds
points on the accuracy/cost curve that searching either one alone misses.

## Hardware-aware neural architecture search: the core idea, testably

Full neural architecture search (NAS) trains thousands of candidate
networks — infeasible here. What's fully testable and carries the same
idea: given a fixed accuracy proxy per candidate architecture and a
hardware cost model for a specific target, search for the Pareto-optimal
set (no other candidate is both more accurate *and* cheaper).

```python
import numpy as np

def hardware_cost_model(depth, width, use_depthwise, target="mcu"):
    """A simplified, hand-built cost model standing in for a real
    per-op hardware simulator (e.g. what a real NAS system queries
    thousands of times per search). Different targets weight depth,
    width, and depthwise-vs-regular convs differently, mirroring how
    the same architectural choice costs differently on different chips."""
    base_flops = depth * (width ** 2)
    if use_depthwise:
        base_flops *= 0.15   # depthwise convs cost far fewer FLOPs...

    if target == "mcu":
        # ...but MCUs (no wide SIMD, CMSIS-NN kernels) don't fully
        # realize the FLOP savings -- memory access pattern dominates.
        depthwise_efficiency = 0.6 if use_depthwise else 1.0
        latency_proxy = base_flops / depthwise_efficiency
    elif target == "npu":
        # NPUs' systolic arrays are tuned for dense, regular convs;
        # depthwise convs under-utilize the array's parallelism.
        depthwise_efficiency = 0.35 if use_depthwise else 1.0
        latency_proxy = base_flops / depthwise_efficiency
    else:
        latency_proxy = base_flops
    return latency_proxy


def accuracy_proxy(depth, width, use_depthwise):
    """A simplified accuracy proxy: deeper and wider generally helps,
    depthwise convs cost a small accuracy penalty relative to full convs
    at matched FLOPs -- a real, well-documented effect in efficient-net
    literature, simplified here into a closed-form proxy for testability."""
    base = np.log(depth + 1) * np.log(width + 1)
    penalty = 0.9 if use_depthwise else 1.0
    return base * penalty


candidates = []
for depth in [4, 8, 12]:
    for width in [16, 32, 64]:
        for use_dw in [True, False]:
            candidates.append({"depth": depth, "width": width, "use_depthwise": use_dw})

for target in ["mcu", "npu"]:
    print(f"\n--- target: {target} ---")
    scored = []
    for c in candidates:
        acc = accuracy_proxy(c["depth"], c["width"], c["use_depthwise"])
        cost = hardware_cost_model(c["depth"], c["width"], c["use_depthwise"], target=target)
        scored.append({**c, "accuracy_proxy": acc, "cost_proxy": cost})

    # Pareto front: keep only candidates with no other candidate that is
    # both more accurate AND cheaper.
    pareto = []
    for cand in scored:
        dominated = any(
            other["accuracy_proxy"] >= cand["accuracy_proxy"] and
            other["cost_proxy"] <= cand["cost_proxy"] and
            other != cand
            for other in scored
        )
        if not dominated:
            pareto.append(cand)

    pareto.sort(key=lambda c: c["cost_proxy"])
    for c in pareto:
        print(f"  depth={c['depth']:>2} width={c['width']:>2} depthwise={str(c['use_depthwise']):5} "
              f"-> acc={c['accuracy_proxy']:.3f} cost={c['cost_proxy']:.1f}")
```

Running this prints the full Pareto front for each target:

```
--- target: mcu ---
  depth= 4 width=16 depthwise=True  -> acc=4.104 cost=256.0
  depth= 8 width=16 depthwise=True  -> acc=5.603 cost=512.0
  depth=12 width=16 depthwise=True  -> acc=6.540 cost=768.0
  depth= 8 width=32 depthwise=True  -> acc=6.914 cost=2048.0
  depth=12 width=32 depthwise=True  -> acc=8.072 cost=3072.0
  depth= 8 width=64 depthwise=True  -> acc=8.255 cost=8192.0
  depth=12 width=64 depthwise=True  -> acc=9.636 cost=12288.0
  depth=12 width=64 depthwise=False -> acc=10.707 cost=49152.0

--- target: npu ---
  depth= 4 width=16 depthwise=True  -> acc=4.104 cost=438.9
  depth= 8 width=16 depthwise=True  -> acc=5.603 cost=877.7
  depth=12 width=16 depthwise=True  -> acc=6.540 cost=1316.6
  depth=12 width=16 depthwise=False -> acc=7.267 cost=3072.0
  depth=12 width=32 depthwise=True  -> acc=8.072 cost=5266.3
  depth=12 width=32 depthwise=False -> acc=8.968 cost=12288.0
  depth=12 width=64 depthwise=True  -> acc=9.636 cost=21065.1
  depth=12 width=64 depthwise=False -> acc=10.707 cost=49152.0
```

The detail worth reading closely: on the MCU front, depthwise
architectures dominate almost the entire Pareto set — only the single
most-accurate point needs a full convolution. On the NPU front, full
(non-depthwise) architectures appear *earlier* in the Pareto set, at
depth=12/width=16 and depth=12/width=32, because the NPU cost model
penalizes depthwise convolutions more heavily (dividing by 0.35 instead
of 0.6) — the same depthwise-vs-full choice, at the same depth and width,
is worth taking on the MCU but is displaced by a full-conv alternative on
the NPU. This is the concrete, testable version of the co-design claim:
the two hardware cost models produce genuinely different optimal
architecture sets from the identical candidate pool, without either
architecture family being intrinsically "better."

## Build vs. buy: when co-design means choosing silicon, not just a model

At sufficient volume, co-design extends past model architecture into the
chip itself: a custom ASIC amortizes a large non-recurring engineering
(NRE) cost across units, while an off-the-shelf NPU or FPGA has higher
per-unit cost but no upfront investment. This is a standard cost-crossover
calculation, worth doing explicitly rather than assuming custom silicon
is always better at scale (it depends entirely on volume and NRE).

```python
def compute_crossover_volume(nre_cost, custom_unit_cost, off_the_shelf_unit_cost):
    """The unit volume at which building custom silicon becomes cheaper
    than buying an off-the-shelf chip, given the fixed NRE investment
    building custom silicon requires."""
    if off_the_shelf_unit_cost <= custom_unit_cost:
        return None  # custom silicon never wins if it doesn't reduce unit cost
    return nre_cost / (off_the_shelf_unit_cost - custom_unit_cost)


scenarios = [
    {"name": "small ASIC", "nre_cost": 2_000_000, "custom_unit_cost": 1.20, "off_the_shelf_unit_cost": 4.50},
    {"name": "FPGA-based", "nre_cost": 150_000, "custom_unit_cost": 8.00, "off_the_shelf_unit_cost": 4.50},
]
for s in scenarios:
    crossover = compute_crossover_volume(s["nre_cost"], s["custom_unit_cost"], s["off_the_shelf_unit_cost"])
    if crossover is None:
        print(f"{s['name']}: never cheaper than off-the-shelf at any volume")
    else:
        print(f"{s['name']}: cheaper than off-the-shelf above {crossover:,.0f} units")
```

Running this prints:

```
small ASIC: cheaper than off-the-shelf above 606,061 units
FPGA-based: never cheaper than off-the-shelf at any volume
```

The ASIC scenario needs over 600,000 units before its lower per-unit cost
pays back the $2M NRE — a volume most edge AI products never reach, which
is exactly why off-the-shelf NPUs (Modules 01-02) and general-purpose
compiler stacks (Module 06 of Level 3) are the default choice, and custom
silicon is reserved for genuinely high-volume consumer products (phones,
smart speakers) where the crossover volume is realistically achievable.

## Edge-AI tradeoffs

| Factor | Fixed hardware, optimize model only (Levels 1-3) | Hardware/software co-design |
|---|---|---|
| Search space | model architecture only | model architecture x hardware target jointly |
| Requires | one target chip, already chosen | either multiple candidate off-the-shelf targets, or the option to design silicon |
| Upfront cost | none beyond normal development | potentially large (ASIC NRE) if custom silicon is on the table |
| When it pays off | almost always worth doing given a fixed target | only clearly worth custom silicon above the NRE crossover volume |
| This module's verification | tested cost-model code; real hardware measurement not possible here | — |

## Exercise

Add a third `target="fpga"` branch to `hardware_cost_model` with its own
depthwise-efficiency weighting (FPGAs can implement custom dataflow for
depthwise convs reasonably well, unlike NPUs — a reasonable assumption to
encode as, say, `depthwise_efficiency = 0.8`). Re-run the Pareto-front
search for all three targets and confirm whether the FPGA's optimal
architecture set looks more like the MCU's or the NPU's — and think
about why that resemblance (or lack of it) makes sense given how each
platform actually executes convolutions.
