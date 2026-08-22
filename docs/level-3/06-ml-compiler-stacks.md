# ML Compiler Stacks (TVM, IREE)

Modules 01-02 covered vendor-specific compilers (`edgetpu_compiler`, NXP's
eIQ Neutron compiler, Arm's Vela) — each one built for exactly one
accelerator family, with a fixed, hand-written set of supported ops. This
module covers the other branch of the tree: **general-purpose ML
compiler stacks** like Apache TVM and Google's IREE, which take a model
from *any* framework and generate optimized code for *any* target
(CPU, GPU, or accelerator) by treating "compile a neural network" as a
real compiler problem — with an intermediate representation, optimization
passes, and code generation — rather than a fixed op-mapping table.

## Why a fixed op table stops scaling

Vendor compilers work well because they're narrow: eIQ only needs to
handle ops that show up in models people actually deploy on i.MX chips.
But every new architecture idea (a new activation function, a fused
attention block, a novel normalization layer) means writing a new
hand-tuned kernel for every target you support — an O(frameworks x
targets) problem. General-purpose compiler stacks instead define one
**intermediate representation (IR)** that any frontend can lower into, and
one set of backend code generators that any IR graph can lower out of —
turning the problem into O(frameworks + targets) by construction.

```
PyTorch model  \                                  / CPU (LLVM codegen)
TensorFlow model -- frontend importers --> IR --> optimization passes --> GPU (CUDA/Metal)
ONNX model     /                                  \ NPU (custom backend)
```

TVM calls its IR **Relay** (graph-level) lowering to **TIR** (loop-level,
tensor-index representation). IREE compiles through MLIR dialects,
lowering progressively from a framework-level representation down to
target-specific machine code. Both follow the same shape: import once,
optimize in a target-agnostic middle layer, generate code many times.

## Optimization passes: the part that actually earns the speedup

The interesting engineering isn't the IR itself, it's the graph-rewriting
passes that run on it before code generation — the same kind of pass a
general-purpose compiler runs (constant folding, dead code elimination),
specialized for tensor graphs. The two with the biggest real-world impact:

**Operator fusion** — merging adjacent ops (e.g. Conv2D + BiasAdd + ReLU)
into a single generated kernel, so intermediate results never round-trip
through memory between ops.

```python
"""
Illustrative model of what a fusion pass decides, not a real TVM/IREE
call (those require the actual compiler installed and a target device;
this environment doesn't have either). Manually reviewed against how
TVM's Relay fusion pass documents its own behavior -- not executed
against the real compiler.
"""

FUSIBLE_PAIRS = {
    ("CONV_2D", "BIAS_ADD"), ("BIAS_ADD", "RELU"),
    ("CONV_2D", "RELU"), ("MATMUL", "BIAS_ADD"),
}

def greedy_fuse(op_sequence):
    """op_sequence: list of op-type strings in graph order.
    Returns a list of fused groups (each a list of op names)."""
    groups = [[op_sequence[0]]]
    for op in op_sequence[1:]:
        prev = groups[-1][-1]
        if (prev, op) in FUSIBLE_PAIRS:
            groups[-1].append(op)
        else:
            groups.append([op])
    return groups

graph = ["CONV_2D", "BIAS_ADD", "RELU", "MAX_POOL_2D",
         "CONV_2D", "BIAS_ADD", "RELU"]
fused = greedy_fuse(graph)
print(fused)
# [['CONV_2D', 'BIAS_ADD', 'RELU'], ['MAX_POOL_2D'],
#  ['CONV_2D', 'BIAS_ADD', 'RELU']]
```

Three separate ops on each side collapse into one fused kernel — in a real
compiler this typically means 2 fewer round trips to memory per fused
group, which on a bandwidth-starved microcontroller (where SRAM bandwidth,
not compute, is often the actual bottleneck) can matter more than any
compute-side optimization.

**Auto-tuning** — rather than hand-writing a loop tiling/ordering strategy
per op per target, TVM's AutoTVM/Ansor and similar systems generate many
candidate implementations of the same op (different loop orders, tile
sizes, vectorization widths) and *measure* which one runs fastest on the
actual target hardware, keeping the winner. This is a genuinely different
philosophy from every compiler in Modules 01-02: instead of a human
encoding "this is the right kernel for this chip," the compiler searches
and benchmarks its way to an answer, per target, per operator shape.

## Where this fits versus TFLite/ONNX Runtime

It's tempting to think of TVM/IREE as a replacement for TFLite or ORT;
in practice they usually sit one layer lower, sometimes *generating* the
kernels a runtime like TFLite Micro links against, rather than replacing
the runtime's model-loading and scheduling logic. The realistic decision
tree:

- Deploying to a chip with an existing, well-supported vendor toolchain
  (Coral, Ethos-U, i.MX) → use that toolchain (Modules 01-02). It's more
  mature and better documented for that one target.
- Deploying the same model to several unrelated targets (say, an x86
  gateway, an Arm Cortex-A board, and a RISC-V microcontroller) from one
  build pipeline → a general-purpose compiler stack earns its complexity,
  because the alternative is maintaining three separate toolchains.
- Need a kernel for an operator no vendor toolchain supports at all
  (a novel op from a research paper) → auto-tuning compilers can generate
  a working, reasonably fast kernel without anyone hand-writing assembly.

## Edge-AI tradeoffs

| Factor | Vendor toolchain (Modules 01-02) | General compiler stack (TVM/IREE) |
|---|---|---|
| Setup complexity | low (one CLI tool) | higher (build/install the compiler, define a target spec) |
| Portability across chips | none — locked to one vendor | high — same source model, many backends |
| Peak performance on a known target | very good, hand-tuned by the vendor | good to excellent, but needs auto-tuning time to match hand-tuned kernels |
| Handles novel/custom ops | no — falls back to CPU or fails | yes, if you can express it in the IR |
| Build/tune time | fast | auto-tuning search can take hours per operator shape |
| Best fit | single fixed target, ship fast | multi-target fleets, custom ops, research-to-production |

## Exercise

Extend `greedy_fuse` above with a new fusible pair, `("DEPTHWISE_CONV_2D",
"BIAS_ADD")`, and re-run it against a graph that mixes regular and
depthwise convolutions (the pattern a MobileNet-style backbone actually
produces). Count how many total fused groups result versus the raw op
count, and use that ratio as a rough proxy for how much a fusion pass
would reduce memory round-trips for that specific architecture.
