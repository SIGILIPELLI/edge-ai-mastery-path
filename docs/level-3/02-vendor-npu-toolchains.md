# Vendor NPU Toolchains (i.MX, Ethos-U)

Module 01 used Coral's Edge TPU as a clean, single-vendor example: one
compiler, one supported-ops list, one runtime. Real embedded projects
rarely get that luxury — NXP's i.MX8/9 family ships its own NPU with the
**eIQ** software stack, and Arm's **Ethos-U** NPU (found inside dozens of
Cortex-M-class SoCs from STMicro, Renesas, and others) has a completely
different compiler, the **Vela** graph optimizer. Same TFLite input file,
two unrelated toolchains, two different sets of rules about what runs on
the accelerator versus what falls back to the CPU. This module walks both,
because "which NPU toolchain" is usually decided by which chip you're
already committed to, not the other way around.

## Two philosophies, one input format

Both vendors accept the same starting point — a quantized `.tflite` file
— which is the one piece of good news: your Level 2 quantization workflow
doesn't change. What differs is everything downstream.

| | NXP eIQ (i.MX) | Arm Ethos-U (Vela) |
|---|---|---|
| Target class | Linux-capable apps processors (i.MX8M Plus, i.MX93) | Cortex-M microcontrollers with an Ethos-U55/U65 co-processor |
| Compiler | eIQ Neutron compiler (part of eIQ Toolkit) | Vela (Python, open source) |
| Runtime | TFLite delegate (`libneutron_delegate.so`) | TFLite Micro with the Ethos-U driver as a custom op |
| Output artifact | delegate-annotated `.tflite`, loaded at runtime | a `.tflite` with an embedded pre-compiled command stream |
| OS assumption | full Linux, dynamic loading | bare-metal or RTOS, static image |
| Typical power envelope | 1-5 W | 10-100 mW |

The Ethos-U row is the more unusual design: Vela doesn't produce a
separate binary you load — it rewrites the `.tflite` file itself, replacing
every NPU-eligible subgraph with a single custom op containing a
pre-compiled command stream for the Ethos-U hardware. TFLite Micro's
runtime then just executes that custom op like any other. There is no
"delegate" concept at runtime at all; the compilation decision is frozen
into the model file ahead of time.

## Walking through eIQ (i.MX, delegate model)

eIQ follows the same delegate pattern later modules will see with ONNX
Runtime's execution providers: the model file is unmodified TFLite, and a
runtime flag tells the interpreter to hand eligible ops to the Neutron
delegate instead of executing them in the default CPU kernels.

```c
/* Illustrative C -- this is the shape of an eIQ / TFLite-Micro delegate
 * setup, not a runnable program (it depends on the NXP BSP, MCUXpresso
 * SDK, and physical i.MX hardware this environment cannot provide).
 * Manually reviewed for API-shape correctness against NXP's published
 * eIQ examples; not executed. */

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "neutron_delegate.h"

TfLiteStatus setup_interpreter_with_npu(const uint8_t* model_data,
                                         tflite::MicroInterpreter** out_interp) {
    const tflite::Model* model = tflite::GetModel(model_data);

    /* Neutron delegate created with a device handle; NULL falls back
     * to whatever CPU reference kernels TFLite Micro ships with. */
    TfLiteDelegate* npu_delegate = NeutronDelegate_Create(/*device=*/NULL);
    if (npu_delegate == NULL) {
        return kTfLiteError;  /* NPU not present or driver not loaded */
    }

    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();

    static tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);

    /* Ops the delegate can't claim silently run through `resolver`'s
     * CPU kernels instead -- exactly the fallback behavior from
     * Module 01, just with a different compiler drawing the line. */
    interpreter.ModifyGraphWithDelegate(npu_delegate);

    *out_interp = &interpreter;
    return kTfLiteOk;
}
```

The delegate model means you can develop and test on CPU-only hardware
(delegate creation just returns NULL) and the exact same binary runs
faster the day it lands on real i.MX silicon with the NPU driver present
— a meaningfully different deployment story than Ethos-U's ahead-of-time
compilation, where the target chip must be known at compile time.

## Walking through Vela (Ethos-U, ahead-of-time model)

Vela is a Python tool you run once, offline, against your quantized
`.tflite` file. It needs a config describing the specific Ethos-U variant
(U55 vs U65), its SRAM size, and clock speed, because the command stream
it emits is tuned to that exact configuration.

```python
"""
Vela's real CLI is `vela model.tflite --accelerator-config ethos-u55-256
--config vela_cfg.ini`. This function models Vela's op-partitioning
decision -- which ops it can fold into the NPU command stream versus
which stay as separate TFLite-Micro ops -- based on Arm's published
constraints. Manual review only; Vela itself isn't installed here.
"""

ETHOS_U55_SUPPORTED = {
    "CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED",
    "MAX_POOL_2D", "AVERAGE_POOL_2D", "ADD", "MUL", "CONCATENATION",
}
# Vela also rejects ops whose per-tensor shapes exceed on-chip SRAM,
# even if the op type is otherwise supported.

def vela_partition(op_list, sram_bytes, sram_limit=256 * 1024):
    npu_ops, cpu_ops, running_bytes = [], [], 0
    for name, op_type, tensor_bytes in op_list:
        fits_sram = running_bytes + tensor_bytes <= sram_limit
        if op_type in ETHOS_U55_SUPPORTED and fits_sram:
            npu_ops.append(name)
            running_bytes += tensor_bytes
        else:
            cpu_ops.append(name)
    return {"npu_ops": npu_ops, "cpu_ops": cpu_ops,
            "sram_used_kb": running_bytes / 1024}

graph = [
    ("conv1", "CONV_2D", 32 * 1024),
    ("dw1", "DEPTHWISE_CONV_2D", 64 * 1024),
    ("custom_attn", "CUSTOM", 8 * 1024),   # unsupported op type
    ("conv2", "CONV_2D", 180 * 1024),      # would overflow SRAM budget
]
print(vela_partition(graph, sram_bytes=256 * 1024))
# npu_ops: ['conv1', 'dw1']; cpu_ops: ['custom_attn', 'conv2']
# conv2 is a *supported op type* rejected purely on SRAM fit --
# a failure mode that doesn't exist on Coral's larger, off-chip memory.
```

That last line is the detail that trips up people moving from Coral to
Ethos-U: on Coral, "is this op supported" is the only question. On
Ethos-U, an op can be fully supported and still get bounced to the CPU
because the tiny on-chip SRAM (commonly 128-512 KB, versus a full DRAM
budget on i.MX or a host-attached Coral) can't hold its working set.
Reducing tile sizes or restructuring convolutions to use fewer channels
per layer is a real Ethos-U optimization technique with no Coral
equivalent.

## Choosing between them isn't really a choice you make in software

In practice this decision is made by procurement and BOM cost long before
firmware is written: i.MX8/9 parts run full Linux and cost more per unit;
Ethos-U-equipped Cortex-M parts are cents-to-low-dollars and run bare
metal or an RTOS. The toolchain follows the silicon. What transfers
between them is the underlying discipline from Module 01 — read the
compiler's op-mapping report, don't trust TOPS numbers in isolation, and
budget for on-chip memory limits as a first-class constraint, not an
afterthought.

## Edge-AI tradeoffs

| Factor | eIQ / i.MX (delegate, AoT-optional) | Vela / Ethos-U (ahead-of-time only) |
|---|---|---|
| Compile-time target binding | optional (delegate resolves at runtime) | mandatory (baked into the `.tflite`) |
| Memory model | DRAM, generous | on-chip SRAM, tightly bounded |
| OS requirement | Linux | bare-metal / RTOS |
| Dev-without-hardware | yes (CPU fallback path) | limited (Vela needs the target config, but no HW needed to *compile*) |
| Typical use case | camera + vision gateway, industrial HMI | always-on sensor node, battery-powered wearable |

## Exercise

Pick a quantized model from an earlier level. Using the `vela_partition`
function above as a template, write down its op list with approximate
per-layer activation sizes, and manually partition it against a
128 KB SRAM budget (half of the U55-256 example). Identify which layers
would need reshaping (fewer channels, smaller tiles) purely to fit
on-chip memory, independent of whether the op type itself is supported.
