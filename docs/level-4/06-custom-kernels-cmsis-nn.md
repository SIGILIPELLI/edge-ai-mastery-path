# Custom Kernels with CMSIS-NN

Levels 1-3 treated CMSIS-NN as a library you link against — call
`arm_convolve_s8()`, get a fast int8 convolution. This module goes one
level deeper: when a model has an operation CMSIS-NN doesn't provide
(a custom activation, an unusual pooling variant, unique feature-fusion
math you can't express in stock library ops), you write the kernel
yourself, and the discipline of doing that well is what separates a
correct-but-slow custom op from one that's actually competitive with the
library it sits next to. This is manual-review-only territory — there's
no ARM Cortex-M hardware or CMSIS-NN toolchain in this environment to
compile and run against, so every code sample here is reviewed carefully
for correctness against CMSIS-NN's public API and documented conventions,
and flagged plainly as unverified rather than claimed as tested.

## Why hand-write a kernel at all

CMSIS-NN covers the common cases (conv, depthwise conv, fully-connected,
pooling, standard activations) extremely well — reimplementing those
yourself would almost certainly be slower than the library's hand-tuned
SIMD (`SMLAD`/`SMLAL` instruction sequences on Cortex-M4/M7/M55). The
legitimate reasons to write a custom kernel are narrow: an operation type
CMSIS-NN simply doesn't ship (a novel attention-like fusion block, a
domain-specific signal transform baked into the model graph), or a fusion
opportunity across your specific model's op sequence that no
general-purpose library can know about ahead of time.

## The int8 fixed-point arithmetic CMSIS-NN kernels rely on

Any custom kernel operating on the same quantized tensors as the rest of
the model must implement the same fixed-point requantization convention
CMSIS-NN uses, or values will silently be wrong at the tensor boundary
even though each kernel individually looks correct in isolation.

```c
/* Illustrative C -- manually reviewed against the CMSIS-NN quantization
 * convention (a multiplier + right-shift applied post-accumulation, the
 * standard TFLite-compatible requantization scheme) but NOT compiled or
 * executed; there is no Cortex-M toolchain or hardware in this
 * environment. Treat this as a careful design/code review, not a
 * verified implementation. */

#include <stdint.h>

/* CMSIS-NN's convention: a 32-bit accumulator is requantized to int8
 * output via a fixed-point multiplier and right-shift, computed once
 * ahead-of-time from the layer's known input/weight/output scales
 * (this mirrors arm_nn_requantize's documented behavior). */
static inline int8_t requantize_int32_to_int8(int32_t acc,
                                               int32_t multiplier,
                                               int shift,
                                               int32_t output_offset) {
    /* 64-bit intermediate to avoid overflow during the multiply --
     * a mistake here (using 32-bit) is one of the most common custom
     * kernel bugs, since accumulators can be large before requantization. */
    int64_t scaled = (int64_t)acc * (int64_t)multiplier;
    int32_t rounded = (int32_t)((scaled + (1LL << (shift - 1))) >> shift);
    int32_t result = rounded + output_offset;

    /* Saturate to int8 range -- CMSIS-NN kernels always saturate rather
     * than wrap on overflow, and a custom kernel that silently wraps
     * instead will produce corrupted-looking output only under specific
     * input conditions, making it a hard bug to catch in casual testing. */
    if (result > 127) result = 127;
    if (result < -128) result = -128;
    return (int8_t)result;
}

/* A hypothetical custom activation: a hard-swish-like function not in
 * CMSIS-NN's standard activation set, applied per-element to an int8
 * tensor using the same requantization convention as the rest of the
 * model so it composes correctly with surrounding CMSIS-NN layers. */
void custom_hard_activation_s8(const int8_t *input, int8_t *output,
                                int32_t size, int32_t input_offset,
                                int32_t multiplier, int shift,
                                int32_t output_offset) {
    for (int32_t i = 0; i < size; i++) {
        int32_t x = (int32_t)input[i] - input_offset;   /* dequant to centered int */
        int32_t relu6_like = x < 0 ? 0 : (x > 6 * 256 ? 6 * 256 : x);
        int32_t acc = (x * relu6_like) >> 8;             /* fixed-point multiply */
        output[i] = requantize_int32_to_int8(acc, multiplier, shift, output_offset);
    }
}
```

The saturate-rather-than-wrap behavior is the single detail most likely
to bite a first custom kernel: a signed 8-bit wraparound (`127 + 1` silently
becoming `-128` instead of clamping) produces output that looks like
sporadic random noise in a few pixels/samples rather than an obvious
crash, which makes it exactly the kind of bug that survives casual manual
testing and only shows up against edge-case inputs.

## Memory layout: why loop order determines cache/SRAM behavior more than op count

A microcontroller's SRAM access pattern matters as much as arithmetic
op count, because SRAM bandwidth (not raw ALU throughput) is frequently
the actual bottleneck on Cortex-M-class cores. CMSIS-NN kernels are
written with a specific NHWC (batch, height, width, channel) memory
layout and loop nesting that maximizes sequential memory access; a custom
kernel that doesn't respect this can be arithmetically identical and
still meaningfully slower.

```c
/* Two logically equivalent ways to loop over a NHWC tensor for a
 * per-channel scale-and-shift op -- illustrative of the loop-order
 * decision, not a specific benchmarked result (no hardware available
 * to measure the actual cycle counts here). Manually reviewed for
 * correctness of the two loop orderings against NHWC layout only. */

/* Version A: channel-innermost -- matches NHWC's actual memory layout,
 * so consecutive iterations of the innermost loop touch consecutive
 * bytes in SRAM. This is the loop order CMSIS-NN itself uses. */
void scale_shift_nhwc_fast(const int8_t *in, int8_t *out,
                            int32_t h, int32_t w, int32_t c,
                            const int32_t *per_channel_mult, int shift) {
    for (int32_t y = 0; y < h; y++)
        for (int32_t x = 0; x < w; x++)
            for (int32_t ch = 0; ch < c; ch++) {
                int32_t idx = (y * w + x) * c + ch;
                out[idx] = requantize_int32_to_int8(
                    in[idx], per_channel_mult[ch], shift, 0);
            }
}

/* Version B: channel-outermost -- arithmetically identical result, but
 * every innermost-loop step jumps `c` elements ahead in memory instead
 * of 1, which on a real Cortex-M defeats sequential prefetch and
 * generally runs meaningfully slower for the same op count. */
void scale_shift_nhwc_slow(const int8_t *in, int8_t *out,
                            int32_t h, int32_t w, int32_t c,
                            const int32_t *per_channel_mult, int shift) {
    for (int32_t ch = 0; ch < c; ch++)
        for (int32_t y = 0; y < h; y++)
            for (int32_t x = 0; x < w; x++) {
                int32_t idx = (y * w + x) * c + ch;
                out[idx] = requantize_int32_to_int8(
                    in[idx], per_channel_mult[ch], shift, 0);
            }
}
```

Both functions produce identical output for identical input — this is a
pure loop-reordering difference, not a correctness difference — which is
exactly why it's an easy mistake to miss in code review: nothing about
Version B looks wrong, it's simply slower on real hardware in a way no
unit test running on a desktop CPU (with its much larger cache) would
reliably catch.

## Integrating a custom kernel into a TFLite Micro graph

A custom op has to register itself with TFLite Micro's op resolver so the
interpreter knows to call it for nodes of that type — this is the wiring
that makes a hand-written kernel usable from an otherwise-normal `.tflite`
model exported the usual way.

```c
/* Illustrative registration pattern, manually reviewed against TFLite
 * Micro's documented custom-op API shape -- not compiled or run. */
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

TfLiteStatus CustomHardActivationEval(TfLiteContext* context, TfLiteNode* node) {
    const TfLiteTensor* input = tflite::micro::GetEvalInput(context, node, 0);
    TfLiteTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
    /* Extract per-tensor quantization params from `input`/`output`,
     * compute multiplier/shift once (outside any per-element loop --
     * a common performance bug is recomputing these per element), then
     * call custom_hard_activation_s8 over the full tensor. */
    custom_hard_activation_s8(
        input->data.int8, output->data.int8,
        tflite::NumElements(input),
        input->params.zero_point, /* multiplier, shift computed ahead of time */ 0, 0,
        output->params.zero_point);
    return kTfLiteOk;
}

/* Registered under the same custom-op name the model exporter used when
 * generating the .tflite file's custom-op node. */
static TfLiteRegistration custom_hard_activation_reg = {
    .init = nullptr, .free = nullptr, .prepare = nullptr,
    .invoke = CustomHardActivationEval,
};
```

## Edge-AI tradeoffs

| Factor | Stock CMSIS-NN op | Hand-written custom kernel |
|---|---|---|
| Correctness risk | low, extensively tested by Arm | entirely on you — saturation, requantization, loop order all must be right |
| Performance ceiling | already near-optimal for the target core | matches or loses to CMSIS-NN unless carefully tuned |
| When justified | almost always, if the op exists in the library | only for ops the library genuinely doesn't provide |
| Verification method used here | — | manual code review only; genuinely untested, no hardware available |

## Exercise

Take `scale_shift_nhwc_slow` above and rewrite it with the loop order
fixed (channel-innermost), producing a third version. Then write out, in
prose, the specific unit test you would run on real Cortex-M hardware to
confirm both versions produce bit-identical output across a range of
tensor shapes and quantization parameters — since this module's honest
limitation is that no such test was actually run here, only manually
reasoned through.
