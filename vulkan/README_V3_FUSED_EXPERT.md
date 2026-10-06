# JR-Strata-Vulkan V3 — Fused Quantized Expert Path

V2 validated the complete expert function on all 48 production layers, but it
materialized each quantized matrix as F32 before GEMV.

V3 removes that architectural dead end.

```text
quantized expert in imported RAM
        |
        +--> fused quant decode + F32 dot --> gate
        |
        +--> fused quant decode + F32 dot --> up
                                      |
                                   SwiGLU
                                      |
quantized down in imported RAM
        |
        +--> fused quant decode + F32 dot --> expert output
```

There is no full F32 weight matrix allocation.

This is already the intended VRAM/RAM expert storage architecture. The next
optimization after parity is to replace F32 activations/dots with Strata's
q8_1 + integer-dot path and then group multiple routed entries.

## Build

```bash
cmake -S vulkan -B build-vulkan -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan -j"$(nproc)"
```

## Validate all 10 production format combinations

The existing V2 parity driver accepts an alternate executable:

```bash
export STRATA_GGUF_PY=/data/strata-lab/Strata/third_party/llama.cpp/gguf-py

JR_VK_DEVICE=8086:e211 \
python3 tools/vulkan_full_expert_parity.py \
  --exe ./build-vulkan/jr-vk-fused-expert
```

## Strong gate: all 48 layers

```bash
JR_VK_DEVICE=8086:e211 \
python3 tools/vulkan_full_expert_parity.py \
  --exe ./build-vulkan/jr-vk-fused-expert \
  --all-layers
```

The per-process runner also prints the complete single-expert execution time.
Do not compare that number directly with production token latency yet: the
parity harness starts a new process and creates pipelines for every case.
