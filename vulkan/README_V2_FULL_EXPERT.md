# JR-Strata-Vulkan V2 — Full Expert Runtime

V2 stops treating individual quant kernels as milestones.

The target is one complete real routed expert from the production Swift
IQ3_XXS model:

```text
real GGUF bytes
    |
    +-- imported system RAM (VK_EXT_external_memory_host)
    |
dequant gate / up / down
    |
gate GEMV + up GEMV
    |
SwiGLU
    |
down GEMV
    |
2560-float expert output
```

All routed-expert quant formats present in this Swift model are handled by the
same V2 runtime:

- 16 IQ2_XXS
- 17 IQ2_XS
- 18 IQ3_XXS
- 20 IQ4_NL
- 21 IQ3_S
- 22 IQ2_S
- 42 Q2_0

The default parity driver finds the unique `gu_type/d_type` combinations in
`native_experts.txt`, selects one real layer for every combination, extracts a
real expert directly from the GGUF shards, computes an independent
llama.cpp/NumPy reference, and executes the complete Vulkan expert path.

This is a correctness-first full-expert implementation. It deliberately
materializes dequantized weights in VRAM and uses F32 GEMV. The production
optimization step is to fuse dequant + dot and introduce q8_1 activations
without changing the externally validated expert function.

## Build

```bash
cd /data/strata-lab/JR-Strata-Vulkan

cmake -S vulkan -B build-vulkan \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-vulkan -j"$(nproc)"
```

## Run all production format combinations

```bash
JR_VK_DEVICE=8086:e211 \
python3 tools/vulkan_full_expert_parity.py
```

## Stronger gate: one real expert from every model layer

```bash
JR_VK_DEVICE=8086:e211 \
python3 tools/vulkan_full_expert_parity.py --all-layers
```

## Gate

V2 passes when every selected real expert completes:

```text
real quantized GGUF bytes
-> Vulkan imported RAM
-> complete gate/up/SwiGLU/down
-> output parity against independent CPU reference
```
