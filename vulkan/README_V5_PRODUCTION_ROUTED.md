# JR-Strata-Vulkan V5 — Production Routed Engine

V4 proved correctness for top-10 routed MoE across all 48 Swift layers.

V5 changes the execution architecture rather than adding another quant-format
test.

## What V5 changes

### 1. Persistent Vulkan objects

Pipelines, descriptor sets, expert arenas, metadata, and the command buffer are
built once. The warm benchmark does not create a Vulkan pipeline per expert.

### 2. Two-segment expert address table

Selected experts are packed once into:

- segment 0 — imported system RAM
- segment 1 — device-local VRAM

Each route rank has an 8-word metadata record containing tier, formats, role
offsets, and row sizes.

This is the first production form of the expert address table. A future whole
model cache can use more segments without changing the grouped-kernel model.

### 3. Top-k grouped execution

The whole routed group is five dispatches, independent of k:

```text
F32 -> q8_1
grouped gate+up       [all k experts]
grouped SwiGLU->q8_1 [all k experts]
grouped down          [all k experts]
route-weighted combine
```

V4 performed those stages expert by expert.

### 4. Packed Q8 integer-dot path

The activation remains the real 36-byte q8_1 representation. V5 consumes its
packed signed int8 words directly. The shader uses a portable dp4a-shaped
packed-integer primitive so the execution contract is ready for an
`VK_KHR_shader_integer_dot_product` specialization after B60 capability and
performance are measured.

## Build

```bash
cd /data/strata-lab/JR-Strata-Vulkan

cmake -S vulkan -B build-vulkan \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-vulkan -j"$(nproc)"
```

## Correctness gate

Reuse the V4 real-router / real-GGUF oracle, but point it at V5:

```bash
export STRATA_GGUF_PY=/data/strata-lab/Strata/third_party/llama.cpp/gguf-py

JR_VK_DEVICE=8086:e211 \
python3 tools/vulkan_routed_expert_parity.py \
  --exe ./build-vulkan/jr-vk-routed-engine-v5
```

Then:

```bash
JR_VK_DEVICE=8086:e211 \
python3 tools/vulkan_routed_expert_parity.py \
  --exe ./build-vulkan/jr-vk-routed-engine-v5 \
  --all-layers
```

## Warm performance gate

```bash
JR_VK_DEVICE=8086:e211 \
python3 tools/vulkan_v5_bench.py
```

Default benchmark:

- layers 18 and 35
- top-10
- 10 warmup iterations
- 50 timed iterations
- all-RAM / 5-VRAM+5-RAM / all-VRAM

The executable reports Vulkan GPU timestamps, so pipeline creation, descriptor
creation, bundle extraction, and initial expert uploads are outside the timed
region.
