# JR-Strata-Vulkan V7 — Xe2 Quad-Row / 8-Lane Routed Engine

## Why V7 exists

V6 proved that B60 hardware packed signed INT8 dot is real and accelerated,
but the benefit depends strongly on memory tier:

- all-RAM: essentially no gain from OpSDot
- 5/5 mixed: only a few percent
- all-VRAM: meaningful gain

That means the next bottleneck is weight feeding and per-row execution overhead,
especially for imported host memory.

## V7 kernel shape

V6 used:

```text
1 workgroup = 1 output row
32 lanes per row
```

V7 uses:

```text
1 workgroup = 4 output rows
8 lanes per row
```

Within each 8-lane row group, every lane performs more useful work and reuses
the quant block scale/metadata across its four sub-blocks.

This also mirrors the useful `kExpertLanes=8` production shape in the SYCL
backend.

## Additional memory-feed changes

- explicit unaligned 32-bit fetch: at most two `uint` loads
- IQ2/IQ3 block scale and aux metadata reused across four sub-blocks
- IQ4_NL: 16 packed bytes fetched once per block and reused
- Q2_0: 16 packed bytes fetched once per block and reused across all 8 parts
- four independent 8-lane reductions
- reduction barriers reduced from 5 to 3
- row workgroup count reduced by 4x
- hardware `OpSDot` retained

## Requirements

V7 intentionally targets the B60/Xe2 performance profile and requires:

```text
shaderIntegerDotProduct = YES
integerDotProduct4x8BitPackedSignedAccelerated = YES
```

## Build

Use the local glslang 15.3.0 already installed for V6:

```bash
cd /data/strata-lab/JR-Strata-Vulkan
GLSL="$PWD/.tools/build/glslang-15.3.0/StandAlone/glslangValidator"

cmake -S vulkan -B build-vulkan \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGLSLANG_VALIDATOR="$GLSL"

cmake --build build-vulkan -j"$(nproc)"
```

## Correctness gate

```bash
export STRATA_GGUF_PY=/data/strata-lab/Strata/third_party/llama.cpp/gguf-py

JR_VK_DEVICE=8086:e211 \
python3 tools/vulkan_routed_expert_parity.py \
  --exe ./build-vulkan/jr-vk-routed-engine-v7
```

Then 48 layers:

```bash
JR_VK_DEVICE=8086:e211 \
python3 tools/vulkan_routed_expert_parity.py \
  --exe ./build-vulkan/jr-vk-routed-engine-v7 \
  --all-layers
```

## Direct V6 vs V7 benchmark

```bash
JR_VK_DEVICE=8086:e211 \
python3 tools/vulkan_v7_bench.py
```

The script generates one real route bundle and runs both engines against the
exact same input and expert selection, then prints a final speedup table.
