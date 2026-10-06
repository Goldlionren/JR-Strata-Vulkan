# JR-Strata-Vulkan V8 — Format-Specialized Xe2 Routed Engine

V7 is a major performance milestone:

- 48/48 real layers pass
- all-VRAM routed expert throughput reaches roughly 55–65 GiB/s
- mixed 5/5 improves roughly 14%
- imported-RAM path improves modestly to roughly 4.9 GiB/s

## Why V8

A production Swift layer has one fixed gate/up quant format and one fixed down
quant format for all 512 experts in that layer.

V7 nevertheless ships all seven production decoders in one monolithic shader:

```text
IQ2_XXS / IQ2_XS / IQ3_XXS / IQ4_NL / IQ3_S / IQ2_S / Q2_0
```

and chooses the format at runtime.

That keeps unnecessary branches and decoder state live in the hot shader and
can increase register pressure.

V8 compiles one shader per real production quant type and chooses only two
pipelines per layer:

```text
GU specialized shader
DOWN specialized shader
```

The V7 execution shape is retained:

```text
4 rows/workgroup
8 lanes/row
packed Q8_1
accelerated OpSDot
quant-block reuse
```

## Build

```bash
cd /data/strata-lab/JR-Strata-Vulkan

GLSL="$PWD/.tools/build/glslang-15.3.0/StandAlone/glslangValidator"

cmake -S vulkan -B build-vulkan \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGLSLANG_VALIDATOR="$GLSL"

cmake --build build-vulkan -j"$(nproc)"
```

## Correctness

```bash
export STRATA_GGUF_PY=/data/strata-lab/Strata/third_party/llama.cpp/gguf-py

JR_VK_DEVICE=8086:e211 \
python3 tools/vulkan_routed_expert_parity.py \
  --exe ./build-vulkan/jr-vk-routed-engine-v8
```

Strong gate:

```bash
JR_VK_DEVICE=8086:e211 \
python3 tools/vulkan_routed_expert_parity.py \
  --exe ./build-vulkan/jr-vk-routed-engine-v8 \
  --all-layers
```

## V7 vs V8

```bash
JR_VK_DEVICE=8086:e211 \
python3 tools/vulkan_v8_bench.py
```

Defaults include 9-VRAM / 1-RAM because the next production question is no
longer only the artificial 5/5 split: once VRAM kernels are fast, even a single
RAM miss among top-10 experts can dominate one layer's MoE latency.
