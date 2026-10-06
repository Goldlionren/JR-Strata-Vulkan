# JR-Strata-Vulkan V6 — Hardware Integer-Dot Routed Engine

V5 exposed the next bottleneck:

- all-RAM routed path: ~4.5–4.6 GiB/s
- all-VRAM routed path: ~20.7–21.0 GiB/s
- raw V0.4 device-local read: ~184 GiB/s

The VRAM path is therefore compute/ALU limited rather than memory limited.

V6 targets the packed signed-int8 dot itself.

## Vulkan capability

V6 adds Vulkan 1.3 feature/property discovery:

```text
shaderIntegerDotProduct
integerDotProduct4x8BitPackedSignedAccelerated
```

and enables `shaderIntegerDotProduct` when supported.

The hardware shader uses:

```glsl
#extension GL_EXT_integer_dot_product : require
dotPacked4x8EXT(int(a), int(b))
```

which maps to signed packed `OpSDotKHR`.

## Runtime selection

```text
JR_VK_DOT=auto   default
JR_VK_DOT=soft   V5 software packed-byte multiply-add
JR_VK_DOT=hw     force OpSDot if feature exists
```

`auto` chooses hardware only when the driver explicitly reports the signed
packed-4x8 form as accelerated.

## Build

```bash
cd /data/strata-lab/JR-Strata-Vulkan

cmake -S vulkan -B build-vulkan \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-vulkan -j"$(nproc)"
```

## Correctness

Representative layers:

```bash
export STRATA_GGUF_PY=/data/strata-lab/Strata/third_party/llama.cpp/gguf-py

JR_VK_DEVICE=8086:e211 \
JR_VK_DOT=auto \
python3 tools/vulkan_routed_expert_parity.py \
  --exe ./build-vulkan/jr-vk-routed-engine-v6
```

Strong gate:

```bash
JR_VK_DEVICE=8086:e211 \
JR_VK_DOT=auto \
python3 tools/vulkan_routed_expert_parity.py \
  --exe ./build-vulkan/jr-vk-routed-engine-v6 \
  --all-layers
```

## A/B benchmark

```bash
JR_VK_DEVICE=8086:e211 \
python3 tools/vulkan_v6_bench.py
```

This runs the same real layer/router/expert bundle through `soft` and `auto`
for all-RAM, 5/5 mixed, and all-VRAM paths.

If `packed S8 accel: YES`, the all-VRAM comparison is the important V6 result.
