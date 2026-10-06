# JR-Strata-Vulkan V4 — Routed Expert Engine

V4 moves from a single expert to the real routed-expert execution boundary.

The interface matches the boundary of Strata's `native_expert_grouped`: routing
decisions are already known (`expert_id`, normalized route weight), and the
expert engine executes the selected experts.

The parity driver obtains those decisions from the **real Swift layer router**
as an independent CPU oracle.

## V4 path

```text
real layer hidden state
        |
        +--> real router weights --> top-10 ids + normalized route weights
        |
        +--> q8_1 activation
                 |
     +-----------+---------------------+
     |           |                     |
 expert rank 0  ...                expert rank 9
 VRAM or imported RAM              VRAM or imported RAM
     |                                  |
 native quant weight × q8_1             |
 gate/up                                |
     |                                  |
 SwiGLU + q8_1                          |
     |                                  |
 native quant down × q8_1               |
     +----------------+-----------------+
                      |
             route-weighted combine
                      |
                 routed MoE output
```

Default tier split is top-5 selected experts in VRAM and ranks 5-9 in imported
system RAM. It can be changed with `--vram-count`.

V4 uses the production q8_1 activation representation (36 bytes per 32
values). The dot kernel consumes q8_1 directly and native quantized weights
directly. For this correctness milestone the q8 int8 values are widened to
float inside the dot; packed integer-dot optimization is intentionally a later
performance specialization and does not change the V4 routed-engine contract.

## Build

```bash
cd /data/strata-lab/JR-Strata-Vulkan

cmake -S vulkan -B build-vulkan \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-vulkan -j"$(nproc)"
```

## Representative routed gate

```bash
export STRATA_GGUF_PY=/data/strata-lab/Strata/third_party/llama.cpp/gguf-py

JR_VK_DEVICE=8086:e211 \
python3 tools/vulkan_routed_expert_parity.py
```

Default layers:

```text
0, 1, 8, 28, 35, 36
```

They span early/late layers, Q2_0/IQ4_NL down paths, and representative GU
formats.

## Strong gate — all 48 layers

```bash
JR_VK_DEVICE=8086:e211 \
python3 tools/vulkan_routed_expert_parity.py --all-layers
```

## Tier tests

All imported RAM:

```bash
python3 tools/vulkan_routed_expert_parity.py --vram-count 0
```

All VRAM:

```bash
python3 tools/vulkan_routed_expert_parity.py --vram-count 10
```

Mixed production-shaped path:

```bash
python3 tools/vulkan_routed_expert_parity.py --vram-count 5
```

The next stage after V4 correctness is persistent pipelines/descriptors,
expert-address tables, grouped dispatch, and packed integer dot performance.
