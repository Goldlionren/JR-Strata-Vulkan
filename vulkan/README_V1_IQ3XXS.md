# JR-Strata-Vulkan V1.0 — IQ3_XXS dequant parity

This is the first Vulkan kernel that understands the actual quantization format
used by the validated Swift IQ3_XXS model.

## Contract

- GGML type: 18 (`IQ3_XXS`)
- block size: 98 bytes
- values per block: 256
- workgroup size: 32
- one workgroup decodes one IQ3_XXS block
- one lane writes eight FP32 values

The implementation is based on the same GGML block definition, lookup tables,
and decode mapping used by Strata's CUDA/SYCL path.

## Test paths

The same shader is checked twice:

1. IQ3_XXS source bytes in imported system RAM
2. identical source bytes in device-local VRAM

Both are compared against an independent CPU implementation built from the
official `ggml-common.h` tables and Strata's exact fp16-bit converter.

## Build

```bash
cmake -S vulkan -B build-vulkan -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan -j"$(nproc)"
```

## Run

```bash
JR_VK_DEVICE=8086:e211 \
JR_VK_IQ3_BLOCKS=4096 \
./build-vulkan/jr-vk-iq3xxs-parity
```

Expected:

```text
imported-host: ... tolerance failures 0 ...
device-local: ... tolerance failures 0 ...

JR-VK V1.0 IQ3_XXS: PASS
```

The test also reports bitwise differences. Numerical correctness is the V1.0
contract; bitwise identity is informative and may depend on floating-point
lowering.

## Next

After V1.0 parity:

- extract real IQ3_XXS blocks from the production GGUF and repeat parity
- q8_1 activation quantization
- IQ3_XXS × Q8_1 dot product
- IQ3_XXS MMVQ/GEMV
