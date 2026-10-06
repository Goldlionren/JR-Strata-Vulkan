# JR-Strata-Vulkan V0

This phase is intentionally independent from the Strata model runtime.

## Goal

1. Create a Vulkan 1.3 instance.
2. Enumerate all physical devices.
3. Select Intel Arc Pro B60 by vendor/device ID `8086:e211`.
4. Create a compute queue and logical device.
5. Check the initial JR Intel Arc performance-profile capabilities.
6. Compile and execute a compute shader.
7. Compare GPU output against a CPU reference.

## Build

```bash
cmake -S vulkan -B build-vulkan -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan -j"$(nproc)"
```

## Run

Default target is the B60:

```bash
./build-vulkan/jr-vk-v0
```

Explicit target:

```bash
JR_VK_DEVICE=8086:e211 ./build-vulkan/jr-vk-v0
```

Expected ending:

```text
JR-VK V0 vector-add
  elements     : 4096
  max abs error: 0
  mismatches   : 0
JR-VK V0: PASS
```

## Next

V0.2:
- device-local buffer copy bandwidth
- host-visible buffer bandwidth

V0.3:
- `VK_EXT_external_memory_host`
- aligned host allocation
- `vkGetMemoryHostPointerPropertiesEXT`
- imported host memory
- GPU direct read from system RAM
- PCIe bandwidth measurement
