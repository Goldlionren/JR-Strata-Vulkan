# V10 Swift token runtime

`jr-vk-runtime-v10` runs the full Swift IQ3_XXS forward on Vulkan: token embedding,
hyperconnections, GDN, QSA with its indexer, real top-10 routing, V8/V9 experts,
shared experts, PLE, the final hyperconnection norm, output projection and greedy
sampling. The target is Intel Arc Pro B60 (`8086:e211`). CUDA and SYCL are not used.

The first configuration is text, one request, 8192 context, FP16 KV, greedy, and
MTP disabled. Unsupported tensor shapes or formats fail at startup. PLE rows
come from `per_layer_token_embd.weight` in the original GGUF, using a fixed
65536-row host cache. The embedding and all layer arithmetic execute on Vulkan;
PLE hashing and row I/O execute on the host.

## Build and run

Build with a Vulkan 1.3 SDK and glslang supporting `GL_EXT_integer_dot_product`:

```bash
cmake -S vulkan -B build-vulkan -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan --target jr-vk-server-v10 -j6
export JR_VK_DEVICE=8086:e211
export JR_VK_PYTHON=/data/strata-lab/Strata/.venv/bin/python
build-vulkan/jr-vk-runtime-v10 \
  --config strata-vulkan-swift-iq3_xxs.json \
  --prompt "Write a detailed technical explanation of MoE inference." \
  --max-new 3000 --greedy --checks
```

`JR_VK_PYTHON` selects a Python environment with `regex` and `jinja2`. If unset,
`python3` is used. Tokenization and chat formatting reuse the pack's vocabulary,
merges, token types and original chat template. `--raw-prompt` skips chat formatting.
`--prompt-file` reads a UTF-8 text file. The native executable also accepts
`--ids tokens.i32`, containing little-endian int32 IDs.

## Fixed expert residency

The first production configuration uses `data/v10-cache-plan.bin`: 10,850 of
24,576 experts, 17.5684 GiB. This plan is frozen. Its SHA-256 is
`4f9685174ea0e0d3e86cedb9a9f88a7cfb54767bee520257114cace310e8b51e`.
The V8/V9 expert shaders, imported host expert memory and V9 residency format are
unchanged. `--route-profile` can export real routing in STRP format for diagnosis;
it never changes residency while the process runs.

Dense Q6_K, IQ4_XS and IQ3_S projections use a lossless private execution layout:
packed six-bit Q6 symbols or four-bit IQ symbols, with the original FP16
block scales and integer scale multipliers. Expanded byte weights and separate
FP32 scale arrays are not retained. The model artifact and expert layouts stay
unchanged. The CPU check in `tools/vulkan_v10_dense_check.cpp` compares the layout
with the pinned GGML dequantizer using real model blocks. Q6 decoding reuses
aligned packed words, IQ3 sign/magnitude decoding operates on four symbols at
once, and small IQ4 projections use an exact 1 KiB pair lookup.

Startup prints geometry, pack identity, PLE source, residency and B60 heap usage.
Compiled programs and descriptors persist for the process. Two recorded command
buffers share all buffers: ordinary decode and periodic GPU timing. Position,
token ID and PLE inputs are read from mapped control memory on every token.
Independent projections share activation quantization and a dependency barrier.
Hyperconnection projections fuse injection, activations and stream mixing;
residual updates share normalization. Attention and indexer dispatch sizes come
from the token control buffer, removing inactive groups without changing the
recorded commands. Long-context attention groups query heads and retains their
queries in registers; short contexts keep the original split attention path.
State, KV and intermediate buffers stay on
the B60. Routed and shared expert projections reuse one Q8_1 activation buffer.
Ordinary commands overlap independent shared projections with routed expert
work; sampled commands keep their timing buckets separate. Decode uses one queue
submission and one fence wait per token.

Throughput measures the complete model-worker round trip, including host PLE
row I/O, submission, waiting, diagnostics and greedy sampler readback. Prefill,
startup and client network backpressure are separate. GPU bucket timing is
sampled at `--stats-every` intervals (default 128). `--profile` adds per-kernel
timing overhead on the sampled tokens and should be used only for diagnosis.
The sampled GPU buckets describe the serial instrumentation commands; full decode
throughput also includes the ordinary commands that overlap independent work.

## API

```bash
build-vulkan/jr-vk-server-v10 --config strata-vulkan-swift-iq3_xxs.json
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"swift-iq3_xxs-vulkan","messages":[{"role":"user","content":"Explain sparse MoE inference."}],"max_tokens":3000,"temperature":0}'
```

The API supports plain text messages, `n=1`, greedy decoding, and optional SSE
streaming. Concurrent generation requests receive HTTP 429. State resets between
requests. `/v1/models` lists the configured model. Binding beyond loopback requires
`--api-key`; authenticated calls use `Authorization: Bearer <key>`. SIGINT/SIGTERM
close the worker and free its Vulkan resources.

## Diagnostics and acceptance

`--checks` or `JR_VK_CHECKS=1` enables finite checks and min/max/L2 statistics after
major stages. Ordinary tokens latch nonfinite values inside the producing
kernels; full min/max/L2 statistics are sampled at the GPU timing interval.
Explicit `--state-checks` scans the full GDN matrix state on every token. The
output logits have an unconditional fused finite guard. A failure
writes `v10-crash.json` (or places it in `--dump-dir`) with position, token,
previous 16 IDs, layer, stage and statistics. `--logits-out` writes full FP32 logits
for reference comparison. `--state-checks` also scans the complete recurrent
state and is slower than ordinary finite checks. Device errors produce crash
snapshots; nonfinite statistics are written as JSON null.

```bash
JR_VK_DEVICE=8086:e211 JR_VK_PYTHON=/data/strata-lab/Strata/.venv/bin/python \
  python3 tools/vulkan_v10_acceptance.py \
  --config strata-vulkan-swift-iq3_xxs.json --output /tmp/v10-acceptance
```

The gate runs 256, 1024, and three fresh-process 3000-token generations. It checks
completion count, repeated-token collapse, exit status and the 20 tok/s minimum.
One-token reference parity, memory bounds, A770M isolation and a real API response
are additional requirements before declaring V10 PASS. The CPU-only oracle helper
in `tools/vulkan_v10_oracle.cpp` links against the pinned llama.cpp CPU library.

## Memory and MTP status

On this B60 (24 GB, PCIe 4.0 x4), the compact-layout runtime held device heap usage
at 21.5123 GiB and its reported budget at 23.5586 GiB across a 3000-token run:
2.0463 GiB of headroom. Host RSS stayed at about 22.84 GiB, with a 24.92 GiB load
peak. The fixed scratch workspace is 3,288,064 bytes and is shared by all layers.
Layer KV and recurrent states have their own persistent allocations because each
layer must retain its state.

The 273,152,000-byte token embedding uses imported host memory. Its GPU bucket
averaged 0.023 ms per token in that run. The complete Q5_K output projection stays
in device-local memory. No per-token lm-head transfer was introduced.

Existing MTP artifacts under `/data/strata-lab/data/mtp/rt` contain 675 MiB of
expert weights, about 110.72 MiB of dense weights, and 106,299 draft-vocabulary
IDs. They are not loaded by V10. A future implementation should share the main
embedding and output weights. Copying the draft-vocabulary subset of the main
head alone would cost about 178.42 MiB. An initial snapshot of recurrent, conv,
PLE-history and indexer-tail state costs about 112.59 MiB; extra proposal-prefix
snapshots would add to that. Rejected KV cells can be overwritten after rewinding
the logical position rather than copying the complete KV allocation.

Memory headroom has reached the requested 1.5–2 GiB range, but MTP remains disabled
until base production acceptance passes. This is base-runtime headroom; future
MTP weights and rollback buffers must also fit the budget. Full Vulkan proposal, batched verification
and exact state rollback are not implemented; merely loading MTP weights or doing
sequential verification would not establish an acceleration benefit.

## Measured acceptance

Measured on 2026-10-07: Intel Arc Pro B60 24 GB on PCIe 4.0 x4, i7-12700H,
64 GB RAM, Ubuntu 24.04. The configuration is 8192 context, FP16 KV, greedy,
MTP disabled, and the frozen 17.5684 GiB plan above. Major-stage finite guards
were enabled. The performance gate covers every model-worker decode round trip
across the full generation, excluding startup and prefill.

| Test | Result |
| --- | --- |
| CPU-only oracle, raw token 9707 | Same top-1 token 16451; logits cosine 0.991553 |
| Lossless dense layouts against pinned GGML | 193 tensors, 3088 sampled real blocks, exact |
| Exact dense decoder contraction checks | Selected real matrices, zero difference from the preceding Vulkan decoder |
| Fused finite-guard injection | Finite input stays clear; NaN and Inf latch failure |
| 256-token deterministic decode | Passed; same IDs before and after dense decoder optimization |
| 1024-token full decode | 22.138 tok/s |
| 3000 tokens, fresh process 1 | 20.012 tok/s |
| 3000 tokens, fresh process 2 | 20.016 tok/s |
| 3000 tokens, fresh process 3 | 20.021 tok/s |
| Fresh-process reproducibility | All three 3000-token ID sequences identical |
| Full GDN state checks | One full forward passed; top-1 token 16451 |

All three long runs generated 3000 tokens, finished by the length limit, and had
893 distinct IDs (209 in the final 512). Device usage stayed at 21.5123 GiB,
headroom at 2.0463 GiB, host RSS at about 22.845 GiB, and load peak at 24.918 GiB.
No nonfinite failure, engine crash or repeated-token collapse occurred.

The long prompt asks for a 6000-word technical chapter about computer memory and
concurrent programming, covering caches, memory ordering, virtual memory,
asynchronous I/O, memory pools, synchronization and failure handling. The frozen
plan was calibrated on a prior 1024-token trace of this prompt; these measurements
are not a held-out cache evaluation. No cache changes were made for the long runs.

The CPU oracle uses the CPU-only build of the pinned llama.cpp source at commit
`3cf03257f`; it never initializes a GPU backend. Runtime-owned DRM counters showed
zero A770M allocations and zero A770M engine time. The selected logical device
was always `8086:e211`.

The final OpenAI API returned a valid 3000-token completion at 20.044 tok/s
model-worker decode (153.32 s including prefill and the HTTP request). Its text
matched the accepted CLI output exactly. Authentication, malformed scalar and
message rejection, deterministic reset, SSE completion, and concurrent-request
HTTP 429 checks passed. SIGTERM cleanly exited both server and worker and released
port 8080.

**V10 PASS** for this production configuration and measured prompt. Performance
tuning stopped after the full 3000-token speed gate passed. MTP stays disabled.
The recommended final branch is `v10-production-runtime`.
