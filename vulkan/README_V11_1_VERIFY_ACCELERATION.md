# V11.1 target verification acceleration

V11.1 changes target verification on Intel Arc Pro B60 (`8086:e211`). The
production RC at `c943d61` / `jr-vulkan-v11-production-rc-20261007` and the V10
fallback at `565b84e` remain frozen. Use a separate build directory to preserve
their executables and shaders. The V11.1 freeze is tagged
`jr-vulkan-v11.1-production-20261007`.

The configuration remains `strata-vulkan-swift-iq3_xxs-mtp.json`: 8192 context,
FP16 KV, greedy, one request, four maximum proposals, probability floor 0.5,
checks enabled. The existing controller selects proposal depth. Base expert
residency stays at 10,850 experts / 17.5684 GiB. Embedding import, device-local
head, MTP weights, buffers, snapshots, API and committed state rules are unchanged.

## Verification changes

Dense projections use 128-thread workgroups with sixteen output rows. Each
output retains its original eight-lane accumulation and reduction. Position
specializations remove redundant dynamic row checks in dense and hyperconnection
projections. Batched dense execution uses 16-lane subgroups, with 32 lanes for the output head. Grouped gate/up
verification uses measured 16/32-lane choices and position specializations;
grouped down execution retains 32 lanes. Gate/up uses wider workgroups only where measured useful: 64 threads for three-position windows and two-position IQ2_XS, and 128 threads for two-position IQ3 windows. Each output keeps its original eight lanes; group dispatch counts follow the selected width. Other windows retain 32 threads. Native Q2 byte packing and aligned
FP16 scale reads preserve weight values.

HC down projections specialize their fixed BF16 column width while retaining
all 256 original virtual lanes and their sums. HC up projections compute four
independent coordinates per subgroup and pack the exact Q8 block in the same
dispatch. Both the original mixed vector and rounded input buffers are written.

GDN convolution and Q/K normalization share one dispatch, retaining the original causal history, four-term convolution and padded reduction tree. GDN output normalization, gating and Q8 input packing share one dispatch while retaining both original buffers. QSA query splitting, normalization and rotary
preparation are fused; key preparation also writes the original FP16 KV cells.
The original reduction trees and attention selection remain unchanged. Late
windows use the existing KV tile kernel; early windows retain the RC attention
kernel, including any window crossing position 512. Late KV tiles retain packed FP16 words in shared memory, halving shared-memory use without changing values or arithmetic. Two indirect records in
existing reserved controls select one producer, followed by the original combine.
Independent embedding rows share a completion barrier. Routing and Q8 packing
share a completion barrier before grouped expert execution. QSA key/query preparation also shares a completion barrier.

Indexer scoring batches positions and evaluates four query heads together. Each subgroup calculates the same four independent 32-coordinate partials and the original zero-padded final reduction. Head scores are added in their original order. Later pooled-key appends only change blocks at or above an earlier position’s block count; earlier scoring reads completed blocks below that count and the constant spare key for its tail. Causal pool updates and snapshots retain their original order. Scores can therefore run after the window’s pool updates. A dispatch in existing reserved control words skips scoring below the selection limit.

Low-confidence windows with one target position reuse the recorded verifier.
They omit snapshots because no rejected suffix exists. A one-token window that
executes no draft input retains the RC paired submission, including preparation
of the preceding MTP KV pair. Multi-position snapshots, rejection restore,
committed residuals and logical KV progression follow the RC path.

## Native host expert reads

Profiling measured grouped gate/up and down execution as the largest remaining
verification cost. Raw native-byte prefetch passed real-weight bit checks and
reduced isolated copy-plus-projection time. It uses the existing snapshot record
3 for windows with at most four positions. Such a window can only restore a
prefix before its last position, so record 3 is never live. Five-position windows
keep all four live records and use the ordinary native host path. A bounds check
ensures forty complete native expert blobs fit the unused record.

The GPU group pass preserves native group order, metadata, residency flags and
position slots. It also generates a host-expert list and copy dispatch in existing
spare words. Staging metadata uses the HC normalization workspace after its final
consumer. Coalesced uint4 reads copy original native bytes into the existing
snapshot buffer. The unchanged expert decoder reads the copied bytes. The next
layer overwrites the dead workspace. No model buffers or persistent expert cache
entries are added. Snapshots omit the last position because full acceptance uses
the current state directly; all possible rejection prefixes remain intact.

## Production validation

Measured on Intel Arc Pro B60 (`8086:e211`), Swift IQ3_XXS, 8192 context allocation, FP16 KV, greedy, single request, spec 4 / probability floor 0.5, checks enabled. Each run starts a fresh native process. Throughput covers all 3000 generated tokens, excluding separately reported prompt prefill. No profiling instrumentation is enabled in these acceptance runs.

| Fresh run | Full 3000-token tok/s | Verify ms/window | Minimum headroom GiB |
| --- | ---: | ---: | ---: |
| 1 | 28.815 | 59.315 | 0.86146 |
| 2 | 28.893 | 59.736 | 0.86146 |
| 3 | 28.777 | 59.043 | 0.86146 |

All three runs pass exact comparison of every token ID with `c943d61`, deterministic reproduction, finite execution and bounded VRAM. The longest identical-token run is two. Allocated VRAM stays at 22.7577 GiB with zero growth; minimum reported headroom is 0.86146 GiB (882 MiB). The calibrated 10,850-expert / 17.5684 GiB plan is unchanged.

The RC production measurement supplied for this milestone was about 25.14 tok/s and 70.356 ms per verification window. The unchanged adaptive controller selects window depths from measured cost, so verification averages include its selected window mix. A warm development run of the frozen candidate measured 30.482 tok/s; the fresh production measurements above are the acceptance results.

All 17 accept/reject/continuation cases pass, including low-confidence single-position windows and staging reuse followed by fourth-prefix rejection restore. Direct real-weight checks preserve every tested FP32/Q8/KV bit. Recurrent preparation checks compare final state, every live prefix and canonical nonfinite flags. Indexer checks compare serial RC pooling/scoring with batched scoring at positions 0, 73, 2048 through 2053, 3071 and 8187, across widths one through five, including repeated updates. Native expert checks cover all five gate/up formats, both down formats, RAM/resident mixtures and shared/disjoint routing; wider workgroup checks use the actual compiled production pipelines. Earlier 1024-token final-head capture matched all 248,320 RC FP32 logits.

The authenticated OpenAI API 3000-token request passes at **29.053 tok/s**, with 59.170 ms per verification window and exact RC completion text. Authentication, input validation, deterministic reset, SSE output, one-token metrics, busy-request rejection and recovery after stream disconnect all pass. API VRAM growth is zero. The server remains available at `http://127.0.0.1:8080` with the existing private API key.

All acceptance gates pass. Performance tuning stopped after reaching the threshold. No API code, draft/state controller, model memory allocation sizes, expert residency or V10/V11 RC tag was changed. Every GPU test and the production server selected `JR_VK_DEVICE=8086:e211`; the A770M received no model allocations or execution from this work.

Machine-local evidence is in `/tmp/v11.1-final-gate/acceptance.json`, `/tmp/v11.1-final-api/api-check.json`, and `/tmp/v11.1-wide-gu-correctness.json`. Logs, captures, models, build artifacts and credentials are not part of the release commit.

## Build and gates

```bash
cmake -S vulkan -B build-vulkan-v11.1 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGLSLANG_VALIDATOR="$PWD/.tools/build/glslang-15.3.0/StandAlone/glslangValidator"
cmake --build build-vulkan-v11.1 -j8
export JR_VK_DEVICE=8086:e211
export JR_VK_PYTHON=/data/strata-lab/Strata/.venv/bin/python
"$JR_VK_PYTHON" tools/vulkan_v11_correctness.py \
  --runtime build-vulkan-v11.1/jr-vk-runtime-v11 \
  --reference /tmp/v11-final-gate/3-3000.json \
  --output /tmp/v11.1-correctness.json
"$JR_VK_PYTHON" tools/vulkan_v11_acceptance.py \
  --runtime build-vulkan-v11.1/jr-vk-runtime-v11 \
  --reference /tmp/v11-final-gate/3-3000.json \
  --output /tmp/v11.1-final-gate --lengths 256,1024,3000,3000,3000
```

The reference paths point to this machine's frozen RC artifacts. Direct kernel
diagnostics compare with the preserved `build-vulkan/shaders` RC shaders.
Keep that directory intact. Every GPU command must select `8086:e211`.
The A770M must receive no model allocations or runtime work.

The API launcher is `build-vulkan-v11.1/jr-vk-server-v11`; it selects its sibling
runtime and the existing frontend. Bind to `127.0.0.1:8080` and provide the
existing private API key through `STRATA_API_KEY`. See the frozen
[API instructions](README_V11_RUNTIME.md) for the unchanged request contract.
