# V11 MTP Vulkan runtime

V11 adds real MTP proposals and batched target verification to the frozen V10
Swift IQ3_XXS runtime. It targets Intel Arc Pro B60 (`8086:e211`) only. The V10
fallback at commit `565b84e`, its executable and its expert residency plan remain
unchanged. CUDA and SYCL do not execute model arithmetic in this runtime.

## Configuration and build

The production configuration is `strata-vulkan-swift-iq3_xxs-mtp.json`: 8192
context, FP16 KV, greedy sampling, one request, up to four MTP proposals and a
per-proposal probability floor of 0.5. An observed-cost controller may shorten
proposal depth. The base expert plan is exactly 10,850 experts / 17.5684 GiB:

`4f9685174ea0e0d3e86cedb9a9f88a7cfb54767bee520257114cace310e8b51e`

```bash
cmake -S vulkan -B build-vulkan -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan --target jr-vk-server-v11 jr-v11-hc-check -j6
export JR_VK_DEVICE=8086:e211
export JR_VK_PYTHON=/data/strata-lab/Strata/.venv/bin/python
build-vulkan/jr-vk-runtime-v11 \
  --config strata-vulkan-swift-iq3_xxs-mtp.json \
  --prompt-file /tmp/v11-production-prompt.txt \
  --max-new 3000 --greedy --checks --report /tmp/v11-run.json
```

The Python environment needs the existing pack tokenizer dependencies, including
`regex` and `jinja2`. Model paths and MTP assets are explicit in the configuration.
The native worker refuses any device selector other than `8086:e211`.

## Proposal, verification and rollback

MTP uses its own attention KV and compact Q8/BF16/Q2 weights. Its token embedding
and full output projection are shared with the main model. The embedding is
imported host memory; the complete output head stays in VRAM. Draft sampling uses
the existing 106,299-ID vocabulary subset without duplicating head weights.
Draft input pairs are the preceding target residual and the next known token;
additional proposals chain through the MTP residual. Committed target pairs
refresh draft KV. The MTP layer attends over its complete configured history.

Verification is layer-major across one through five target positions. Dense and
hyperconnection projections share weight decoding across positions. GDN scans
positions causally in registers. Attention sees only each position's prefix.
Routed expert work is grouped by identical expert within the window, preserving
the frozen residency assignment and native quantized weight values. Router,
SwiGLU and shared/routed output work are batched across positions.
Two-position IQ3_XXS/IQ3_S gate/up verification kernels specialize the existing
column count to two. Real-weight tests measured a 5–8% kernel gain with bit-exact
FP32 outputs; down projections and other formats retain their existing setting.
This changes register usage without changing weights, residency or MTP state.

All target positions execute in one recorded verification command buffer. The
sampler accepts the longest matching draft prefix and returns the target bonus
or rejection correction. Prefix snapshots restore recurrent matrices, convolution
history, PLE history and indexer tails. Position and token-history counters advance
only for committed input rows. KV suffixes are rewound logically and overwritten
before reuse. Greedy sampling has no RNG state to restore.

Four persistent prefix records suffice: the fifth verified row is the final
all-accepted state and cannot be a rejection restore point. Snapshot storage is
450 MiB. Proposal, verification, restore and commit commands, descriptors and
workspaces persist; there are no per-layer CPU waits or hot-path allocations.
Small mapped controls carry IDs and positions. Model state and intermediates
stay on the B60.

## API and fallback

```bash
build-vulkan/jr-vk-server-v11 \
  --config strata-vulkan-swift-iq3_xxs-mtp.json --api-key "$STRATA_API_KEY"
curl http://127.0.0.1:8080/health -H "Authorization: Bearer $STRATA_API_KEY"
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $STRATA_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"swift-iq3_xxs-vulkan","messages":[{"role":"user","content":"Explain memory ordering."}],"max_tokens":3000,"temperature":0}'
```

The API supports `/health`, `/v1/models`, plain text chat completions, optional
SSE, bearer authentication, request reset and SIGINT/SIGTERM shutdown. Concurrent
generation receives HTTP 429. Binding beyond loopback requires an API key. Forced
proposal/rejection debug options are refused by the server.

Non-streaming responses include `strata_stats`; streaming requests report the
same completion metrics in the server log. Metrics include proposed/accepted
counts, depth and window histograms, draft/verification/rollback/catch-up time,
full decode throughput, peak device heap and minimum budget headroom.
Throughput counts decode after the first prediction from prefill and includes
native worker round trips, host PLE I/O, device execution, waiting, sampler
readback and protocol handling. Startup and prefill are separate.
`--profile` adds GPU timing overhead and is for diagnosis.

For MTP off, pass `--spec 0` to V11. For the frozen production fallback, use:

```bash
build-vulkan/jr-vk-server-v10 --config strata-vulkan-swift-iq3_xxs.json \
  --api-key "$STRATA_API_KEY"
```

Stop the current server before loading either fallback on this GPU.

## Acceptance

```bash
"$JR_VK_PYTHON" tools/vulkan_v11_correctness.py \
  --tokens 64 --benchmark 256 --output /tmp/v11-correctness.json
"$JR_VK_PYTHON" tools/vulkan_v11_acceptance.py \
  --output /tmp/v11-acceptance --reference /tmp/v10-final-gate/2-3000.json
"$JR_VK_PYTHON" tools/vulkan_v11_api_check.py \
  --api-key "$STRATA_API_KEY" --output /tmp/v11-api
```

The correctness gate runs real proposals twice, forced acceptance, each rejection
position and 16 target-only continuation tokens after every case. The acceptance
gate runs fresh processes for 256, 1024 and three 3000-token generations. Each run
must complete its requested length, preserve greedy IDs when a frozen reference
is supplied, avoid repeated-token collapse and keep device heap usage bounded
with at least 512 MiB headroom. The separate performance gate remains 28 tok/s. The API gate additionally checks
authentication, input errors, reset, SSE, busy rejection and a 3000-token response.
The reference sequence is used only for comparison, never proposal generation
or performance output. `tools/vulkan_v11_draft_check.py` provides a separate
CPU-only numerical oracle using the pinned GGML dequantizer.

The acceptance tools keep `functional_pass` and `speed_pass` separate. Their
exit status still requires the 28 tok/s speed target. A functional pass below that
target is not a performance PASS. The release decision allows a production V11
release candidate at 25–27 tok/s after exact 3000-token CLI/API, bounded-memory
and recovery checks pass. Further performance changes then belong to V11.1.
Short-window or forced-accept throughput does not satisfy the full-run speed gate.

## Frozen V11 production release candidate

The final focused verification pass is complete. Release status is **production
RC**, with functional gates passing and the **28 tok/s speed target unmet**.
Further performance work belongs to V11.1; V10 `565b84e` remains the fallback.
The following measurements use Intel Arc Pro B60 (`JR_VK_DEVICE=8086:e211`),
the Swift IQ3_XXS model, 8192 context, FP16 KV, greedy sampling, one request,
checks enabled, spec 4 with adaptive depth and per-proposal minimum probability
0.5. Each CLI row starts a fresh process. The prompt has 74 tokens; startup and
prefill are excluded from decode throughput.

| Path | Generated tokens | Full decode tok/s | Functional result |
| --- | ---: | ---: | --- |
| CLI fresh process | 256 | 27.309 | PASS |
| CLI fresh process | 1024 | 26.534 | PASS |
| CLI fresh run 1 | 3000 | 25.610 | PASS |
| CLI fresh run 2 | 3000 | 24.883 | PASS |
| CLI fresh run 3 | 3000 | 24.926 | PASS |
| OpenAI API | 3000 | 25.325 | PASS |

The three 3000-token runs average 25.140 tok/s. All five CLI sequences match
the frozen V10 reference exactly, and all three full sequences match each other.
The API response matches the CLI text exactly. There is no crash, repeated-token
collapse or device heap growth. The focused specialization shows bit-exact
5–8% gains in the measured two-position gate/up kernels; full decode still runs
at about 25 tok/s and does not earn a speed PASS.

Peak device heap is 22.757679 GiB and minimum measured budget
headroom is 0.863415 GiB (884.1 MiB).
Every CLI run records 0 MiB device heap growth. The fixed base expert residency
is 17.5684 GiB. Maximum logged host RSS is 22.8431 GiB; startup peak is
24.9160 GiB. The full output head remains in VRAM.

The final API generation proposes 1930 tokens, accepts 1486
(76.99%), and commits 2999 decode tokens across
1513 verification windows, averaging
1.982 committed tokens/window. Draft time is
2.511 ms/proposed token and verification time is
70.356 ms/window. There are 327 rollbacks
with 532.943 ms total restore time; committed-pair catch-up takes
856.929 ms. Verification remains the dominant measured stage.

Final real-proposal repeat, forced acceptance and rejection positions 0–3 all
match 64 target IDs plus 16 continuation IDs. Independent CPU draft logits
agree on the top token. Hyperconnection and QSA selector diagnostics preserve
reference output bits, including attention selection ties and the 8K boundary.
The explicit V11 `--spec 0` fallback matches 64 frozen V10 IDs with no proposals
and 2.046 GiB headroom.

API authentication, validation, deterministic reset, SSE, one-token metrics,
disconnect recovery, concurrent-request rejection, and 3000-token completion
all pass. SIGTERM shuts down the native worker in
1.154 seconds. The tested configuration can be
restarted with the API command above. Local validation uses an existing private
key file at `/data/strata-lab/tmp/v11-api-key`; no credential is committed.

The frozen V10 executable SHA-256 remains:

`283d8c489c9a45d8740c4bec108cc7a9540c0a1cf3ee303e5e255d15283a1575`

Full local reports are `/tmp/v11-final-correctness.json`,
`/tmp/v11-final-gate/acceptance.json`, `/tmp/v11-final-fallback.json`,
`/tmp/v11-final-api/api-check.json` and `/tmp/v11-final-api/shutdown.json`.
Acceptance commands return a nonzero status because `speed_pass` is false;
`functional_pass` is true. This RC decision does not lower the published speed
target or claim full V11 performance acceptance.
