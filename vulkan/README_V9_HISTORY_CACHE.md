# V9 History-Aware Cache Manager

The single-trace experiment showed why a static "train on one conversation"
profile is not enough:

- shipped profile on a new trace: ~49%
- profile trained on an older trace, tested on the new trace: ~76%
- profile trained and tested on the same trace: ~95%

So the correct production policy is not "retrain on the last prompt".  It is a
rolling workload profile built from multiple historical traces.

## Policy

For each archived trace, V9 computes per-layer expert selection probabilities.

Trace contribution is then:

```text
selection_probability
× exponential_recency_decay
× short-trace_confidence
```

Default half-life: 14 days.

A 512-position trace gets full confidence; shorter traces are down-weighted by
the square root of their relative length.

This has two important properties:

1. One giant session does not drown every other user's/session's workload.
2. Old behavior fades instead of remaining in the cache forever.

The shipped `expert-profile.bin` is retained only as tie-break/fallback for
pairs that history has not established.

## Important correction: do NOT rank by frequency / bytes

Imported-RAM expert cost is approximately proportional to expert blob size.
Caching a 2 MiB expert costs more VRAM than a 1.7 MiB expert, but also avoids a
proportionally larger PCIe miss.

So the primary residency value is routing probability/heat. Blob size belongs
in the capacity constraint, not as another divisor in the heat score.

## Files

```text
tools/vulkan_v9_history_update.py
    Scans history, applies decay/normalization, writes both profile and plan.

tools/vulkan_v9_prestart_update.sh
    Archives the previous current trace and calls the updater.
```

Stable outputs:

```text
/data/strata-lab/Strata/data/expert-profile-v9-history.bin
/data/strata-lab/JR-Strata-Vulkan/data/v9-cache-plan-history.bin
/data/strata-lab/JR-Strata-Vulkan/data/v9-history-report.json
```

## First run with the two traces already collected

The updater also discovers:

```text
/data/strata-lab/Strata/data/v9-routing-*.bin
```

so the existing train/test files do not need to be moved first.

Run:

```bash
cd /data/strata-lab/JR-Strata-Vulkan

python3 tools/vulkan_v9_history_update.py
```

With at least two traces it automatically performs a leave-latest-trace-out
cross-validation.  That number is much more useful than the final plan's
in-sample scores.

## Normal startup workflow

Before starting Strata:

```bash
cd /data/strata-lab/JR-Strata-Vulkan
bash tools/vulkan_v9_prestart_update.sh
```

Then start Strata normally.

Keep the routing output configured as:

```text
--dump-routing
/work/Strata/data/v9-production-routing.bin
```

If you also want the SYCL engine to use the same rolling history profile, set
its JSON once to:

```text
--expert-profile
/work/Strata/data/expert-profile-v9-history.bin
```

No JSON path needs to change again.  Each pre-start refresh atomically replaces
the file at the same path.

## Tuning cadence

Running the updater at every start is cheap because it parses routing traces,
not model weights.

The defaults retain at most 64 trace files from the last 60 days and use a
14-day half-life.

For a shared/public machine with changing users, a shorter half-life such as
7 days is reasonable:

```bash
bash tools/vulkan_v9_prestart_update.sh --half-life-days 7
```

For one person's relatively stable workload, 14–30 days is usually a better
starting point.

## Safety

The updater writes new profile/plan/report files atomically.  It never modifies
the shipped `expert-profile.bin`.

The pre-start wrapper only moves the previous
`v9-production-routing.bin` into `routing-history/`; it does not delete archived
traces.
