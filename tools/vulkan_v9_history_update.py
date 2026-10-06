#!/usr/bin/env python3
"""JR-Strata-Vulkan V9 history-aware expert cache manager.

Purpose
-------
Build one *rolling* expert profile/cache plan from multiple historical routing
traces instead of overfitting a single prompt/session.

Policy
------
For each trace:
  1. Count routed (layer, expert) selections.
  2. Normalize within each layer, so one very long trace does not automatically
     dominate every shorter trace.
  3. Apply a recency weight with an exponential half-life.
  4. Down-weight very short traces until they reach --full-weight-positions.

Across traces:
  weighted_heat(layer, expert) =
      sum(trace_weight * per-layer selection probability)

Pairs are ranked by weighted_heat.  The shipped/base profile is used only as a
stable tie-break/fallback for unseen pairs.  Expert byte size is used for
budgeting, not divided into the heat score: imported-RAM miss latency is
approximately proportional to blob bytes, so "heat / bytes" would double
penalize large experts.

Outputs
-------
1. A Strata-compatible STRP v1 profile for SYCL / CUDA use.
2. A JRVKC9 whole-model residency plan for JR-Strata-Vulkan.
3. A JSON report with trace weights, resident counts and optional latest-trace
   cross-validation.

The output files are replaced atomically.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import glob
import json
import math
from pathlib import Path
import struct
import tempfile
import os


STRP_MAGIC = b"STRP"
STRP_VERSION = 1
PLAN_MAGIC = b"JRVKC9\0\0"
PLAN_VERSION = 1
ALIGN = 256


@dataclass
class LayerInfo:
    layer: int
    gu_type: int
    d_type: int
    blob_bytes: int


@dataclass
class TraceStats:
    path: Path
    mtime: float
    age_days: float
    positions: float
    weight: float
    counts: dict
    layer_total: list[int]


def align_up(v: int, a: int = ALIGN) -> int:
    return (v + a - 1) // a * a


def atomic_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_text(path: Path, text: str):
    atomic_bytes(path, text.encode("utf-8"))


def read_native_experts(path: Path):
    layers = {}
    n_expert = None
    total = None
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            import re
            m = re.search(r"n_expert\s+(\d+)", s)
            if m:
                n_expert = int(m.group(1))
            m = re.search(r"total\s+(\d+)", s)
            if m:
                total = int(m.group(1))
            continue
        f = s.split()
        if len(f) < 8:
            raise SystemExit(f"{path}: malformed native expert row: {line}")
        l, gt, dt, off, blob = map(int, f[:5])
        layers[l] = LayerInfo(l, gt, dt, blob)

    if not layers:
        raise SystemExit(f"{path}: no layer rows")
    if n_expert is None:
        n_expert = 512
    if total is None:
        total = sum(x.blob_bytes * n_expert for x in layers.values())
    return layers, n_expert, total


def read_profile(path: Path):
    b = path.read_bytes()
    if len(b) < 24 or b[:4] != STRP_MAGIC:
        raise SystemExit(f"{path}: not a STRP profile")
    ver, nl, ne, slots, n = struct.unpack_from("<5I", b, 4)
    if ver != STRP_VERSION:
        raise SystemExit(f"{path}: unsupported STRP version {ver}")
    if len(b) < 24 + 4 * n:
        raise SystemExit(f"{path}: truncated ranked list")
    ranked = [struct.unpack_from("<HH", b, 24 + 4 * i) for i in range(n)]
    return nl, ne, slots, ranked


def write_profile_bytes(nl: int, ne: int, ranked):
    table = [[-1] * ne for _ in range(nl)]
    for slot, (l, e) in enumerate(ranked):
        table[l][e] = slot

    out = bytearray()
    out += STRP_MAGIC
    out += struct.pack("<5I", STRP_VERSION, nl, ne, len(ranked), len(ranked))
    for l, e in ranked:
        out += struct.pack("<HH", l, e)
    for l in range(nl):
        out += struct.pack(f"<{ne}i", *table[l])
    return bytes(out)


def parse_trace(path: Path, nl: int, ne: int):
    b = path.read_bytes()
    off = 0
    counts = defaultdict(int)
    layer_total = [0] * nl
    records = 0

    while off + 8 <= len(b):
        layer, k = struct.unpack_from("<ii", b, off)
        off += 8
        if k <= 0 or k > 64:
            raise SystemExit(f"{path}: invalid k={k} at record {records}")
        need = 8 * k
        if off + need > len(b):
            # An interrupted final record is ignored; the already-complete
            # routing history is still useful.
            break

        ids = struct.unpack_from(f"<{k}i", b, off)
        off += 4 * k
        off += 4 * k  # weights

        if 0 <= layer < nl:
            for e in ids:
                if 0 <= e < ne:
                    counts[(layer, e)] += 1
                    layer_total[layer] += 1
        records += 1

    nonzero_layers = [x for x in layer_total if x]
    if not nonzero_layers:
        raise SystemExit(f"{path}: no usable routing records")

    # k selections per layer position. Median is robust to a trace stopped
    # halfway through its last token/layer sweep.
    sorted_tot = sorted(nonzero_layers)
    median_total = sorted_tot[len(sorted_tot) // 2]
    # Infer k from the common divisibility only approximately; positions is
    # used for confidence weighting, not file parsing correctness.
    positions = median_total / 10.0

    return counts, layer_total, records, positions


def discover_traces(history_dir: Path, extra_glob: str, max_files: int, max_age_days: float):
    now = datetime.now(timezone.utc).timestamp()
    paths = set()

    if history_dir.exists():
        paths.update(p for p in history_dir.glob("*.bin") if p.is_file())

    if extra_glob:
        paths.update(Path(p) for p in glob.glob(extra_glob) if Path(p).is_file())

    rows = []
    for p in paths:
        st = p.stat()
        age = max(0.0, (now - st.st_mtime) / 86400.0)
        if max_age_days > 0 and age > max_age_days:
            continue
        rows.append((st.st_mtime, p, age))

    rows.sort(reverse=True)
    if max_files > 0:
        rows = rows[:max_files]
    rows.sort()  # chronological for reporting / latest holdout
    return rows


def trace_stats(path: Path, age_days: float, nl: int, ne: int,
                half_life_days: float, full_weight_positions: float):
    counts, layer_total, records, positions = parse_trace(path, nl, ne)

    if half_life_days > 0:
        decay = math.exp(-math.log(2.0) * age_days / half_life_days)
    else:
        decay = 1.0

    if full_weight_positions > 0:
        confidence = min(1.0, math.sqrt(max(positions, 1.0) / full_weight_positions))
    else:
        confidence = 1.0

    weight = decay * confidence
    return TraceStats(path, path.stat().st_mtime, age_days, positions,
                      weight, counts, layer_total)


def build_heat(stats: list[TraceStats], nl: int, ne: int):
    heat = [[0.0] * ne for _ in range(nl)]
    for t in stats:
        for (l, e), c in t.counts.items():
            denom = t.layer_total[l]
            if denom:
                heat[l][e] += t.weight * (c / denom)
    return heat


def rank_pairs(heat, base_ranked, nl: int, ne: int):
    base_rank = {(l, e): i for i, (l, e) in enumerate(base_ranked)}
    fallback = len(base_ranked) + nl * ne

    pairs = [(l, e) for l in range(nl) for e in range(ne)]
    pairs.sort(key=lambda p: (
        -heat[p[0]][p[1]],
        base_rank.get(p, fallback + p[0] * ne + p[1]),
        p[0], p[1]
    ))
    return pairs


def build_plan(layers, nl: int, ne: int, ranked, budget: int):
    slots = [-1] * (nl * ne)
    used = 0
    resident = 0
    per_layer = [0] * nl

    for l, e in ranked:
        need = align_up(layers[l].blob_bytes)
        if used + need > budget:
            break
        slots[l * ne + e] = resident
        resident += 1
        per_layer[l] += 1
        used += need

    return slots, resident, used, per_layer


def write_plan_bytes(nl, ne, resident, budget, used, slots):
    out = bytearray()
    out += PLAN_MAGIC
    out += struct.pack("<IIIIIQQ",
                       PLAN_VERSION, nl, ne, resident, 0, budget, used)
    out += struct.pack(f"<{len(slots)}i", *slots)
    return bytes(out)


def score_trace(path: Path, slots, nl: int, ne: int):
    counts, layer_total, records, positions = parse_trace(path, nl, ne)
    hits = 0
    total = 0
    for (l, e), c in counts.items():
        total += c
        if slots[l * ne + e] >= 0:
            hits += c
    return hits, total, (hits / total if total else 0.0)


def build_from_stats(stats, layers, base_ranked, nl, ne, budget):
    heat = build_heat(stats, nl, ne)
    ranked = rank_pairs(heat, base_ranked, nl, ne)
    slots, resident, used, per_layer = build_plan(
        layers, nl, ne, ranked, budget)
    return heat, ranked, slots, resident, used, per_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--history-dir",
        default="/data/strata-lab/Strata/data/routing-history")
    ap.add_argument(
        "--extra-glob",
        default="/data/strata-lab/Strata/data/v9-routing-*.bin",
        help="also include matching archived traces outside history-dir")
    ap.add_argument(
        "--base-profile",
        default="/data/strata-lab/Strata/data/expert-profile.bin")
    ap.add_argument(
        "--native-experts",
        default="/data/strata-lab/data/packs/swift-iq3_xxs/native_experts.txt")
    ap.add_argument("--budget-gib", type=float, default=17.57)
    ap.add_argument("--half-life-days", type=float, default=14.0)
    ap.add_argument("--max-age-days", type=float, default=60.0)
    ap.add_argument("--max-files", type=int, default=64)
    ap.add_argument(
        "--full-weight-positions", type=float, default=512.0,
        help="shorter traces are confidence-downweighted; >= this gets full weight")
    ap.add_argument(
        "--profile-out",
        default="/data/strata-lab/Strata/data/expert-profile-v9-history.bin")
    ap.add_argument(
        "--plan-out",
        default="/data/strata-lab/JR-Strata-Vulkan/data/v9-cache-plan-history.bin")
    ap.add_argument(
        "--report-out",
        default="/data/strata-lab/JR-Strata-Vulkan/data/v9-history-report.json")
    ap.add_argument(
        "--no-latest-cv", action="store_true",
        help="skip leave-latest-trace-out cross-validation")
    args = ap.parse_args()

    layers, manifest_ne, total_expert_bytes = read_native_experts(
        Path(args.native_experts))
    nl, ne, base_slots, base_ranked = read_profile(Path(args.base_profile))
    if ne != manifest_ne:
        raise SystemExit("base profile and native manifest disagree on expert count")
    if set(layers) != set(range(nl)):
        raise SystemExit("native manifest does not cover every profile layer")

    found = discover_traces(
        Path(args.history_dir), args.extra_glob,
        args.max_files, args.max_age_days)

    stats = [
        trace_stats(p, age, nl, ne,
                    args.half_life_days, args.full_weight_positions)
        for _, p, age in found
    ]

    budget = int(args.budget_gib * (1 << 30))

    if stats:
        heat, ranked, slots, resident, used, per_layer = build_from_stats(
            stats, layers, base_ranked, nl, ne, budget)
        source = "history"
    else:
        ranked = list(base_ranked)
        slots, resident, used, per_layer = build_plan(
            layers, nl, ne, ranked, budget)
        source = "base-profile fallback"

    atomic_bytes(Path(args.profile_out),
                 write_profile_bytes(nl, ne, ranked))
    atomic_bytes(Path(args.plan_out),
                 write_plan_bytes(nl, ne, resident, budget, used, slots))

    trace_rows = []
    for t in stats:
        h, tot, rate = score_trace(t.path, slots, nl, ne)
        trace_rows.append({
            "path": str(t.path),
            "mtime": datetime.fromtimestamp(
                t.mtime, timezone.utc).isoformat(),
            "age_days": t.age_days,
            "positions_approx": t.positions,
            "weight": t.weight,
            "hits_final_plan": h,
            "routed_final_plan": tot,
            "hit_rate_final_plan": rate,
        })

    cv = None
    if not args.no_latest_cv and len(stats) >= 2:
        train = stats[:-1]
        latest = stats[-1]
        _, _, cv_slots, cv_resident, cv_used, _ = build_from_stats(
            train, layers, base_ranked, nl, ne, budget)
        h, tot, rate = score_trace(latest.path, cv_slots, nl, ne)
        cv = {
            "held_out": str(latest.path),
            "train_trace_count": len(train),
            "resident": cv_resident,
            "used_bytes": cv_used,
            "hits": h,
            "routed": tot,
            "hit_rate": rate,
        }

    report = {
        "schema": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "source": source,
            "half_life_days": args.half_life_days,
            "max_age_days": args.max_age_days,
            "max_files": args.max_files,
            "full_weight_positions": args.full_weight_positions,
            "normalization": "per-trace, per-layer selection probability",
            "base_profile_role": "tie-break/fallback for unseen pairs",
            "expert_size_role": "budget packing only",
        },
        "model": {
            "layers": nl,
            "experts_per_layer": ne,
            "pairs": nl * ne,
            "raw_expert_bytes": total_expert_bytes,
        },
        "cache": {
            "budget_bytes": budget,
            "used_bytes": used,
            "resident": resident,
            "per_layer_min": min(per_layer),
            "per_layer_max": max(per_layer),
            "per_layer_mean": sum(per_layer) / len(per_layer),
            "per_layer": per_layer,
        },
        "traces": trace_rows,
        "latest_cross_validation": cv,
        "outputs": {
            "profile": args.profile_out,
            "plan": args.plan_out,
        },
    }
    atomic_text(Path(args.report_out), json.dumps(report, indent=2) + "\n")

    print("JR-Strata-Vulkan V9 history-aware cache update")
    print(f"  source         : {source}")
    print(f"  traces used    : {len(stats)}")
    print(f"  half-life      : {args.half_life_days:.1f} days")
    print(f"  budget         : {budget/(1<<30):.3f} GiB")
    print(f"  used           : {used/(1<<30):.3f} GiB")
    print(f"  resident       : {resident}/{nl*ne}")
    print(f"  per-layer      : min {min(per_layer)}, "
          f"mean {sum(per_layer)/len(per_layer):.2f}, max {max(per_layer)}")
    print()
    if stats:
        print("  trace weights / final-plan scores:")
        for t, row in zip(stats, trace_rows):
            print(
                f"    {t.path.name}: age={t.age_days:.2f}d "
                f"positions~{t.positions:.0f} weight={t.weight:.4f} "
                f"hit={row['hit_rate_final_plan']*100:.2f}%")
    if cv is not None:
        print()
        print("  latest-trace cross-validation:")
        print(f"    trained on   : {cv['train_trace_count']} older trace(s)")
        print(f"    held out     : {Path(cv['held_out']).name}")
        print(f"    hit rate     : {cv['hit_rate']*100:.3f}% "
              f"({cv['hits']}/{cv['routed']})")
    print()
    print(f"  profile wrote  : {args.profile_out}")
    print(f"  plan wrote     : {args.plan_out}")
    print(f"  report wrote   : {args.report_out}")


if __name__ == "__main__":
    main()
