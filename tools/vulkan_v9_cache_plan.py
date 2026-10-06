#!/usr/bin/env python3
"""JR-Strata-Vulkan V9 whole-model expert-cache planner.

Inputs:
  * Strata expert-profile.bin (STRP v1)
  * native_experts.txt from the model's native pack
  * a VRAM cache budget

Output:
  * JRVKC9 residency table: (layer, expert) -> VRAM slot or -1

The planner deliberately uses the same 256-byte slot alignment as Strata's
variable-sized ExpertCache::open_sized().  It preserves the profile as a strict
prefix: the first ranked pair that does not fit ends the resident prefix.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import struct


MAGIC = b"JRVKC9\0\0"
VERSION = 1
PROFILE_MAGIC = b"STRP"
PROFILE_VERSION = 1
ALIGN = 256


@dataclass
class LayerInfo:
    layer: int
    gu_type: int
    d_type: int
    offset: int
    blob_bytes: int


def align_up(v: int, a: int = ALIGN) -> int:
    return (v + a - 1) // a * a


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
        layers[l] = LayerInfo(l, gt, dt, off, blob)

    if not layers:
        raise SystemExit(f"{path}: no layer rows")
    if n_expert is None:
        n_expert = 512
    if total is None:
        total = sum(x.blob_bytes * n_expert for x in layers.values())
    return layers, n_expert, total


def read_profile(path: Path):
    blob = path.read_bytes()
    if len(blob) < 24 or blob[:4] != PROFILE_MAGIC:
        raise SystemExit(f"{path}: not a STRP profile")
    ver, nl, ne, slots, n = struct.unpack_from("<5I", blob, 4)
    if ver != PROFILE_VERSION:
        raise SystemExit(f"{path}: unsupported STRP version {ver}")
    need = 24 + 4 * n
    if len(blob) < need:
        raise SystemExit(f"{path}: truncated ranked-pair list")
    ranked = [
        struct.unpack_from("<HH", blob, 24 + 4 * i)
        for i in range(n)
    ]
    return nl, ne, slots, ranked


def build_plan(layers, nl, ne, ranked, budget_bytes):
    slots = [-1] * (nl * ne)
    used = 0
    resident = 0
    per_layer = [0] * nl

    for layer, expert in ranked:
        if layer >= nl or expert >= ne:
            raise SystemExit(f"profile pair outside dimensions: ({layer}, {expert})")
        if layer not in layers:
            raise SystemExit(f"profile references layer {layer}, absent from native_experts.txt")
        need = align_up(layers[layer].blob_bytes)
        if used + need > budget_bytes:
            break
        slots[layer * ne + expert] = resident
        resident += 1
        per_layer[layer] += 1
        used += need

    return slots, resident, used, per_layer


def write_plan(path: Path, nl, ne, resident, budget, used, slots):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack(
            "<IIIIIQQ",
            VERSION, nl, ne, resident, 0, budget, used))
        f.write(struct.pack(f"<{len(slots)}i", *slots))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--profile",
        default="/data/strata-lab/Strata/data/expert-profile.bin")
    ap.add_argument(
        "--native-experts",
        default="/data/strata-lab/data/packs/swift-iq3_xxs/native_experts.txt")
    ap.add_argument(
        "--budget-gib", type=float, default=17.22,
        help="expert VRAM cache budget; default mirrors the established B60 SYCL cache")
    ap.add_argument(
        "--out",
        default="/data/strata-lab/JR-Strata-Vulkan/data/v9-cache-plan.bin")
    args = ap.parse_args()

    layers, manifest_ne, total = read_native_experts(Path(args.native_experts))
    nl, ne, profile_slots, ranked = read_profile(Path(args.profile))

    if ne != manifest_ne:
        raise SystemExit(
            f"profile has {ne} experts/layer but native manifest says {manifest_ne}")
    if set(layers) != set(range(nl)):
        raise SystemExit(
            f"native manifest layers do not exactly cover 0..{nl-1}")

    budget = int(args.budget_gib * (1 << 30))
    slots, resident, used, per_layer = build_plan(
        layers, nl, ne, ranked, budget)

    write_plan(Path(args.out), nl, ne, resident, budget, used, slots)

    print("JR-Strata-Vulkan V9 cache plan")
    print(f"  profile        : {args.profile}")
    print(f"  native experts : {args.native_experts}")
    print(f"  dimensions     : {nl} layers x {ne} experts = {nl*ne}")
    print(f"  profile ranked : {len(ranked)} pairs (header slots={profile_slots})")
    print(f"  expert bytes   : {total / (1<<30):.3f} GiB total")
    print(f"  budget         : {budget / (1<<30):.3f} GiB")
    print(f"  used           : {used / (1<<30):.3f} GiB")
    print(f"  resident       : {resident} experts ({resident/(nl*ne)*100:.2f}%)")
    print(f"  host fallback  : {(total-used) / (1<<30):.3f} GiB raw expert bytes")
    print(f"  per-layer min  : {min(per_layer)}")
    print(f"  per-layer max  : {max(per_layer)}")
    print(f"  per-layer mean : {sum(per_layer)/len(per_layer):.2f}")
    print("  layer counts   :")
    for base in range(0, nl, 8):
        chunk = " ".join(
            f"{l:02d}:{per_layer[l]:3d}"
            for l in range(base, min(base + 8, nl)))
        print("   ", chunk)
    print(f"  wrote          : {args.out}")


if __name__ == "__main__":
    main()
