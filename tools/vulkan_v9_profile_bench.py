#!/usr/bin/env python3
"""JR-Strata-Vulkan V9 profile-driven cache benchmark.

This is the bridge from synthetic `--vram-count N` tests to a real whole-model
cache policy.  Each routed expert is marked VRAM/RAM from the JRVKC9 residency
table generated from expert-profile.bin and native_experts.txt.

For GPU timing the same real expert bytes and V8 specialized kernels are used;
the V9 executable itself resolves the hit/miss tier from the residency table.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import iq_pack as P
import vulkan_routed_expert_parity as V4

PLAN_MAGIC = b"JRVKC9\0\0"


def read_plan(path: Path):
    b = path.read_bytes()
    if len(b) < 44 or b[:8] != PLAN_MAGIC:
        raise SystemExit(f"{path}: not a V9 residency plan")
    ver, nl, ne, nr, flags, budget, used = struct.unpack_from("<IIIIIQQ", b, 8)
    if ver != 1:
        raise SystemExit(f"{path}: unsupported plan version {ver}")
    off = 8 + struct.calcsize("<IIIIIQQ")
    n = nl * ne
    need = off + 4 * n
    if len(b) < need:
        raise SystemExit(f"{path}: truncated residency table")
    slots = np.frombuffer(b, dtype="<i4", count=n, offset=off).reshape(nl, ne)
    return nl, ne, nr, budget, used, slots


def write_bundle_raw(path: Path, model: P.Model, layer: int, ids, weights, n_expert: int):
    with path.open("wb") as f:
        f.write(b"JRVKRT1\0")
        f.write(struct.pack("<IIIII", 1, layer, len(ids), 2560, 640))
        for eid, weight in zip(ids, weights):
            gt, gate = V4.role_bytes(model, layer, "gate", int(eid), n_expert)
            _, up = V4.role_bytes(model, layer, "up", int(eid), n_expert)
            dt, down = V4.role_bytes(model, layer, "down", int(eid), n_expert)

            # Bundle tier is intentionally a dummy. V9 overrides it through
            # the whole-model residency map.
            f.write(struct.pack(
                "<iiiIfIII",
                int(eid), int(gt.type_id), int(dt.type_id), 0,
                float(weight), len(gate), len(up), len(down)))
            f.write(gate.tobytes())
            f.write(up.tobytes())
            f.write(down.tobytes())


def parse_gpu_stats(stdout: str):
    def one(pattern):
        m = re.search(pattern, stdout)
        return float(m.group(1)) if m else None
    return (
        one(r"GPU avg\s*:\s*([0-9.eE+-]+)\s*ms"),
        one(r"effective routed weight throughput:\s*([0-9.eE+-]+)\s*GiB/s"),
    )


def parse_layers(s: str):
    if s.strip().lower() == "all":
        return list(range(48))
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gguf",
        default="/data/strata-lab/data/models/swift-IQ3_XXS/"
                "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf")
    ap.add_argument(
        "--plan",
        default="/data/strata-lab/JR-Strata-Vulkan/data/v9-cache-plan.bin")
    ap.add_argument(
        "--exe",
        default=str(ROOT / "build-vulkan" / "jr-vk-routed-engine-v9"))
    ap.add_argument("--layers", default="18,35")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    nl, ne, nr, budget, used, slots = read_plan(Path(args.plan))
    model = P.Model(Path(args.gguf))
    layers = parse_layers(args.layers)
    if any(l < 0 or l >= nl for l in layers):
        raise SystemExit("layer outside residency plan")

    env = os.environ.copy()
    env.setdefault("JR_VK_DEVICE", "8086:e211")
    tmp_base = Path(os.environ.get(
        "JR_VK_TMP", "/data/strata-lab/vulkan-tmp"))
    tmp_base.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0x4A525639)

    total_routes = 0
    total_hits = 0
    total_ms = 0.0
    layer_rows = []

    print("JR-Strata-Vulkan V9 profile-cache benchmark")
    print(f"  plan resident : {nr}/{nl*ne} experts")
    print(f"  plan budget   : {budget/(1<<30):.3f} GiB")
    print(f"  plan used     : {used/(1<<30):.3f} GiB")
    print(f"  layers        : {layers}")
    print(f"  samples/layer : {args.samples}")
    print(f"  top-k         : {args.k}")
    print(f"  warmup/iters  : {args.warmup}/{args.iters}")
    print()

    with tempfile.TemporaryDirectory(
            prefix="jr-vk-v9-", dir=tmp_base) as td:
        td = Path(td)
        for layer in layers:
            l_hits = 0
            l_routes = 0
            l_ms = 0.0
            l_bw = 0.0

            for sample in range(args.samples):
                x = (rng.standard_normal(2560).astype(np.float32) *
                     np.float32(0.05))
                ids, weights = V4.router_weights(
                    model, layer, x, args.k)

                hits = int(np.count_nonzero(slots[layer, ids] >= 0))
                l_hits += hits
                l_routes += args.k

                bundle = td / f"l{layer:02d}-s{sample:03d}.jrvk"
                xp = td / f"l{layer:02d}-s{sample:03d}.x.f32"
                yp = td / f"l{layer:02d}-s{sample:03d}.y.f32"

                write_bundle_raw(
                    bundle, model, layer, ids, weights, ne)
                xp.write_bytes(
                    np.ascontiguousarray(x, dtype="<f4").tobytes())

                cp = subprocess.run([
                    args.exe,
                    "--bundle", str(bundle),
                    "--x", str(xp),
                    "--out", str(yp),
                    "--residency", args.plan,
                    "--warmup", str(args.warmup),
                    "--iters", str(args.iters),
                ], env=env, text=True, capture_output=True)

                if cp.returncode:
                    print(cp.stdout)
                    print(cp.stderr, file=sys.stderr)
                    raise SystemExit(cp.returncode)

                ms, bw = parse_gpu_stats(cp.stdout)
                if ms is None or bw is None:
                    print(cp.stdout)
                    raise SystemExit("could not parse V9 engine timing")
                l_ms += ms
                l_bw += bw

                tags = " ".join(
                    f"{int(e)}{'V' if slots[layer, int(e)] >= 0 else 'R'}"
                    for e in ids)
                print(
                    f"layer {layer:2d} sample {sample:2d}  "
                    f"hits={hits}/{args.k}  avg={ms:.4f} ms  route={tags}")

            hit_rate = l_hits / l_routes
            avg_ms = l_ms / args.samples
            avg_bw = l_bw / args.samples
            layer_rows.append((layer, hit_rate, avg_ms, avg_bw))

            total_hits += l_hits
            total_routes += l_routes
            total_ms += avg_ms

    print()
    print("=" * 78)
    print("V9 SUMMARY")
    print("layer   hit-rate   avg-ms   routed-GiB/s")
    for layer, h, ms, bw in layer_rows:
        print(f"{layer:>5}   {h*100:>7.2f}%   {ms:>7.4f}   {bw:>11.3f}")

    global_hit = total_hits / max(total_routes, 1)
    print()
    print(f"sampled top-k hit rate : {global_hit*100:.2f}% "
          f"({total_hits}/{total_routes})")
    print(f"sum avg latency across sampled layers : {total_ms:.4f} ms")

    if len(layers) == 48:
        print(f"48-layer sampled MoE-only budget      : {total_ms:.4f} ms/token")
        if total_ms > 0:
            print(f"48-layer MoE-only ceiling             : {1000.0/total_ms:.2f} tok/s")


if __name__ == "__main__":
    main()
