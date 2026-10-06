#!/usr/bin/env python3
"""JR-Strata-Vulkan V8 A/B: generic V7 vs format-specialized V8."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import iq_pack as P
import vulkan_routed_expert_parity as V4


def make_bundle(model, layer, x, k, vram_count, work):
    n_expert = int(model.where["blk.0.ffn_gate_inp.weight"][1].shape[1])
    ids, weights = V4.router_weights(model, layer, x, k)
    xq = V4.q8_quant_dequant(x)
    records = []
    for eid in ids:
        gt, dt, gr, ur, dr, _ = V4.expert_output(
            model, layer, int(eid), n_expert, xq)
        records.append((gt, dt, gr, ur, dr))

    stem = f"layer{layer:02d}-v{vram_count}"
    bundle = work / f"{stem}.jrvk"
    xp = work / f"{stem}.x.f32"
    V4.write_bundle(bundle, layer, ids, weights, records, vram_count)
    xp.write_bytes(np.ascontiguousarray(x, dtype="<f4").tobytes())
    return bundle, xp


def run_engine(exe, bundle, xp, yp, env, warmup, iters):
    cp = subprocess.run([
        exe,
        "--bundle", str(bundle),
        "--x", str(xp),
        "--out", str(yp),
        "--warmup", str(warmup),
        "--iters", str(iters),
    ], env=env, text=True, capture_output=True)
    print(cp.stdout, end="")
    if cp.stderr:
        print(cp.stderr, end="", file=sys.stderr)
    if cp.returncode:
        raise SystemExit(cp.returncode)

    m_ms = re.search(r"GPU avg\s*:\s*([0-9.eE+-]+)\s*ms", cp.stdout)
    m_bw = re.search(
        r"effective routed weight throughput:\s*([0-9.eE+-]+)\s*GiB/s",
        cp.stdout)
    if not m_ms or not m_bw:
        raise SystemExit("could not parse benchmark output")
    return float(m_ms.group(1)), float(m_bw.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gguf",
        default="/data/strata-lab/data/models/swift-IQ3_XXS/"
                "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf")
    ap.add_argument(
        "--v7",
        default=str(ROOT / "build-vulkan" / "jr-vk-routed-engine-v7"))
    ap.add_argument(
        "--v8",
        default=str(ROOT / "build-vulkan" / "jr-vk-routed-engine-v8"))
    ap.add_argument("--layers", default="18,35")
    ap.add_argument(
        "--tiers", default="0,5,9,10",
        help="VRAM expert count out of top-k; 9/1 is useful for hot-cache projections")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    model = P.Model(Path(args.gguf))
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    tiers = [int(x) for x in args.tiers.split(",") if x.strip()]

    rng = np.random.default_rng(0x4A525638)
    base_x = rng.standard_normal(2560).astype(np.float32) * np.float32(0.05)

    tmp_base = Path(os.environ.get("JR_VK_TMP", "/data/strata-lab/vulkan-tmp"))
    tmp_base.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("JR_VK_DEVICE", "8086:e211")

    summaries = []

    print("JR-Strata-Vulkan V8 specialization A/B benchmark")
    print("  V7 :", args.v7)
    print("  V8 :", args.v8)
    print("  layers :", layers)
    print("  tiers  :", tiers)
    print("  warmup :", args.warmup)
    print("  iters  :", args.iters)
    print()

    with tempfile.TemporaryDirectory(prefix="jr-vk-v8-bench-", dir=tmp_base) as td:
        work = Path(td)
        for layer in layers:
            x = (base_x * np.float32(1.0 + (layer % 7) * 0.01)).astype(np.float32)
            for nv in tiers:
                bundle, xp = make_bundle(model, layer, x, args.k, nv, work)

                print("=" * 102)
                print(f"layer {layer}  VRAM/RAM={nv}/{args.k-nv}  engine=V7")
                v7_ms, v7_bw = run_engine(
                    args.v7, bundle, xp, work / "v7.f32",
                    env, args.warmup, args.iters)

                print("-" * 102)
                print(f"layer {layer}  VRAM/RAM={nv}/{args.k-nv}  engine=V8")
                v8_ms, v8_bw = run_engine(
                    args.v8, bundle, xp, work / "v8.f32",
                    env, args.warmup, args.iters)

                speedup = v7_ms / v8_ms
                latency_drop = (1.0 - v8_ms / v7_ms) * 100.0
                bw_gain = (v8_bw / v7_bw - 1.0) * 100.0
                summaries.append(
                    (layer, nv, args.k - nv, v7_ms, v8_ms,
                     speedup, latency_drop, v7_bw, v8_bw, bw_gain))

    print()
    print("=" * 102)
    print("V8 SUMMARY")
    print("layer  V/R    V7-ms    V8-ms   speedup  latency-drop   V7-GiB/s  V8-GiB/s  BW-gain")
    for s in summaries:
        layer, nv, nr, a, b, sp, ld, bwa, bwb, bg = s
        print(
            f"{layer:>5}  {nv}/{nr:<2}  "
            f"{a:>7.4f}  {b:>7.4f}   {sp:>6.3f}x   "
            f"{ld:>8.2f}%    {bwa:>8.3f}  {bwb:>8.3f}  {bg:>7.2f}%"
        )


if __name__ == "__main__":
    main()
