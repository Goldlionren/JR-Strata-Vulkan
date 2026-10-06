#!/usr/bin/env python3
"""JR-Strata-Vulkan V6 hardware-dot A/B benchmark.

Runs the same real routed bundles through:
  JR_VK_DOT=soft  -> V5 software packed-int8 dot
  JR_VK_DOT=auto  -> V6 hardware OpSDot only when driver marks it accelerated

Optional --modes soft,hw forces the hardware shader even if the acceleration
property is false, as long as shaderIntegerDotProduct itself is supported.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
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
    yp = work / f"{stem}.y.f32"
    V4.write_bundle(bundle, layer, ids, weights, records, vram_count)
    xp.write_bytes(np.ascontiguousarray(x, dtype="<f4").tobytes())
    return bundle, xp, yp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gguf",
        default="/data/strata-lab/data/models/swift-IQ3_XXS/"
                "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf")
    ap.add_argument(
        "--exe",
        default=str(ROOT / "build-vulkan" / "jr-vk-routed-engine-v6"))
    ap.add_argument("--layers", default="18,35")
    ap.add_argument("--tiers", default="0,5,10")
    ap.add_argument("--modes", default="soft,auto")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    model = P.Model(Path(args.gguf))
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    tiers = [int(x) for x in args.tiers.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]

    rng = np.random.default_rng(0x4A525636)
    base_x = rng.standard_normal(2560).astype(np.float32) * np.float32(0.05)

    tmp_base = Path(os.environ.get("JR_VK_TMP", "/data/strata-lab/vulkan-tmp"))
    tmp_base.mkdir(parents=True, exist_ok=True)

    print("JR-Strata-Vulkan V6 integer-dot A/B benchmark")
    print("  layers :", layers)
    print("  tiers  :", tiers)
    print("  modes  :", modes)
    print("  warmup :", args.warmup)
    print("  iters  :", args.iters)
    print()

    with tempfile.TemporaryDirectory(prefix="jr-vk-v6-bench-", dir=tmp_base) as td:
        work = Path(td)
        for layer in layers:
            x = (base_x * np.float32(1.0 + (layer % 7) * 0.01)).astype(np.float32)
            for nv in tiers:
                bundle, xp, yp = make_bundle(
                    model, layer, x, args.k, nv, work)
                for mode in modes:
                    print("=" * 96)
                    print(f"layer {layer}  VRAM/RAM={nv}/{args.k-nv}  dot={mode}")
                    env = os.environ.copy()
                    env.setdefault("JR_VK_DEVICE", "8086:e211")
                    env["JR_VK_DOT"] = mode
                    cp = subprocess.run([
                        args.exe,
                        "--bundle", str(bundle),
                        "--x", str(xp),
                        "--out", str(yp),
                        "--warmup", str(args.warmup),
                        "--iters", str(args.iters),
                    ], env=env, text=True)
                    if cp.returncode:
                        raise SystemExit(cp.returncode)
                    print()


if __name__ == "__main__":
    main()
