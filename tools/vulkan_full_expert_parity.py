#!/usr/bin/env python3
"""JR-Strata-Vulkan V2: real Swift full-expert parity.

One command validates the complete expert function on real GGUF bytes:

    gate GEMV
    up GEMV
    SwiGLU
    down GEMV

The Vulkan runner reads all quantized weight bytes directly through
VK_EXT_external_memory_host.  This orchestration script uses bundled llama.cpp
gguf-py only as an independent CPU/Numpy oracle.

Default mode tests one real expert from every gu_type/d_type combination found
in native_experts.txt.  --all-layers tests one expert from all 48 layers.
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

import gguf_reader as G
import iq_pack as P


TYPE_NAME = G.GGML_TYPES


def parse_manifest(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split()
        rows.append({
            "layer": int(f[0]),
            "gu_type": int(f[1]),
            "d_type": int(f[2]),
        })
    if not rows:
        raise RuntimeError(f"{path}: no expert rows")
    return rows


def role_bytes(model: P.Model, layer: int, role: str, expert: int, n_expert: int):
    name = f"blk.{layer}.ffn_{role}_exps.weight"
    _, t, _, _ = model.where[name]
    raw = model.bytes(name).reshape(n_expert, -1)[expert]
    return t, np.asarray(raw, dtype=np.uint8)


def dequant_q2_0(raw: np.ndarray) -> np.ndarray:
    """CPU oracle for Strata/GSQ-RCO Q2_0 (ggml type 42).

    llama.cpp's current gguf-py enum knows Q2_0, but its Python dequantizer
    intentionally has no implementation for it.  The format contract is the
    same one used by Strata's native kernel:

        block_q2_0 = fp16 scale d + 16 packed bytes
        64 values/block
        value = d * (2-bit-code - 1)
    """
    raw = np.asarray(raw, dtype=np.uint8).reshape(-1)
    if raw.size % 18:
        raise ValueError(f"Q2_0 byte count {raw.size} is not a multiple of 18")

    nb = raw.size // 18
    out = np.empty(nb * 64, dtype=np.float32)

    for b in range(nb):
        block = raw[b * 18:(b + 1) * 18]
        # GGUF is little-endian; widening fp16 -> fp32 is exact.
        d = np.frombuffer(block[:2].tobytes(), dtype="<f2", count=1)[0].astype(np.float32)
        qs = block[2:18]

        base = b * 64
        for i in range(64):
            code = (int(qs[i >> 2]) >> ((i & 3) * 2)) & 3
            out[base + i] = np.float32(d * np.float32(code - 1))

    return out


def dequant(raw: np.ndarray, type_name: str, shape):
    # Q2_0 is Strata's GSQ-RCO type 42.  Bundled gguf-py recognizes the enum
    # but raises NotImplementedError for dequantization, so keep an independent
    # CPU oracle here instead of depending on an unimplemented helper.
    if type_name == "Q2_0":
        v = dequant_q2_0(raw)
    else:
        v = P.dequantize(raw, type_name)
    return np.asarray(v, dtype=np.float32).reshape(shape)


def cpu_reference(model: P.Model, layer: int, expert: int, n_expert: int, x: np.ndarray):
    gt, gate_raw = role_bytes(model, layer, "gate", expert, n_expert)
    ut, up_raw = role_bytes(model, layer, "up", expert, n_expert)
    dt, down_raw = role_bytes(model, layer, "down", expert, n_expert)

    if gt.type_id != ut.type_id:
        raise RuntimeError(f"layer {layer}: gate/up types differ")

    # NativeExpertLayout contract in Strata:
    # gate/up: n_ff rows x n_embd
    # down:    n_embd rows x n_ff
    gate = dequant(gate_raw, gt.type_name, (640, 2560))
    up = dequant(up_raw, ut.type_name, (640, 2560))
    down = dequant(down_raw, dt.type_name, (2560, 640))

    g = gate @ x
    u = up @ x
    h = (g / (1.0 + np.exp(-g, dtype=np.float32))) * u
    y = down @ h

    return (
        gt, dt,
        gate_raw, up_raw, down_raw,
        np.asarray(y, dtype=np.float32)
    )


def write(path: Path, a: np.ndarray):
    path.write_bytes(np.ascontiguousarray(a).tobytes())


def run_case(exe: Path, tmp: Path, layer: int, expert: int, gt, dt,
             gate_raw, up_raw, down_raw, x, ref):
    gate_p = tmp / "gate.bin"
    up_p = tmp / "up.bin"
    down_p = tmp / "down.bin"
    x_p = tmp / "x.f32"
    y_p = tmp / "y.f32"

    write(gate_p, gate_raw)
    write(up_p, up_raw)
    write(down_p, down_raw)
    write(x_p, x.astype(np.float32))

    cmd = [
        str(exe),
        "--gu-type", str(gt.type_id),
        "--d-type", str(dt.type_id),
        "--gate", str(gate_p),
        "--up", str(up_p),
        "--down", str(down_p),
        "--x", str(x_p),
        "--out", str(y_p),
        "--n-embd", "2560",
        "--n-ff", "640",
    ]
    env = os.environ.copy()
    env.setdefault("JR_VK_DEVICE", "8086:e211")

    cp = subprocess.run(cmd, env=env, text=True, capture_output=True)
    if cp.returncode:
        print(cp.stdout)
        print(cp.stderr, file=sys.stderr)
        raise RuntimeError(f"Vulkan runner failed for layer {layer}")

    got = np.fromfile(y_p, dtype="<f4")
    if got.size != 2560:
        raise RuntimeError(f"layer {layer}: Vulkan returned {got.size} values")

    diff = got.astype(np.float64) - ref.astype(np.float64)
    max_abs = float(np.max(np.abs(diff)))
    ref_peak = float(np.max(np.abs(ref.astype(np.float64))))
    peak_rel = max_abs / max(1.0, ref_peak)
    rmse = float(np.sqrt(np.mean(diff * diff)))
    ref_rms = float(np.sqrt(np.mean(ref.astype(np.float64) ** 2)))
    nrmse = rmse / max(ref_rms, 1e-12)
    finite = bool(np.isfinite(got).all())

    # This is a full F32 reduction-path comparison, not a bitwise dequant test.
    # The GPU reduction tree/FMA lowering differs from NumPy BLAS.
    ok = finite and nrmse <= 5e-4 and peak_rel <= 2e-3

    print(
        f"layer {layer:2d} expert {expert:3d}  "
        f"{gt.type_name:8s}/{dt.type_name:7s}  "
        f"max_abs={max_abs:.3e}  peak_rel={peak_rel:.3e}  "
        f"nrmse={nrmse:.3e}  {'PASS' if ok else 'FAIL'}"
    )

    if not ok:
        worst = np.argsort(np.abs(diff))[-8:][::-1]
        for i in worst:
            print(f"    y[{i:4d}] gpu={got[i]: .8e} ref={ref[i]: .8e} diff={diff[i]: .3e}")
        raise RuntimeError(f"full expert parity failed at layer {layer}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gguf",
        default="/data/strata-lab/data/models/swift-IQ3_XXS/"
                "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf")
    ap.add_argument(
        "--pack",
        default="/data/strata-lab/data/packs/swift-iq3_xxs")
    ap.add_argument(
        "--exe",
        default=str(ROOT / "build-vulkan" / "jr-vk-full-expert"))
    ap.add_argument("--all-layers", action="store_true")
    ap.add_argument("--expert", type=int,
                    help="use the same expert index for every layer; default varies by layer")
    args = ap.parse_args()

    gguf = Path(args.gguf)
    pack = Path(args.pack)
    exe = Path(args.exe)
    manifest = pack / "native_experts.txt"

    if not gguf.is_file():
        ap.error(f"GGUF not found: {gguf}")
    if not manifest.is_file():
        ap.error(f"manifest not found: {manifest}")
    if not exe.is_file():
        ap.error(f"Vulkan executable not found: {exe}; build first")

    model = P.Model(gguf)
    rows = parse_manifest(manifest)
    n_expert = int(model.where["blk.0.ffn_gate_inp.weight"][1].shape[1])

    if args.all_layers:
        selected = rows
        mode = f"all {len(selected)} layers"
    else:
        seen = set()
        selected = []
        for r in rows:
            key = (r["gu_type"], r["d_type"])
            if key not in seen:
                selected.append(r)
                seen.add(key)
        mode = f"{len(selected)} unique production format combinations"

    print("JR-Strata-Vulkan V2 Full Expert Runtime")
    print("  GGUF      :", gguf)
    print("  manifest  :", manifest)
    print("  experts   :", n_expert)
    print("  mode      :", mode)
    print("  source    : real GGUF expert bytes")
    print("  Vulkan    : imported system RAM -> dequant -> gate/up GEMV -> SwiGLU -> down GEMV")
    print()

    rng = np.random.default_rng(0x4A52564B)
    base_x = (rng.standard_normal(2560).astype(np.float32) * np.float32(0.05))

    tmp_base = Path(os.environ.get("JR_VK_TMP", "/data/strata-lab/vulkan-tmp"))
    tmp_base.mkdir(parents=True, exist_ok=True)

    passed = 0
    for r in selected:
        layer = r["layer"]
        expert = args.expert if args.expert is not None else ((layer * 37 + 11) % n_expert)
        if not (0 <= expert < n_expert):
            raise SystemExit(f"expert index {expert} outside 0..{n_expert-1}")

        # Slight deterministic layer variation while keeping activations bounded.
        x = base_x * np.float32(1.0 + (layer % 7) * 0.01)

        gt, dt, gate_raw, up_raw, down_raw, ref = cpu_reference(
            model, layer, expert, n_expert, x)

        if gt.type_id != r["gu_type"] or dt.type_id != r["d_type"]:
            raise RuntimeError(
                f"manifest/type mismatch layer {layer}: "
                f"{r['gu_type']}/{r['d_type']} vs {gt.type_id}/{dt.type_id}")

        with tempfile.TemporaryDirectory(prefix=f"jr-vk-expert-l{layer:02d}-", dir=tmp_base) as td:
            run_case(
                exe, Path(td), layer, expert, gt, dt,
                gate_raw, up_raw, down_raw, x, ref)
        passed += 1

    print()
    print(f"JR-VK V2 full expert: PASS ({passed}/{len(selected)} real production cases)")
    if not args.all_layers:
        print("Next confidence run:")
        print("  python3 tools/vulkan_full_expert_parity.py --all-layers")


if __name__ == "__main__":
    main()
