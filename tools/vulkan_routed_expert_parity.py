#!/usr/bin/env python3
"""JR-Strata-Vulkan V4 routed expert parity.

V4 contract starts at the same boundary as Strata native_expert_grouped:
routing decisions are already known.  This driver obtains those decisions from
the REAL layer router as an independent CPU oracle, then feeds the selected
top-10 expert IDs + normalized route weights to the Vulkan routed engine.

The Vulkan path validates:
  * q8_1 input activation
  * top-10 routed experts
  * native quantized weights
  * mixed VRAM + imported-RAM expert tiers
  * gate/up -> SwiGLU -> q8_1 -> down
  * router-weighted accumulation

Default representative layers cover both Q2_0 and IQ4_NL down paths and all
major GU formats. --all-layers is the strong 48-layer gate.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import gguf_reader as G
import iq_pack as P


def dequant_q2_0(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.uint8).reshape(-1)
    if raw.size % 18:
        raise ValueError("Q2_0 bytes not multiple of 18")
    nb = raw.size // 18
    out = np.empty(nb * 64, dtype=np.float32)
    for b in range(nb):
        block = raw[b * 18:(b + 1) * 18]
        d = np.frombuffer(block[:2].tobytes(), dtype="<f2", count=1)[0].astype(np.float32)
        qs = block[2:18]
        for i in range(64):
            code = (int(qs[i >> 2]) >> ((i & 3) * 2)) & 3
            out[b * 64 + i] = np.float32(d * np.float32(code - 1))
    return out


def dequant(raw: np.ndarray, type_name: str, shape):
    if type_name == "Q2_0":
        v = dequant_q2_0(raw)
    else:
        v = P.dequantize(raw, type_name)
    return np.asarray(v, dtype=np.float32).reshape(shape)


def role_bytes(model: P.Model, layer: int, role: str, expert: int, n_expert: int):
    name = f"blk.{layer}.ffn_{role}_exps.weight"
    _, t, _, _ = model.where[name]
    raw = model.bytes(name).reshape(n_expert, -1)[expert]
    return t, np.asarray(raw, dtype=np.uint8)


def round_away(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.floor(np.abs(x) + np.float32(0.5))


def q8_quant_dequant(x: np.ndarray) -> np.ndarray:
    """Strata q8_1 value path: q uses full-f32 d, dot reads fp16(d)."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size % 32:
        raise ValueError("q8_1 vector width must be divisible by 32")
    out = np.empty_like(x)
    for b in range(x.size // 32):
        v = x[b * 32:(b + 1) * 32]
        amax = np.max(np.abs(v)).astype(np.float32)
        d = np.float32(min(float(amax / np.float32(127.0)), 65504.0))
        if amax == 0:
            q = np.zeros(32, dtype=np.int8)
        else:
            qf = round_away(v / d)
            q = np.clip(qf, -127, 127).astype(np.int8)
        # Stored q8_1 d is fp16 and the dot product reads that widened value.
        dh = np.float16(d).astype(np.float32)
        out[b * 32:(b + 1) * 32] = q.astype(np.float32) * dh
    return out


def router_weights(model: P.Model, layer: int, x: np.ndarray, k: int):
    name = f"blk.{layer}.ffn_gate_inp.weight"
    _, t, _, _ = model.where[name]
    raw = model.bytes(name)
    w = P.dequantize(raw, t.type_name).astype(np.float32)

    n_expert = int(t.shape[1])
    n_embd = int(t.shape[0])
    if n_embd != x.size:
        raise RuntimeError(
            f"router {name} shape {t.shape} incompatible with x={x.size}")
    w = w.reshape(n_expert, n_embd)
    logits = w @ x

    # Strata semantics: softmax all experts, stable descending top-k, then
    # renormalize selected probabilities with a 2^-14 lower clamp.
    mx = float(np.max(logits))
    ex = np.exp(logits.astype(np.float64) - mx)
    p = (ex / np.sum(ex)).astype(np.float32)
    ids = np.argsort(-p, kind="stable")[:k].astype(np.int32)
    selected = p[ids].astype(np.float64)
    denom = max(float(np.sum(selected)), 2.0 ** -14)
    weights = (selected / denom).astype(np.float32)
    return ids, weights


def expert_output(model: P.Model, layer: int, expert: int,
                  n_expert: int, xq: np.ndarray):
    gt, gr = role_bytes(model, layer, "gate", expert, n_expert)
    ut, ur = role_bytes(model, layer, "up", expert, n_expert)
    dt, dr = role_bytes(model, layer, "down", expert, n_expert)

    gate = dequant(gr, gt.type_name, (640, 2560))
    up = dequant(ur, ut.type_name, (640, 2560))
    down = dequant(dr, dt.type_name, (2560, 640))

    g = (gate @ xq).astype(np.float32)
    u = (up @ xq).astype(np.float32)

    # Keep the standalone product as f32 before q8_1, like the production path.
    h = ((g / (np.float32(1.0) + np.exp(-g).astype(np.float32))) * u).astype(np.float32)
    hq = q8_quant_dequant(h)
    y = (down @ hq).astype(np.float32)
    return gt, dt, gr, ur, dr, y


def write_bundle(path: Path, layer: int, ids, weights, records, vram_count: int):
    with path.open("wb") as f:
        f.write(b"JRVKRT1\0")
        f.write(struct.pack("<IIIII", 1, layer, len(ids), 2560, 640))
        for rank, (eid, weight, rec) in enumerate(zip(ids, weights, records)):
            gt, dt, gate, up, down = rec
            tier = 1 if rank < vram_count else 0
            f.write(struct.pack(
                "<iiiIfIII",
                int(eid), int(gt.type_id), int(dt.type_id), tier,
                float(weight), len(gate), len(up), len(down)))
            f.write(gate.tobytes())
            f.write(up.tobytes())
            f.write(down.tobytes())


def run_layer(exe: Path, model: P.Model, layer: int, x: np.ndarray,
              n_expert: int, k: int, vram_count: int, tmp_base: Path):
    ids, weights = router_weights(model, layer, x, k)
    xq = q8_quant_dequant(x)

    records = []
    ref = np.zeros(2560, dtype=np.float32)
    for rank, (eid, weight) in enumerate(zip(ids, weights)):
        gt, dt, gr, ur, dr, y = expert_output(
            model, layer, int(eid), n_expert, xq)
        records.append((gt, dt, gr, ur, dr))
        term = (y * np.float32(weight)).astype(np.float32)
        ref = term if rank == 0 else (ref + term).astype(np.float32)

    with tempfile.TemporaryDirectory(prefix=f"jr-vk-route-l{layer:02d}-", dir=tmp_base) as td:
        td = Path(td)
        bundle = td / "route.jrvk"
        xp = td / "x.f32"
        yp = td / "y.f32"
        write_bundle(bundle, layer, ids, weights, records, vram_count)
        xp.write_bytes(np.ascontiguousarray(x, dtype="<f4").tobytes())

        env = os.environ.copy()
        env.setdefault("JR_VK_DEVICE", "8086:e211")
        cp = subprocess.run(
            [str(exe), "--bundle", str(bundle), "--x", str(xp), "--out", str(yp)],
            env=env, text=True, capture_output=True)
        if cp.returncode:
            print(cp.stdout)
            print(cp.stderr, file=sys.stderr)
            raise RuntimeError(f"Vulkan routed runner failed at layer {layer}")

        got = np.fromfile(yp, dtype="<f4")
        if got.size != 2560:
            raise RuntimeError(f"layer {layer}: returned {got.size} values")

        diff = got.astype(np.float64) - ref.astype(np.float64)
        max_abs = float(np.max(np.abs(diff)))
        ref_peak = float(np.max(np.abs(ref.astype(np.float64))))
        peak_rel = max_abs / max(1.0, ref_peak)
        rmse = float(np.sqrt(np.mean(diff * diff)))
        ref_rms = float(np.sqrt(np.mean(ref.astype(np.float64) ** 2)))
        nrmse = rmse / max(ref_rms, 1e-12)
        finite = bool(np.isfinite(got).all())
        ok = finite and nrmse <= 2e-3 and peak_rel <= 5e-3

        route = " ".join(
            f"{int(e)}:{float(w):.4f}{'[V]' if r < vram_count else '[R]'}"
            for r, (e, w) in enumerate(zip(ids, weights)))

        print(
            f"layer {layer:2d}  top{k}  "
            f"VRAM/RAM={vram_count}/{k-vram_count}  "
            f"max_abs={max_abs:.3e}  peak_rel={peak_rel:.3e}  "
            f"nrmse={nrmse:.3e}  {'PASS' if ok else 'FAIL'}")
        print("    route:", route)

        if not ok:
            print(cp.stdout)
            worst = np.argsort(np.abs(diff))[-8:][::-1]
            for i in worst:
                print(
                    f"    y[{i:4d}] gpu={got[i]: .8e} "
                    f"ref={ref[i]: .8e} diff={diff[i]: .3e}")
            raise RuntimeError(f"V4 routed parity failed layer {layer}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gguf",
        default="/data/strata-lab/data/models/swift-IQ3_XXS/"
                "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf")
    ap.add_argument(
        "--exe",
        default=str(ROOT / "build-vulkan" / "jr-vk-routed-expert"))
    ap.add_argument("--all-layers", action="store_true")
    ap.add_argument("--layers", default="0,1,8,28,35,36")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--vram-count", type=int, default=5)
    args = ap.parse_args()

    if args.k < 1 or args.k > 15:
        ap.error("--k must be 1..15")
    if not (0 <= args.vram_count <= args.k):
        ap.error("--vram-count must be between 0 and k")

    gguf = Path(args.gguf)
    exe = Path(args.exe)
    if not gguf.is_file():
        ap.error(f"GGUF not found: {gguf}")
    if not exe.is_file():
        ap.error(f"Vulkan executable not found: {exe}; build first")

    model = P.Model(gguf)
    n_expert = int(model.where["blk.0.ffn_gate_inp.weight"][1].shape[1])

    if args.all_layers:
        layers = list(range(48))
    else:
        layers = [int(x) for x in args.layers.split(",") if x.strip()]

    print("JR-Strata-Vulkan V4 Routed Expert Engine")
    print("  GGUF        :", gguf)
    print("  experts     :", n_expert)
    print("  route       : real layer router -> top-k oracle")
    print("  k           :", args.k)
    print("  tier split  :", args.vram_count, "VRAM /", args.k - args.vram_count, "imported RAM")
    print("  activation  : q8_1")
    print("  execution   : native quant weights -> q8 dot -> SwiGLU/q8 -> down -> weighted combine")
    print("  layers      :", layers)
    print()

    rng = np.random.default_rng(0x4A525634)
    base_x = (rng.standard_normal(2560).astype(np.float32) * np.float32(0.05))

    tmp_base = Path(os.environ.get("JR_VK_TMP", "/data/strata-lab/vulkan-tmp"))
    tmp_base.mkdir(parents=True, exist_ok=True)

    for layer in layers:
        x = (base_x * np.float32(1.0 + (layer % 7) * 0.01)).astype(np.float32)
        run_layer(exe, model, layer, x, n_expert, args.k, args.vram_count, tmp_base)

    print()
    print(f"JR-VK V4 routed expert: PASS ({len(layers)}/{len(layers)} layers)")
    if not args.all_layers:
        print("Strong gate:")
        print("  python3 tools/vulkan_routed_expert_parity.py --all-layers")


if __name__ == "__main__":
    main()
