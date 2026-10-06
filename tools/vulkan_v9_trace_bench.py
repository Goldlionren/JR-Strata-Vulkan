#!/usr/bin/env python3
from __future__ import annotations
import argparse, math, os, re, struct, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(HERE))
import iq_pack as P
import vulkan_routed_expert_parity as V4
PLAN_MAGIC=b"JRVKC9\0\0"

def read_plan(path:Path):
    b=path.read_bytes()
    if len(b)<44 or b[:8]!=PLAN_MAGIC: raise SystemExit(f"{path}: not a V9 residency plan")
    ver,nl,ne,nr,flags,budget,used=struct.unpack_from("<IIIIIQQ",b,8)
    if ver!=1: raise SystemExit(f"{path}: unsupported plan version {ver}")
    off=8+struct.calcsize("<IIIIIQQ"); n=nl*ne
    if len(b)<off+4*n: raise SystemExit(f"{path}: truncated residency table")
    slots=np.frombuffer(b,dtype="<i4",count=n,offset=off).reshape(nl,ne)
    return nl,ne,nr,budget,used,slots

def read_trace(path:Path):
    b=path.read_bytes(); off=0; rec=[]
    while off+8<=len(b):
        layer,k=struct.unpack_from("<ii",b,off); off+=8
        if k<=0 or k>64 or off+8*k>len(b): break
        ids=struct.unpack_from(f"<{k}i",b,off); off+=4*k
        weights=struct.unpack_from(f"<{k}f",b,off); off+=4*k
        rec.append((layer,ids,weights))
    return rec,off,len(b)

def index_by_occurrence(rec,nl):
    counts=[0]*nl; maps=[{} for _ in range(nl)]
    for l,ids,w in rec:
        if 0<=l<nl:
            p=counts[l]; counts[l]+=1; maps[l][p]=(ids,w)
    return counts,maps

def write_bundle(path,model,layer,ids,weights,n_expert):
    with path.open("wb") as f:
        f.write(b"JRVKRT1\0"); f.write(struct.pack("<IIIII",1,layer,len(ids),2560,640))
        for eid,weight in zip(ids,weights):
            gt,gate=V4.role_bytes(model,layer,"gate",int(eid),n_expert)
            _,up=V4.role_bytes(model,layer,"up",int(eid),n_expert)
            dt,down=V4.role_bytes(model,layer,"down",int(eid),n_expert)
            f.write(struct.pack("<iiiIfIII",int(eid),int(gt.type_id),int(dt.type_id),0,float(weight),len(gate),len(up),len(down)))
            f.write(gate.tobytes()); f.write(up.tobytes()); f.write(down.tobytes())

def parse_stats(s):
    m=re.search(r"GPU avg\s*:\s*([0-9.eE+-]+)\s*ms",s)
    b=re.search(r"effective routed weight throughput:\s*([0-9.eE+-]+)\s*GiB/s",s)
    return (float(m.group(1)) if m else None,float(b.group(1)) if b else None)

def evenly(start,count,want):
    if want<=0 or want>=count: return list(range(start,start+count))
    if want==1: return [start+count-1]
    vals=[]; seen=set()
    for v in np.linspace(start,start+count-1,num=want):
        p=int(round(float(v)))
        if p not in seen: vals.append(p); seen.add(p)
    for p in range(start,start+count):
        if len(vals)>=want: break
        if p not in seen: vals.append(p); seen.add(p)
    return sorted(vals)

def pct(xs,q): return float(np.percentile(np.asarray(xs,dtype=np.float64),q)) if xs else float("nan")

def main():
    ap=argparse.ArgumentParser(description="Replay real Strata routing traces through JR-Strata-Vulkan V9")
    ap.add_argument("trace")
    ap.add_argument("--plan",required=True)
    ap.add_argument("--decode-positions",type=int,required=True)
    ap.add_argument("--sample-tokens",type=int,default=16,help="evenly sampled decode positions; 0=all")
    ap.add_argument("--gguf",default="/data/strata-lab/data/models/swift-IQ3_XXS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf")
    ap.add_argument("--exe",default=str(ROOT/"build-vulkan"/"jr-vk-routed-engine-v9"))
    ap.add_argument("--warmup",type=int,default=3); ap.add_argument("--iters",type=int,default=20)
    ap.add_argument("--truncate-incomplete-tail",action="store_true"); ap.add_argument("--print-each",action="store_true")
    a=ap.parse_args()
    trace=Path(a.trace); plan=Path(a.plan); exe=Path(a.exe)
    for p in (trace,plan,exe):
        if not p.exists(): raise SystemExit(f"missing: {p}")
    nl,ne,nr,budget,used,slots=read_plan(plan)
    rec,parsed,total=read_trace(trace)
    counts,maps=index_by_occurrence(rec,nl); lo,hi=min(counts),max(counts)
    if lo!=hi and not a.truncate_incomplete_tail: raise SystemExit(f"layer counts differ: min={lo} max={hi}; use --truncate-incomplete-tail")
    if hi-lo>32: raise SystemExit(f"layer count skew {hi-lo} too large; refusing")
    if a.decode_positions<=0 or a.decode_positions>lo: raise SystemExit(f"invalid --decode-positions {a.decode_positions}; common positions={lo}")
    start=lo-a.decode_positions; positions=evenly(start,a.decode_positions,a.sample_tokens)
    model=P.Model(Path(a.gguf)); env=os.environ.copy(); env.setdefault("JR_VK_DEVICE","8086:e211")
    tmpbase=Path(os.environ.get("JR_VK_TMP","/data/strata-lab/vulkan-tmp")); tmpbase.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(0x4A525654)
    all_ms=[]; all_bw=[]; hits=routes=0
    l_ms=[[] for _ in range(nl)]; l_h=[0]*nl; l_r=[0]*nl
    print("JR-Strata-Vulkan V9 REAL-TRACE replay benchmark")
    print(f"  trace          : {trace}")
    print(f"  records        : {len(rec)}  parsed={parsed}/{total} bytes")
    print(f"  per-layer pos  : min={lo} max={hi}")
    print(f"  decode suffix  : {a.decode_positions} positions, start={start}")
    print(f"  sampled tokens : {len(positions)} -> {positions}")
    print(f"  plan resident  : {nr}/{nl*ne} experts")
    print(f"  plan used      : {used/(1<<30):.3f} GiB / {budget/(1<<30):.3f} GiB")
    print(f"  warmup/iters   : {a.warmup}/{a.iters}")
    print(f"  device         : {env['JR_VK_DEVICE']}\n")
    with tempfile.TemporaryDirectory(prefix="jr-vk-v9-trace-",dir=tmpbase) as td:
        td=Path(td)
        for si,pos in enumerate(positions):
            x=(rng.standard_normal(2560).astype(np.float32)*np.float32(0.05)); xp=td/f"p{pos}.x.f32"; xp.write_bytes(np.ascontiguousarray(x,dtype="<f4").tobytes())
            tok_ms=tok_h=tok_r=0
            for l in range(nl):
                if pos not in maps[l]: raise SystemExit(f"missing layer {l} occurrence {pos}")
                ids,w=maps[l][pos]
                if any(e<0 or e>=ne for e in ids): raise SystemExit(f"invalid expert id at layer {l}, pos {pos}")
                if not all(math.isfinite(float(v)) for v in w): raise SystemExit(f"non-finite weight at layer {l}, pos {pos}")
                h=sum(1 for e in ids if slots[l,int(e)]>=0); k=len(ids)
                bundle=td/f"p{pos}-l{l}.jrvk"; yp=td/f"p{pos}-l{l}.y.f32"; write_bundle(bundle,model,l,ids,w,ne)
                cp=subprocess.run([str(exe),"--bundle",str(bundle),"--x",str(xp),"--out",str(yp),"--residency",str(plan),"--warmup",str(a.warmup),"--iters",str(a.iters)],env=env,text=True,capture_output=True)
                if cp.returncode: print(cp.stdout); print(cp.stderr,file=sys.stderr); raise SystemExit(cp.returncode)
                ms,bw=parse_stats(cp.stdout)
                if ms is None or bw is None: print(cp.stdout); raise SystemExit("could not parse V9 timing")
                all_ms.append(ms); all_bw.append(bw); hits+=h; routes+=k; l_ms[l].append(ms); l_h[l]+=h; l_r[l]+=k; tok_ms+=ms; tok_h+=h; tok_r+=k
                if a.print_each: print(f"pos {pos:6d} layer {l:02d} hits={h}/{k} avg={ms:.4f} ms bw={bw:.3f}")
            print(f"token-sample {si:02d} pos={pos:6d} hit={tok_h/tok_r*100:6.2f}% MoE48={tok_ms:8.4f} ms ceiling={1000.0/tok_ms:6.2f} tok/s")
    lavg=[float(np.mean(v)) for v in l_ms]; budget_ms=float(sum(lavg)); npos=len(positions)
    print("\n"+"="*88); print("V9 REAL-TRACE SUMMARY")
    print(f"sampled decode tokens          : {npos}")
    print(f"measured layer executions      : {len(all_ms)}")
    print(f"real-trace routed experts      : {routes}")
    print(f"VRAM hits                      : {hits}")
    print(f"real-trace hit rate            : {hits/routes*100:.3f}%")
    print(f"avg VRAM experts / layer       : {hits/(npos*nl):.3f}")
    print(f"avg RAM experts / layer        : {(routes-hits)/(npos*nl):.3f}")
    print(f"layer latency p50              : {pct(all_ms,50):.4f} ms")
    print(f"layer latency p95              : {pct(all_ms,95):.4f} ms")
    print(f"layer latency mean             : {float(np.mean(all_ms)):.4f} ms")
    print(f"mean routed throughput         : {float(np.mean(all_bw)):.3f} GiB/s")
    print(f"48-layer MoE-only budget       : {budget_ms:.4f} ms/token")
    print(f"48-layer MoE-only ceiling      : {1000.0/budget_ms:.2f} tok/s")
    print("\nPER-LAYER\nlayer  hit-rate   avg-ms")
    for l in range(nl): print(f"{l:5d}  {l_h[l]/l_r[l]*100:7.2f}%  {lavg[l]:7.4f}")

if __name__=="__main__": main()
