#!/usr/bin/env python3
import argparse, struct
from pathlib import Path
import numpy as np

PLAN_MAGIC=b"JRVKC9\0\0"

def read_plan(p):
    b=Path(p).read_bytes()
    ver,nl,ne,nr,flags,budget,used=struct.unpack_from("<IIIIIQQ",b,8)
    off=8+struct.calcsize("<IIIIIQQ")
    slots=np.frombuffer(b,dtype="<i4",count=nl*ne,offset=off).reshape(nl,ne)
    return nl,ne,nr,used,slots

def read_trace(p):
    b=Path(p).read_bytes(); off=0; rec=[]
    while off+8<=len(b):
        l,k=struct.unpack_from("<ii",b,off); off+=8
        if k<=0 or k>64: break
        if off+8*k>len(b): break
        ids=struct.unpack_from(f"<{k}i",b,off); off+=4*k
        off+=4*k
        rec.append([l,ids,-1])
    return rec

def score(rec,slots,nl,ne):
    h=t=0; lh=[0]*nl; lt=[0]*nl
    for l,ids,_ in rec:
        if 0<=l<nl:
            for e in ids:
                if 0<=e<ne:
                    t+=1; lt[l]+=1
                    if slots[l,e]>=0:
                        h+=1; lh[l]+=1
    return h,t,lh,lt

def show(title,rec,slots,nl,ne,pl=True):
    h,t,lh,lt=score(rec,slots,nl,ne)
    r=h/t if t else 0
    print(title); print(f"  records        : {len(rec)}")
    print(f"  routed experts : {t}"); print(f"  hits           : {h}")
    print(f"  hit rate       : {r*100:.3f}%")
    if pl:
        print("  per-layer:")
        for l in range(nl):
            if lt[l]:
                print(f"    {l:02d}: {lh[l]/lt[l]*100:7.3f}%  {lh[l]}/{lt[l]}")
    return r

ap=argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--plan",required=True)
ap.add_argument("--decode-positions",type=int,required=True)
ap.add_argument("--truncate-incomplete-tail",action="store_true")
ap.add_argument("--no-all-per-layer",action="store_true")
a=ap.parse_args()

nl,ne,nr,used,slots=read_plan(a.plan)
rec=read_trace(a.trace)
cnt=[0]*nl
for r in rec:
    l=r[0]
    if 0<=l<nl:
        r[2]=cnt[l]; cnt[l]+=1

print("JR-Strata-Vulkan V9 crash-tolerant routing score")
print(f"  resident       : {nr}/{nl*ne}")
print(f"  cache used     : {used/(1<<30):.3f} GiB")
print()
allr=show("ALL WRITTEN POSITIONS",rec,slots,nl,ne,not a.no_all_per_layer)

lo,hi=min(cnt),max(cnt)
if lo!=hi and not a.truncate_incomplete_tail:
    raise SystemExit(f"layer counts differ: min={lo} max={hi}; rerun with --truncate-incomplete-tail for a crash trace")
if hi-lo>32:
    raise SystemExit(f"layer count skew {hi-lo} too large; refusing")
common=[r for r in rec if r[2] < lo]
dec=a.decode_positions
if dec>lo:
    raise SystemExit(f"decode positions {dec} > common positions {lo}")
split=lo-dec
pre=[r for r in common if r[2] < split]
dec_rec=[r for r in common if r[2] >= split]

print()
if lo!=hi:
    print("CRASH-TAIL TRUNCATION")
    print(f"  per-layer min/max : {lo}/{hi}")
    print(f"  common positions  : {lo}")
    print(f"  suffix records dropped: {len(rec)-len(common)}")
print()
print(f"POSITION SPLIT: prefix={split}, decode={dec}, common={lo}")
if pre:
    print(); show("PREFIX / PROMPT+PREFILL",pre,slots,nl,ne,False)
print(); dr=show("DECODE / ANSWER SUFFIX",dec_rec,slots,nl,ne,True)
print()
print(f"decode delta vs all-written: {(dr-allr)*100:+.3f} pp")
