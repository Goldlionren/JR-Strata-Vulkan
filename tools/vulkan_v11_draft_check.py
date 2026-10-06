#!/usr/bin/env python3
"""CPU-only numerical oracle for --draft-check's first actual Vulkan proposal."""
from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import ctypes
import json
from pathlib import Path
import numpy as np
from _paths import add_gguf_py
add_gguf_py()
import gguf

E,HC,FF=2560,10240,640

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dump',type=Path,default=Path('/tmp/v11-draft-check'))
    ap.add_argument('--mtp',type=Path,default=Path('/data/strata-lab/data/mtp/rt'))
    ap.add_argument('--native',type=Path,default=Path('/data/strata-lab/data/models/swift-IQ3_XXS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf'))
    ap.add_argument('--ggml',type=Path,default=Path('build-vulkan/oracle/bin/libggml-base.so'))
    ap.add_argument('--head',action='store_true')
    a=ap.parse_args()
    layout=json.loads((a.dump/'layout.json').read_text())
    gpu=np.fromfile(a.dump/'draft.f32',np.float32);base=np.fromfile(a.dump/'base.f32',np.float32)
    def captured(name,n):return gpu[layout[name]:layout[name]+n]
    dense=np.memmap(a.mtp/'dense.bin',mode='r',dtype=np.uint8);weights={};kinds={}
    for line in (a.mtp/'dense.txt').read_text().splitlines():
        name,kind,rows,cols,off,size=line.split();rows,cols,off,size=map(int,(rows,cols,off,size));raw=np.array(dense[off:off+size])
        if kind=='f32':w=raw.view('<f4').reshape(rows,cols)
        elif kind=='bf16':w=(raw.view('<u2').astype(np.uint32)<<16).view(np.float32).reshape(rows,cols)
        else:
            b=raw.reshape(-1,34);scale=b[:,:2].copy().view('<f2').astype(np.float32);w=(scale*b[:,2:].view(np.int8)).reshape(rows,cols)
        weights[name]=w;kinds[name]=kind
    def W(name):return weights[name]
    def norm(x,gamma):return x*np.float32(1/np.sqrt(np.mean(x*x,dtype=np.float32)+np.float32(1e-6)))*gamma
    def q8(x):
        b=x.reshape(-1,32);d=np.max(abs(b),axis=1)/np.float32(127);q=np.zeros_like(b)
        nonzero=d!=0;v=b[nonzero]/d[nonzero,None];q[nonzero]=np.clip(np.sign(v)*np.floor(abs(v)+np.float32(.5)),-127,127)
        return (q*d.astype(np.float16).astype(np.float32)[:,None]).reshape(x.shape)
    def proj(name,x):return W(name)@(q8(x) if kinds[name]=='q8_0' else x)
    def sigmoid(x):return np.float32(1)/(np.float32(1)+np.exp(-np.clip(x,-80,80)))
    def silu(x):return x*sigmoid(x)
    def hc(R,prefix):
        gamma=W(prefix+'.hc_norm.weight').reshape(4,E);xn=np.stack([norm(R[c],gamma[c]) for c in range(4)])
        lo=silu(W(prefix+'.input_mix_weight_down.weight')@xn.ravel()*np.float32(.25))
        gate=sigmoid(W(prefix+'.input_mix_weight_up.weight')@lo).reshape(4,E)
        mixed=np.sum(xn*gate,axis=0,dtype=np.float32)*np.float32(.25)
        inj=weights.get(prefix+'.block_inject_weight.weight')
        return mixed,None if inj is None else inj@xn.ravel()
    def rope(x,cell):
        v=x.copy();ang=np.float32(cell)*np.power(np.float32(1e7),-np.arange(0,64,2,dtype=np.float32)/np.float32(64));co=np.cos(ang);si=np.sin(ang)
        v[...,:32]=x[...,:32]*co-x[...,32:64]*si;v[...,32:64]=x[...,:32]*si+x[...,32:64]*co;return v
    results={}
    def compare(name,ref,actual):
        ref=np.asarray(ref).ravel().astype(np.float64);actual=np.asarray(actual).ravel().astype(np.float64);diff=ref-actual
        rmse=float(np.sqrt(np.mean(diff*diff)));rms=float(np.sqrt(np.mean(ref*ref)));cos=float(np.dot(ref,actual)/(np.linalg.norm(ref)*np.linalg.norm(actual)))
        row={'rmse':rmse,'relative_rmse':rmse/max(rms,1e-30),'max_abs':float(np.max(abs(diff))),'cosine':cos};results[name]=row;print(name,json.dumps(row),flush=True)
    # Independent pinned GGML dequantizer: CPU library only, no GPU backend.
    lib=ctypes.CDLL(str(a.ggml.resolve()));reader=gguf.GGUFReader(str(a.native));tens={t.name:t for t in reader.tensors};emb=tens['token_embd.weight'];assert int(emb.tensor_type)==21
    fn=lib.dequantize_row_iq3_s;fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64]
    embedding=np.empty(E,np.float32);fn(emb.data.ctypes.data+layout['token']*(E//256*110),embedding.ctypes.data,E)
    en=norm(embedding,W('pre_fc_norm_embedding.weight').ravel());compare('embedding_norm',en,captured('pleemb',E))
    e2=proj('fc_embedding.weight',en);compare('fc_embedding',e2,captured('pleval',E))
    R0=base[layout['R']:layout['R']+HC];hn=norm(R0,W('pre_fc_norm_hidden.weight').ravel());compare('hidden_norm',hn,captured('plenorm',HC))
    R=np.stack([proj('fc_hidden.weight',h) for h in hn.reshape(4,E)])+e2
    x,inj=hc(R,'attn_hyper_connection')
    qfull=proj('self_attn.q_proj.weight',x).reshape(24,512);compare('q_projection',qfull,captured('qfull',12288))
    k=proj('self_attn.k_proj.weight',x).reshape(2,256);k=rope(np.stack([norm(z,W('self_attn.k_norm.weight').ravel()) for z in k]),layout['cell'])
    v=proj('self_attn.v_proj.weight',x).reshape(2,256);compare('key',k,captured('kcur',512));compare('value',v,captured('vcur',512))
    q=rope(np.stack([norm(z[:256],W('self_attn.q_norm.weight').ravel()) for z in qfull]),layout['cell']);compare('query',q,captured('q',6144))
    kv=np.fromfile(a.dump/'kv.bin','<f2').reshape(2,layout['context'],2,256).astype(np.float32);n=layout['cell']+1
    attn=np.empty((24,256),np.float32)
    for h in range(24):
        scores=kv[0,:n,h//12]@q[h]/np.float32(16);prob=np.exp(scores-np.max(scores));prob/=np.sum(prob,dtype=np.float32);attn[h]=prob@kv[1,:n,h//12]
    compare('attention',attn,captured('attn',6144))
    gated=(attn*sigmoid(qfull[:,256:])).astype(np.float16).astype(np.float32).ravel();compare('attention_gate',gated,captured('y',6144))
    bo=proj('self_attn.o_proj.weight',gated);R+=2*sigmoid(inj*np.float32(.25))[:,None]*bo
    x,inj=hc(R,'mlp_hyper_connection');router=proj('mlp.gate.weight',x);compare('router_logits',router,captured('scores',512))
    order=np.lexsort((np.arange(512),-router))[:10];prob=np.exp(router[order]-np.max(router));prob/=np.sum(prob,dtype=np.float32)
    experts=np.memmap(a.mtp/'experts.bin',dtype=np.uint8,mode='r');parts=[]
    for e in order:
        blob=np.array(experts[e*1382400:(e+1)*1382400]);gu=blob[:819200].reshape(1280,40,16);dn=blob[819200:1228800].reshape(2560,10,16)
        gs=blob[1228800:1331200].view('<f2').astype(np.float32).reshape(1280,40);ds=blob[1331200:].view('<f2').astype(np.float32).reshape(2560,10)
        def unpack(codes,sc):
            symbols=((codes[...,None]>>np.arange(0,8,2,dtype=np.uint8))&3).astype(np.int8)-1
            return (symbols.reshape(codes.shape[0],codes.shape[1],64)*sc[...,None]).reshape(codes.shape[0],-1)
        g=unpack(gu[0::2],gs[0::2])@q8(x);u=unpack(gu[1::2],gs[1::2])@q8(x);parts.append(unpack(dn,ds)@q8(silu(g)*u))
    moe=np.sum(np.stack(parts)*prob[:,None],axis=0,dtype=np.float32);compare('routed_experts',moe,captured('moe',E))
    shg=proj('mlp.shared_expert.gate_proj.weight',x);shu=proj('mlp.shared_expert.up_proj.weight',x);compare('shared_gate',shg,captured('shg',FF));compare('shared_up',shu,captured('shu',FF))
    sho=proj('mlp.shared_expert.down_proj.weight',silu(shg)*shu);compare('shared_down',sho,captured('sho',E));shgate=proj('mlp.shared_expert_gate.weight',x)[0]
    block=moe+sho*sigmoid(shgate);compare('ffn_output',block,captured('block',E));R+=2*sigmoid(inj*np.float32(.25))[:,None]*block;compare('final_residual',R,captured('R',HC))
    mixed,_=hc(R,'hyper_connection_mixer');compare('final_mixed',mixed,captured('mixed',E))
    if a.head:
        head=tens['output.weight'];assert int(head.tensor_type)==13;vocab=np.fromfile(a.mtp/'draft_vocab.bin','<i4');weights_head=np.empty((len(vocab),E),np.float32)
        fn=lib.dequantize_row_q5_K;fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64]
        for i,id in enumerate(vocab):fn(head.data.ctypes.data+int(id)*(E//256*176),weights_head[i].ctypes.data,E)
        logits=weights_head@q8(mixed);head_offset=layout['pleemb']-248320;compare('draft_logits',logits,gpu[head_offset:head_offset+len(vocab)])
        results['cpu_top1']=int(vocab[np.argmax(logits)]);results['vulkan_top1']=int(vocab[np.argmax(gpu[head_offset:head_offset+len(vocab)])]);print('CPU/Vulkan top1',results['cpu_top1'],results['vulkan_top1'],flush=True)
    (a.dump/'cpu-check.json').write_text(json.dumps(results,indent=2)+'\n')

if __name__=='__main__':main()
