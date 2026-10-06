#!/usr/bin/env python3
"""Real B60 forced accept/reject and exact committed-state continuation gate."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from vulkan_runtime_v11 import Worker, parser, tokenizer, messages_to_ids

PROMPT = ('Write a technical textbook chapter of at least 6000 words about computer memory and concurrent programming. '
          'Explain caches, memory ordering, virtual memory, asynchronous I/O, memory pools, synchronization and failure handling. '
          'Use extensive technical details, equations and code examples, and continue through all sections without summarizing early.')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',type=Path,default=Path('strata-vulkan-swift-iq3_xxs-mtp.json'))
    ap.add_argument('--output',type=Path,default=Path('/tmp/v11-correctness.json'))
    ap.add_argument('--tokens',type=int,default=64)
    ap.add_argument('--benchmark',type=int,default=0)
    ap.add_argument('--profile',action='store_true')
    ap.add_argument('--next-benchmark',type=int,default=0)
    args=ap.parse_args()
    p=parser();p.set_defaults(**json.loads(args.config.read_text()))
    a=p.parse_args(['--spec','4','--spec-min-p','0','--checks']+(['--profile'] if args.profile else []))
    a.pack=Path(a.pack);a.native=Path(a.native);a.residency=Path(a.residency)
    tk,template=tokenizer(a.pack);ids=messages_to_ids(tk,template,[{'role':'user','content':PROMPT}])
    w=Worker(a);results={}
    def prefill():
        assert w.command('RESET')=='OK'
        next_id=0
        for id in ids:next_id=w.token(id)[0]
        return next_id
    def check_sequence(name,override=False,reject=-1):
        current=prefill();assert w.command(f'REJECT {reject}')=='OK'
        got=[current];metrics={};windows=[]
        while len(got)<args.tokens:
            if override:
                proposed=expected[len(got):min(len(got)+4,len(expected))]
                assert w.command('OVERRIDE '+' '.join(map(str,proposed)))=='OK'
            tokens,elapsed,metrics=w.window(got[-1],args.tokens-len(got))
            windows.append(len(tokens));got.extend(tokens)
            if got!=expected[:len(got)]:
                mismatch=next(i for i,(x,y) in enumerate(zip(got,expected)) if x!=y)
                raise AssertionError(f'{name}: first mismatch {mismatch}: {got[mismatch]} != {expected[mismatch]}')
        # Target-only continuation after the chosen prefix must agree too.
        current=got[-1];tail=[]
        for _ in range(16):current=w.token(current)[0];tail.append(current)
        assert tail==continuation,f'{name}: target-only continuation changed after rollback'
        results[name]={'exact_ids':True,'exact_continuation':True,'window_sizes':windows,'mtp':metrics}
        print(f'{name}: exact {len(got)} IDs + 16 target-only continuation',flush=True)
    try:
        current=prefill();expected=[current]
        for _ in range(args.tokens+15):current=w.token(current)[0];expected.append(current)
        continuation=expected[args.tokens:];expected=expected[:args.tokens]
        frozen=Path('/tmp/v10-final-gate/2-3000.json')
        if frozen.exists():
            assert expected==json.loads(frozen.read_text())['token_ids'][:args.tokens],'target-only V11 changed the frozen V10 IDs'
            results['frozen_v10']={'exact_ids':True}
        check_sequence('real_mtp')
        check_sequence('real_mtp_repeat')
        assert results['real_mtp']['mtp'].get('proposal_hash') == results['real_mtp_repeat']['mtp'].get('proposal_hash')
        assert results['real_mtp']['window_sizes'] == results['real_mtp_repeat']['window_sizes']
        check_sequence('forced_accept',override=True)
        assert results['forced_accept']['mtp']['accepted']==results['forced_accept']['mtp']['proposed']
        for index in range(4):
            check_sequence(f'forced_reject_{index}',override=True,reject=index)
            assert results[f'forced_reject_{index}']['mtp']['rollback_count']>0
        if args.benchmark:
            assert w.command('REJECT -1')=='OK'
            assert w.command('CONFIDENCE 0.5')=='OK'
            w.tk=tk
            text=''.join(w.generate(ids,args.benchmark))
            report={**w.last_stats,'token_ids':w.generated}
            args.output.with_name(args.output.stem+'-benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
            args.output.with_name(args.output.stem+'-benchmark.txt').write_text(text)
            if frozen.exists():
                assert w.generated==json.loads(frozen.read_text())['token_ids'][:args.benchmark],'benchmark changed frozen greedy sequence'
            print(f'Production benchmark: {args.benchmark} tokens, {report["tok_s"]:.3f} tok/s',flush=True)
            if args.next_benchmark and report['tok_s'] >= 28:
                text=''.join(w.generate(ids,args.next_benchmark))
                next_report={**w.last_stats,'token_ids':w.generated}
                args.output.with_name(args.output.stem+f'-{args.next_benchmark}.json').write_text(json.dumps(next_report,indent=2)+'\n')
                args.output.with_name(args.output.stem+f'-{args.next_benchmark}.txt').write_text(text)
                if frozen.exists():
                    assert w.generated == json.loads(frozen.read_text())['token_ids'][:args.next_benchmark], 'extended benchmark changed frozen greedy sequence'
                print(f'Production benchmark: {args.next_benchmark} tokens, {next_report["tok_s"]:.3f} tok/s',flush=True)
        args.output.write_text(json.dumps({'pass':True,'prompt_tokens':len(ids),'completion_tokens':args.tokens,'results':results},indent=2)+'\n')
    finally:w.close()

if __name__=='__main__':main()
