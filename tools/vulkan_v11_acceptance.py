#!/usr/bin/env python3
"""Run the required V11 decode lengths in fresh processes; record, never infer, PASS.

Use the ordinary CLI/server config. Reports and generated text go to a caller's
output directory. This is a milestone gate, not a general benchmark framework.
"""
import argparse
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time

from vulkan_v11_correctness import PROMPT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runtime', type=Path, default=Path('build-vulkan/jr-vk-runtime-v11'))
    ap.add_argument('--config', type=Path, default=Path('strata-vulkan-swift-iq3_xxs-mtp.json'))
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--lengths', default='256,1024,3000,3000,3000')
    ap.add_argument('--minimum-tok-s', type=float, default=28.)

    ap.add_argument('--prompt', default=PROMPT)
    ap.add_argument('--reference', type=Path, help='frozen target-only JSON token sequence for exact parity')
    args = ap.parse_args()
    if os.environ.get('JR_VK_DEVICE') != '8086:e211':
        ap.error('JR_VK_DEVICE=8086:e211 is required')
    args.output.mkdir(parents=True, exist_ok=True)
    reference = json.loads(args.reference.read_text())['token_ids'] if args.reference else None
    results = []
    for index, length in enumerate(map(int, args.lengths.split(','))):
        prefix = args.output / f'{index+1}-{length}'
        report = prefix.with_suffix('.json')
        cmd = [str(args.runtime.resolve()), '--config', str(args.config.resolve()),
               '--prompt', args.prompt, '--max-new', str(length), '--greedy', '--checks',
               '--report', str(report)]
        start = time.monotonic()
        with prefix.with_suffix('.txt').open('w') as text, prefix.with_suffix('.log').open('w') as log:
            child = subprocess.Popen(cmd, stdout=text, stderr=log)
            try:
                code = child.wait(timeout=3600)
            except BaseException:
                child.terminate()
                child.wait()
                raise
        row = {'length': length, 'returncode': code, 'wall_seconds': time.monotonic()-start}
        if code == 0 and report.exists():
            data = json.loads(report.read_text())
            row.update(data)
            ids = data['token_ids']
            longest, current = 0, 0
            previous = None
            for token in ids:
                current = current + 1 if token == previous else 1
                longest = max(longest, current)
                previous = token
            row['longest_token_run'] = longest
            row['reference_exact'] = reference is None or ids == reference[:length]
            heaps = [float(v) for v in re.findall(r'VRAM (?:MTP ready|MTP position \d+|position \d+): ([0-9.]+) GiB', prefix.with_suffix('.log').read_text())]
            row['vram_growth_gib'] = max(heaps)-min(heaps) if heaps else None
            row['bounded_vram'] = bool(heaps) and row['vram_growth_gib'] <= 1/1024
            row['stability_pass'] = len(ids) == length and longest < 64 and row['reference_exact'] and row['bounded_vram']
            mtp = data.get('mtp', {})
            row['mtp_pass'] = data.get('spec') == 4 and mtp.get('proposed', 0) > 0 and mtp.get('verification_windows', 0) > 0 and mtp.get('minimum_headroom_gib', 0) >= 0.5
            peers = [r['token_ids'] for r in results if r.get('length') == length and r.get('token_ids')]
            row['reproducibility_pass'] = all(ids == previous for previous in peers)
            row['same_proposal_pattern'] = all(mtp.get('proposal_hash') == r.get('mtp', {}).get('proposal_hash') for r in results if r.get('length') == length and r.get('mtp'))
            row['speed_pass'] = data['tok_s'] >= args.minimum_tok_s
            row['functional_pass'] = row['stability_pass'] and row['mtp_pass'] and row['reproducibility_pass']
            row['pass'] = row['functional_pass'] and row['speed_pass']
        else:
            row['pass'] = False
        results.append(row)
        (args.output / 'acceptance.json').write_text(json.dumps(results, indent=2) + '\n')
        print(f'{length}: {"PASS" if row["pass"] else "FAIL"}; '
              f'{row.get("completion_tokens", 0)} tokens; {row.get("tok_s", 0):.2f} tok/s', flush=True)
        if code:
            break
    passed = all(row['pass'] for row in results) and len(results) == len(args.lengths.split(','))
    # This gate covers generation only; parity, bounded memory, device isolation,
    # and an actual API response must also be verified before claiming V11 PASS.
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
