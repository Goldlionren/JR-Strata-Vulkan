#!/usr/bin/env python3
"""Exercise a running V11 server: real generation, reset, SSE, busy and errors."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url', default='http://127.0.0.1:8080')
    ap.add_argument('--api-key', default=os.environ.get('STRATA_API_KEY'))
    ap.add_argument('--reference-text', type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--long-tokens', type=int, default=3000)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    headers = {'Content-Type': 'application/json'}
    if args.api_key:
        headers['Authorization'] = 'Bearer ' + args.api_key

    def call(path, body=None, authorized=True, extra_headers=None):
        hs = dict(headers if authorized else {'Content-Type': 'application/json'})
        if extra_headers:
            hs.update(extra_headers)
        request = Request(args.url + path, data=None if body is None else json.dumps(body).encode(), headers=hs)
        try:
            with urlopen(request, timeout=600) as response:
                return response.status, json.loads(response.read())
        except HTTPError as exc:
            return exc.code, json.loads(exc.read())

    report = {}
    status, health = call('/health')
    assert status == 200 and health['device'] == '8086:e211' and health['mtp']['spec'] == 4
    status, models = call('/v1/models')
    assert status == 200
    model = models['data'][0]['id']
    if args.api_key:
        assert call('/health', authorized=False)[0] == 401
        assert call('/health', extra_headers={'Authorization': 'Bearer incorrect-é'})[0] == 401
        report['authentication'] = 'PASS'
    base = {'model': model, 'messages': [{'role': 'user', 'content': 'Explain memory ordering in two sentences.'}],
            'temperature': 0, 'max_tokens': 16}
    for changed in ({'n': True}, {'temperature': False}, {'temperature': 1}, {'max_tokens': 8192}, {'stream': 'true'},
                    {'messages': [{'role': 'user', 'content': [{'type': 'image_url'}]}]}):
        assert call('/v1/chat/completions', {**base, **changed})[0] == 400
    assert call('/unknown')[0] == 404
    report['validation'] = 'PASS'
    a = call('/v1/chat/completions', base)
    b = call('/v1/chat/completions', base)
    assert a[0] == b[0] == 200
    assert a[1]['choices'] == b[1]['choices'] and a[1]['usage'] == b[1]['usage']
    assert a[1]['usage']['completion_tokens'] == 16
    report['deterministic_reset'] = 'PASS'
    request = Request(args.url + '/v1/chat/completions', data=json.dumps({**base, 'stream': True}).encode(), headers=headers)
    events = []
    with urlopen(request, timeout=600) as response:
        for raw in response:
            line = raw.decode().strip()
            if line.startswith('data: '):
                value = line[6:]
                if value == '[DONE]':
                    events.append('DONE')
                else:
                    events.append(json.loads(value))
    assert events[-1] == 'DONE'
    streamed = ''.join(e['choices'][0]['delta'].get('content', '') for e in events[:-1])
    assert streamed == a[1]['choices'][0]['message']['content']
    assert events[-2]['choices'][0]['finish_reason'] == 'length'
    report['streaming'] = 'PASS'
    status, one = call('/v1/chat/completions', {**base, 'max_tokens': 1})
    assert status == 200 and one['usage']['completion_tokens'] == 1
    assert one['strata_stats']['mtp']['proposed'] == 0
    assert one['strata_stats']['mtp']['minimum_headroom_gib'] >= 0.5
    report['one_token_metrics'] = 'PASS'
    request = Request(args.url + '/v1/chat/completions', data=json.dumps({**base, 'stream': True, 'max_tokens': 64}).encode(), headers=headers)
    with urlopen(request, timeout=600) as response:
        assert response.readline().startswith(b'data: ')
    for _ in range(200):
        if not call('/health')[1]['busy']:
            break
        time.sleep(.05)
    else:
        raise AssertionError('Disconnected stream did not release runtime')
    recovered = call('/v1/chat/completions', base)
    assert recovered[0] == 200 and recovered[1]['choices'] == a[1]['choices']
    report['disconnect_recovery'] = 'PASS'
    long_body = {**base, 'max_tokens': args.long_tokens,
        'messages': [{'role': 'user', 'content': 'Write a technical textbook chapter of at least 6000 words about computer memory and concurrent programming. Explain caches, memory ordering, virtual memory, asynchronous I/O, memory pools, synchronization and failure handling. Use extensive technical details, equations and code examples, and continue through all sections without summarizing early.'}]}
    with ThreadPoolExecutor(max_workers=1) as pool:
        start = time.monotonic()
        future = pool.submit(call, '/v1/chat/completions', long_body)
        for _ in range(200):
            if call('/health')[1]['busy']:
                break
            time.sleep(.05)
        else:
            raise AssertionError('Long request did not acquire runtime')
        assert call('/v1/chat/completions', base)[0] == 429
        status, response = future.result(timeout=600)
        report['long_request_seconds'] = time.monotonic() - start
    assert status == 200 and response['object'] == 'chat.completion'
    assert response['usage']['completion_tokens'] == args.long_tokens
    assert response['choices'][0]['finish_reason'] == 'length'
    args.output.joinpath('completion.json').write_text(json.dumps(response, indent=2) + '\n')
    args.output.joinpath('completion.txt').write_text(response['choices'][0]['message']['content'])
    report['busy_rejection'] = 'PASS'
    report['long_generation'] = 'PASS'
    stats = response['strata_stats']
    assert stats['spec'] == 4 and stats['mtp']['minimum_headroom_gib'] >= 0.5
    if args.reference_text:
        assert response['choices'][0]['message']['content'] == args.reference_text.read_text(), 'API changed frozen greedy text'
        report['reference_exact'] = 'PASS'
    report['minimum_tok_s'] = 28
    report['speed_pass'] = stats['tok_s'] >= 28
    report['mtp'] = stats['mtp']
    report['tok_s'] = stats['tok_s']
    report['completion_tokens'] = args.long_tokens
    assert call('/health')[1]['busy'] is False
    report['functional_pass'] = True
    args.output.joinpath('api-check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    return 0 if report['speed_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
