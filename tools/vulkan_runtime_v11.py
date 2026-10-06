#!/usr/bin/env python3
"""Swift text CLI/API for the persistent V11 Vulkan token engine.

The Python path owns tokenization, chat formatting, and HTTP only. Each model
forward and sampler runs in jr-vk-runtime-v11 on PCI vendor/device 8086:e211.
"""
from __future__ import annotations
import argparse
import codecs
import contextlib
import hmac
import ipaddress
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from strata_tokenizer import Tokenizer

EOS = {248044, 248046}
BUCKETS = ('embedding', 'recurrent_attention', 'router', 'moe', 'ple', 'head_sampler')


def tokenizer(pack: Path):
    root = pack / 'tokenizer'
    vocab = json.loads((root / 'vocab.json').read_text())
    tokens = [''] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    cfg = json.loads((root / 'tokenizer.json').read_text())
    tk = Tokenizer(tokens, (root / 'merges.txt').read_text().splitlines(),
                   json.loads((root / 'token_type.json').read_text()),
                   pre=cfg.get('pre', 'qwen35'), special_ids=cfg.get('special_ids'))
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    def raise_exception(message):
        raise ValueError(message)
    env.globals['raise_exception'] = raise_exception
    return tk, env.from_string((root / 'chat_template.jinja').read_text())


class Worker:
    def __init__(self, args):
        if os.environ.get('JR_VK_DEVICE') != '8086:e211':
            raise ValueError('Set JR_VK_DEVICE=8086:e211; V11 only runs on the B60.')
        self.args = args
        cmd = [str(args.runtime), '--worker', '--pack', str(args.pack),
               '--native', str(args.native), '--ple-gguf', str(args.ple_gguf or args.native),
               '--residency', str(args.residency), '--max-context', str(args.max_context),
               '--kv', 'fp16', '--stats-every', str(args.stats_every)]
        cmd.extend(['--mtp', str(args.mtp), '--spec', str(args.spec), '--spec-min-p', str(args.spec_min_p)])
        if args.force_drafts:
            cmd.extend(['--force-drafts', str(args.force_drafts)])
        if args.force_reject >= 0:
            cmd.extend(['--force-reject', str(args.force_reject)])
        if args.route_profile:
            cmd.extend(['--route-profile', str(args.route_profile)])
        if args.checks:
            cmd.append('--checks')
        if args.ignore_eos:
            cmd.append('--ignore-eos')
        if args.state_checks:
            cmd.append('--state-checks')
        if args.fixed_spec:
            cmd.append('--fixed-spec')
        if args.draft_check:
            cmd.append('--draft-check')
        if args.profile:
            cmd.append('--profile')
        if args.logits_out:
            cmd.extend(['--logits-out', str(args.logits_out)])
        if args.dump_dir:
            Path(args.dump_dir).mkdir(parents=True, exist_ok=True)
            cmd.extend(['--dump-dir', str(args.dump_dir)])
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, bufsize=1)
        try:
            if self._read(600) != 'READY':
                raise RuntimeError('Vulkan worker did not become ready')
        except BaseException:
            self.close()
            raise
        self.generated = []

    def _read(self, timeout=130):
        if not select.select([self.p.stdout], [], [], timeout)[0]:
            raise RuntimeError('Vulkan worker response timed out')
        line = self.p.stdout.readline().strip()
        if not line:
            raise RuntimeError(f'Vulkan worker exited ({self.p.poll()})')
        if line.startswith('ERROR '):
            raise RuntimeError(line[6:])
        return line

    def command(self, text):
        if self.p.poll() is not None:
            raise RuntimeError('Vulkan worker is unavailable')
        self.p.stdin.write(text + '\n')
        self.p.stdin.flush()
        return self._read()

    def token(self, token):
        started = time.perf_counter()
        fields = self.command(f'TOKEN {token}').split()
        wall_ms = (time.perf_counter() - started) * 1000
        if len(fields) != 10 or fields[0] != 'TOKEN':
            raise RuntimeError(f'Invalid worker response: {fields!r}')
        return int(fields[1]), float(fields[2]), wall_ms, list(map(float, fields[4:]))

    def close(self):
        if getattr(self, 'p', None) is None:
            return
        if self.p.poll() is None:
            with contextlib.suppress(OSError, BrokenPipeError):
                self.p.stdin.write('QUIT\n')
                self.p.stdin.flush()
            try:
                self.p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.p.terminate()
                try:
                    self.p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.p.kill()
                    self.p.wait()
        if self.p.stdin:
            self.p.stdin.close()
        if self.p.stdout:
            self.p.stdout.close()

    def window(self, token, limit):
        started = time.perf_counter()
        reply = self.command(f'ROUND {token} {limit}')
        elapsed = (time.perf_counter() - started) * 1000
        if not reply.startswith('WINDOW '):
            raise RuntimeError(f'Invalid verification reply: {reply!r}')
        data = json.loads(reply[7:])
        tokens = data.get('tokens')
        if not isinstance(tokens, list) or not 1 <= len(tokens) <= min(limit, self.args.spec+1 if self.args.spec else 1):
            raise RuntimeError('Invalid committed verification prefix')
        if any(type(t) is not int or not 0 <= t < 248320 for t in tokens):
            raise RuntimeError('Invalid committed token')
        return tokens, elapsed, data['metrics']

    def generate(self, ids, maximum, ignore_eos=False):
        if not ids or maximum < 1 or len(ids) + maximum > self.args.max_context:
            raise ValueError('Prompt plus max_tokens must fit the configured context')
        if self.command('RESET') != 'OK':
            raise RuntimeError('Vulkan state reset failed')
        self.generated = []
        started = time.perf_counter()
        for token in ids:
            result = self.token(token)
        prefill_ms = (time.perf_counter() - started) * 1000
        decode_ms, metrics = 0., {}
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        repeated, pending, finish = 0, [result[0]], 'length'
        last_stats_at = 0
        while len(self.generated) < maximum:
            for token in pending:
                if not ignore_eos and token in EOS:
                    finish = 'stop'
                    break
                repeated = repeated + 1 if self.generated and token == self.generated[-1] else 1
                if repeated >= 64:
                    raise RuntimeError(f'Repeated-token collapse: token {token} repeated {repeated} times')
                self.generated.append(token)
                yield decoder.decode(self.tk.token_bytes(token))
            if finish == 'stop' or len(self.generated) >= maximum:
                break
            pending, elapsed, metrics = self.window(self.generated[-1], maximum-len(self.generated))
            decode_ms += elapsed
            if len(self.generated)-last_stats_at >= self.args.stats_every:
                last_stats_at = len(self.generated)
                print(f'V11 generated {len(self.generated)}: {(len(self.generated)-1)*1000/decode_ms:.2f} tok/s; MTP {json.dumps(metrics)}', file=sys.stderr, flush=True)
        if not metrics:
            reply = self.command('STATS')
            if not reply.startswith('STATS '):
                raise RuntimeError('Invalid completion metrics reply')
            metrics = json.loads(reply[6:])
        count = len(self.generated)
        self.last_stats = {'prompt_tokens': len(ids), 'completion_tokens': count,
                           'finish_reason': finish, 'prefill_ms': prefill_ms,
                           'decode_ms': decode_ms, 'tok_s': (count-1)*1000/decode_ms if count > 1 and decode_ms else 0,
                           'mtp': metrics, 'spec': self.args.spec,
                           'gpu_sample_avg_ms': {}, 'gpu_samples': 0}
        yield decoder.decode(b'', final=True)


def messages_to_ids(tk, template, messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError('messages must be a nonempty array')
    for message in messages:
        if not isinstance(message, dict) or message.get('role') not in ('system', 'user', 'assistant'):
            raise ValueError('Only system, user and assistant messages are supported')
        if not isinstance(message.get('content'), str):
            raise ValueError('Only text message content is supported')
    prompt = template.render(messages=messages, add_generation_prompt=True,
                             enable_thinking=False, tools=None)
    return tk.encode(prompt, parse_special=True)


def serve(worker, tk, template, args):
    lock = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        def setup(self):
            super().setup()
            self.connection.settimeout(30)
        def respond(self, code, body):
            raw = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        def error(self, code, message):
            self.close_connection = True
            self.respond(code, {'error': {'message': message, 'type': 'invalid_request_error' if code < 500 else 'server_error'}})
        def auth(self):
            if args.api_key and not hmac.compare_digest(self.headers.get('Authorization', '').encode('utf-8'), ('Bearer ' + args.api_key).encode('utf-8')):
                self.error(401, 'Invalid API key')
                return False
            return True
        def do_GET(self):
            if not self.auth():
                return
            if self.path == '/health':
                healthy = worker.p.poll() is None
                self.respond(200 if healthy else 503, {'status': 'ok' if healthy else 'unavailable',
                             'model': args.model, 'device': '8086:e211', 'busy': lock.locked(), 'mtp': {'spec': args.spec, 'min_p': args.spec_min_p}})
            elif self.path == '/v1/models':
                self.respond(200, {'object': 'list', 'data': [{'id': args.model, 'object': 'model', 'owned_by': 'local'}]})
            else:
                self.error(404, 'Unknown endpoint')
        def do_POST(self):
            if not self.auth():
                self.close_connection = True
                return
            if self.path != '/v1/chat/completions':
                self.error(404, 'Unknown endpoint')
                self.close_connection = True
                return
            acquired = False
            streaming = False
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if size < 1 or size > 2**20:
                    raise ValueError('JSON body must be 1..1048576 bytes')
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError('Request must be a JSON object')
                if body.get('model', args.model) != args.model:
                    raise ValueError('Unknown model')
                if not isinstance(body.get('stream', False), bool):
                    raise ValueError('stream must be a boolean')
                n, temperature = body.get('n', 1), body.get('temperature', 0)
                if (not isinstance(n, int) or isinstance(n, bool) or n != 1
                        or not isinstance(temperature, (int, float))
                        or isinstance(temperature, bool) or temperature != 0):
                    raise ValueError('V11 supports n=1 and temperature=0 (greedy)')
                if body.get('tools') or body.get('stop') or body.get('logit_bias'):
                    raise ValueError('Tools, custom stop and logit_bias are not supported in V11')
                maximum = body.get('max_completion_tokens', body.get('max_tokens', 512))
                if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
                    raise ValueError('max_tokens must be a positive integer')
                ids = messages_to_ids(tk, template, body.get('messages'))
                if len(ids) + maximum > args.max_context:
                    raise ValueError('Prompt plus max_tokens exceeds context')
                if not lock.acquire(blocking=False):
                    self.error(429, 'One request at a time; runtime is busy')
                    return
                acquired = True
                worker.last_stats = None
                rid, created = 'chatcmpl-' + uuid.uuid4().hex, int(time.time())
                def chunk(delta, finish=None):
                    return {'id': rid, 'object': 'chat.completion.chunk', 'created': created,
                            'model': args.model, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}
                if body.get('stream', False):
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Cache-Control', 'no-cache')
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    streaming = True
                    self.close_connection = True
                    def event(value):
                        self.wfile.write(('data: ' + json.dumps(value, ensure_ascii=False) + '\n\n').encode())
                        self.wfile.flush()
                    event(chunk({'role': 'assistant'}))
                    for piece in worker.generate(ids, maximum):
                        if piece:
                            event(chunk({'content': piece}))
                    event(chunk({}, worker.last_stats['finish_reason']))
                    self.wfile.write(b'data: [DONE]\n\n')
                    self.wfile.flush()
                else:
                    text = ''.join(worker.generate(ids, maximum))
                    stats = worker.last_stats
                    self.respond(200, {'id': rid, 'object': 'chat.completion', 'created': created,
                        'model': args.model, 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text},
                         'finish_reason': stats['finish_reason']}],
                        'usage': {'prompt_tokens': len(ids), 'completion_tokens': stats['completion_tokens'],
                                  'total_tokens': len(ids) + stats['completion_tokens']}, 'strata_stats': stats})
            except (ValueError, json.JSONDecodeError) as exc:
                if not streaming:
                    self.error(400, str(exc))
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                print(f'V11 API error: {exc}', file=sys.stderr, flush=True)
                if not streaming:
                    self.error(500, str(exc))
                else:
                    with contextlib.suppress(OSError):
                        self.wfile.write(('data: ' + json.dumps({'error': {'message': str(exc)}}) + '\n\n').encode())
            finally:
                if acquired:
                    if getattr(worker, 'last_stats', None):
                        print(json.dumps({'request_id': rid, **worker.last_stats}), file=sys.stderr, flush=True)
                    lock.release()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    try:
        print(f'V11 OpenAI API: http://{args.host}:{server.server_port}', file=sys.stderr, flush=True)
        server.serve_forever(poll_interval=.25)
    finally:
        server.server_close()


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runtime', type=Path, default=Path(__file__).resolve().parents[1] / 'build-vulkan/jr-vk-runtime-v11')
    ap.add_argument('--config', type=Path)
    ap.add_argument('--mtp', type=Path, default=Path('/data/strata-lab/data/mtp/rt'))
    ap.add_argument('--spec', type=int, choices=[0, 1, 2, 3, 4], default=4)
    ap.add_argument('--fixed-spec', action='store_true', help='disable observed-cost proposal depth selection')
    ap.add_argument('--spec-min-p', type=float, default=0.5)
    ap.add_argument('--force-drafts', type=Path, help=argparse.SUPPRESS)
    ap.add_argument('--force-reject', type=int, default=-1, help=argparse.SUPPRESS)
    ap.add_argument('--pack', type=Path)
    ap.add_argument('--native', type=Path)
    ap.add_argument('--ple-gguf', type=Path)
    ap.add_argument('--residency', type=Path)
    ap.add_argument('--max-context', type=int, default=8192)
    ap.add_argument('--kv', choices=['fp16'], default='fp16')
    ap.add_argument('--greedy', action='store_true')
    ap.add_argument('--prompt')
    ap.add_argument('--prompt-file', type=Path)
    ap.add_argument('--raw-prompt', action='store_true')
    ap.add_argument('--max-new', type=int, default=256)
    ap.add_argument('--ignore-eos', action='store_true')
    ap.add_argument('--stats-every', type=int, default=128)
    ap.add_argument('--checks', action='store_true')
    ap.add_argument('--draft-check', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--profile', action='store_true', help='GPU kernel profiling (adds timing overhead)')
    ap.add_argument('--state-checks', action='store_true', help='also scan recurrent state (slower diagnostics)')
    ap.add_argument('--logits-out', type=Path)
    ap.add_argument('--dump-dir', type=Path)
    ap.add_argument('--report', type=Path)
    ap.add_argument('--route-profile', type=Path)
    ap.add_argument('--server', action='store_true')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--api-key', default=os.environ.get('STRATA_API_KEY'))
    ap.add_argument('--model', default='swift-iq3_xxs-vulkan')
    return ap


def main(argv=None):
    ap = parser()
    # Config sets defaults; explicit CLI arguments always win.
    pre, _ = ap.parse_known_args(argv)
    if pre.config:
        config = json.loads(pre.config.read_text())
        allowed = {'pack', 'native', 'ple_gguf', 'residency', 'max_context', 'kv', 'model', 'host', 'port', 'checks', 'state_checks', 'stats_every', 'mtp', 'spec', 'spec_min_p'}
        if not isinstance(config, dict) or set(config) - allowed:
            ap.error('Unsupported config keys')
        ap.set_defaults(**config)
    args = ap.parse_args(argv)
    for key in ('pack', 'native', 'residency'):
        if not getattr(args, key):
            ap.error(f'--{key} is required (or supply it in --config)')
        setattr(args, key, Path(getattr(args, key)))
    if args.ple_gguf:
        args.ple_gguf = Path(args.ple_gguf)
    if args.spec not in range(5) or not 0 <= args.spec_min_p <= 1:
        ap.error('spec must be 0..4 and spec-min-p 0..1')
    if args.kv != 'fp16':
        ap.error('V11 requires FP16 KV')
    if not 1 <= args.max_context <= 8192 or args.max_new < 1 or args.stats_every < 1:
        ap.error('V11 requires context 1..8192 and positive max-new/stats-every')
    if args.server and (args.force_drafts or args.force_reject != -1 or args.draft_check):
        ap.error('Debug draft overrides/capture cannot be enabled in the API server')
    if args.server:
        try:
            local = ipaddress.ip_address(args.host).is_loopback
        except ValueError:
            local = args.host == 'localhost'
        if not local and not args.api_key:
            ap.error('A non-loopback --host requires --api-key')
    elif args.prompt is None and args.prompt_file is None:
        ap.error('Supply --prompt or --prompt-file')
    tk, template = tokenizer(args.pack)
    def shutdown(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, shutdown)
    worker = None
    try:
        worker = Worker(args)
        worker.tk = tk
        if args.server:
            serve(worker, tk, template, args)
        else:
            prompt = args.prompt_file.read_text() if args.prompt_file else args.prompt
            ids = tk.encode(prompt, parse_special=True) if args.raw_prompt else messages_to_ids(tk, template, [{'role': 'user', 'content': prompt}])
            for piece in worker.generate(ids, args.max_new, args.ignore_eos):
                print(piece, end='', flush=True)
            print()
            stats = worker.last_stats
            print(json.dumps(stats), file=sys.stderr, flush=True)
            if args.report:
                args.report.write_text(json.dumps({**stats, 'token_ids': worker.generated}, indent=2) + '\n')
    finally:
        if worker:
            worker.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f'V11 ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
