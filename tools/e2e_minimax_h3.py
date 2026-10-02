"""End-to-end check on a real ComfyUI server: MiniMax-H3 with and without
Veda, same seed, through the HTTP / websocket API.

    python tools/e2e_minimax_h3.py --server http://127.0.0.1:8188

Builds the API-format graph of the T2VA example workflow (official H3
template: UNET -> Turbo LoRA -> [Veda] -> BasicGuider, 8 steps), queues it
once per mode (Veda first, then full attention), and reports per-mode wall
time, per-step sampling time (first step excluded: it compiles kernels),
Veda's status lines from the node, and the saved video files. The models
named by the flags must already be in ComfyUI's model folders; the Veda
predictor is downloaded by the node itself on first use.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import sys
import time
import urllib.request
import uuid

import aiohttp

_TEXT = 3  # protocol.BinaryEventTypes.TEXT

PROMPT = (
    'Realistic live-action cinematic look: a red fox trots across a snowy '
    'pine forest clearing at dawn, golden light through the trees, breath '
    'visible in the cold air, shallow depth of field. The camera tracks '
    'beside it at ground level. Audio: crunching snow, soft wind in the '
    'pines, a distant bird call.')


def build_graph(args, veda: bool, prefix: str) -> dict:
    """API-format graph of the T2VA example workflow."""
    model = ['2', 0]
    graph = {
        '1': {'class_type': 'UNETLoader',
              'inputs': {'unet_name': args.unet, 'weight_dtype': 'default'}},
        '2': {'class_type': 'LoraLoaderModelOnly',
              'inputs': {'model': ['1', 0], 'lora_name': args.lora,
                         'strength_model': 1.0}},
        '4': {'class_type': 'CLIPLoader',
              'inputs': {'clip_name': args.clip, 'type': 'minimax',
                         'device': 'default'}},
        '5': {'class_type': 'VAELoader', 'inputs': {'vae_name': args.vae}},
        '6': {'class_type': 'VAELoader',
              'inputs': {'vae_name': args.audio_vae}},
        '7': {'class_type': 'MiniMaxH3ImageToVideo',
              'inputs': {'clip': ['4', 0], 'vae': ['5', 0],
                         'prompt': args.prompt, 'width': args.width,
                         'height': args.height, 'length': args.length}},
        '8': {'class_type': 'RandomNoise',
              'inputs': {'noise_seed': args.seed}},
        '9': {'class_type': 'KSamplerSelect',
              'inputs': {'sampler_name': 'res_multistep'}},
        '10': {'class_type': 'BasicScheduler',
               'inputs': {'model': ['2', 0], 'scheduler': 'simple',
                          'steps': args.steps, 'denoise': 1.0}},
        '12': {'class_type': 'SamplerCustomAdvanced',
               'inputs': {'noise': ['8', 0], 'guider': ['11', 0],
                          'sampler': ['9', 0], 'sigmas': ['10', 0],
                          'latent_image': ['7', 1]}},
        '13': {'class_type': 'VAEDecode',
               'inputs': {'samples': ['12', 0], 'vae': ['5', 0]}},
        '14': {'class_type': 'VAEDecodeAudio',
               'inputs': {'samples': ['12', 0], 'vae': ['6', 0]}},
        '15': {'class_type': 'CreateVideo',
               'inputs': {'images': ['13', 0], 'audio': ['14', 0],
                          'fps': 24.0}},
        '16': {'class_type': 'SaveVideo',
               'inputs': {'video': ['15', 0], 'filename_prefix': prefix,
                          'format': 'auto', 'format.codec': 'auto'}},
    }
    if veda:
        graph['3'] = {'class_type': 'VedaSparseAttention',
                      'inputs': {'model': ['2', 0],
                                 'predictor': args.predictor,
                                 'generated_sparsity': args.sparsity,
                                 'reference_sparsity': args.sparsity,
                                 'full_attention_layers': '',
                                 'full_attention_steps': '',
                                 'backend': args.backend,
                                 'verbose': args.verbose}}
        model = ['3', 0]
    graph['11'] = {'class_type': 'BasicGuider',
                   'inputs': {'model': model, 'conditioning': ['7', 0]}}
    return graph


async def run(server: str, graph: dict) -> dict:
    """Queues a graph and follows it to the end over the websocket."""
    client_id = uuid.uuid4().hex
    ws_url = server.replace('http', 'ws', 1) + f'/ws?clientId={client_id}'
    report = {'texts': [], 'steps': [], 'error': None}
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(ws_url, max_msg_size=0) as ws:
            body = json.dumps({'prompt': graph,
                               'client_id': client_id}).encode()
            request = urllib.request.Request(
                f'{server}/prompt', data=body,
                headers={'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(request) as response:
                    prompt_id = json.load(response)['prompt_id']
            except urllib.error.HTTPError as error:
                report['error'] = error.read().decode()
                return report
            start = time.perf_counter()
            async for message in ws:
                now = time.perf_counter() - start
                if message.type == aiohttp.WSMsgType.BINARY:
                    data = message.data
                    if struct.unpack('>I', data[:4])[0] == _TEXT:
                        size = struct.unpack('>I', data[4:8])[0]
                        node = data[8:8 + size].decode()
                        text = data[8 + size:].decode('utf-8', 'replace')
                        report['texts'].append((round(now, 1), node, text))
                    continue
                if message.type != aiohttp.WSMsgType.TEXT:
                    continue
                event = json.loads(message.data)
                kind, payload = event['type'], event.get('data', {})
                if payload.get('prompt_id') not in (None, prompt_id):
                    continue
                if kind == 'progress' and payload.get('node') == '12':
                    report['steps'].append((payload['value'], now))
                elif kind == 'execution_error':
                    report['error'] = payload.get('exception_message')
                    break
                elif kind == 'executing' and payload.get('node') is None:
                    break
                elif kind == 'execution_success':
                    break
            report['seconds'] = round(time.perf_counter() - start, 1)
    with urllib.request.urlopen(f'{server}/history/{prompt_id}') as response:
        history = json.load(response).get(prompt_id, {})
    outputs = history.get('outputs', {}).get('16', {})
    report['files'] = [f['filename'] for value in outputs.values()
                       if isinstance(value, list)
                       for f in value if isinstance(f, dict)
                       and 'filename' in f]
    return report


def main() -> None:
    # Node texts contain emoji; a redirected Windows console is cp1252.
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--server', default='http://127.0.0.1:8188')
    parser.add_argument('--modes', default='veda,dense')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--width', type=int, default=1344)
    parser.add_argument('--height', type=int, default=768)
    parser.add_argument('--length', type=int, default=124,
                        help='frames (17k+5); 124 = 5 s')
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--prompt', default=PROMPT)
    parser.add_argument('--backend', default='auto')
    parser.add_argument('--sparsity', default='90%')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument(
        '--unet', default='minimax_h3_fl2va_pruned_int8_convrot.safetensors')
    parser.add_argument(
        '--lora',
        default='minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors')
    parser.add_argument(
        '--clip', default='qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors')
    parser.add_argument(
        '--vae', default='minimax_h3_video_vae_int8_convrot.safetensors')
    parser.add_argument('--audio-vae',
                        default='minimax_h3_audio_vae_fp32.safetensors')
    parser.add_argument(
        '--predictor',
        default='minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors')
    args = parser.parse_args()
    results = {}
    for mode in args.modes.split(','):
        graph = build_graph(args, mode == 'veda',
                            f'veda_e2e/seed{args.seed}_{mode}')
        print(f'== {mode}: queued', flush=True)
        report = asyncio.run(run(args.server, graph))
        results[mode] = report
        for at, node, text in report['texts']:
            print(f'  [{at:7.1f}s] node {node}: {text}', flush=True)
        if report['error']:
            print(f'  ERROR: {report["error"]}', flush=True)
            continue
        steps = report['steps']
        if len(steps) >= 2:
            per_step = (steps[-1][1] - steps[0][1]) / (len(steps) - 1)
            print(f'  sampling: {len(steps)} steps, {per_step:.1f} s/step '
                  'after the first', flush=True)
        print(f'  total {report["seconds"]} s, files {report["files"]}',
              flush=True)
    failed = [m for m, r in results.items() if r['error']]
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
