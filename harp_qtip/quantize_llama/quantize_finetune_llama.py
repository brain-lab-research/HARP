import argparse
import json
import os
import time
from typing import Any, Dict

import glog

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_attn_mask_utils import \
    _prepare_4d_causal_attention_mask

from lib import utils
from lib.algo import finetune
from lib.codebook import bitshift
from operator import attrgetter


def make_harp_cfg_from_args(args) -> Dict[str, Any]:
    return {
        "harp_b": args.harp_b,
        "harp_max_b": args.harp_max_b,
        "harp_passes": args.harp_passes,
        "ordering_mode": args.harp_ordering_mode,
        "fixed_mixer": args.harp_fixed_mixer,
        "kron_fallback": args.harp_kron_fallback,
        "use_givens_b2": args.harp_use_givens_b2,
        "theta_init_scale": args.harp_theta_init_scale,
        "theta_clip": args.harp_theta_clip,

        "steps": args.harp_steps,
        "strategy": args.harp_strategy,
        "lr_u": args.harp_lr_u,
        "lr_v": args.harp_lr_v,
        "grad_clip": args.harp_grad_clip,
        "reg_theta": args.harp_reg_theta,
        "hbd_lambda": args.harp_hbd_lambda,
        "hbd_block": args.harp_hbd_block,
        "chunk_size": args.harp_chunk_size,
        "q_recompute_every": args.harp_q_recompute_every,
        "q_recompute_every_final": args.harp_q_recompute_every_final,
        "q_refresh_parts": args.harp_q_refresh_parts,
        "early_stop_window": args.harp_early_stop_window,
        "early_stop_min_rel_improve": args.harp_early_stop_min_rel_improve,
    }


def _timing_path(save_path: str, layer_idx: int) -> str:
    return os.path.join(save_path, f'{layer_idx}_timing.json')


def _save_layer_timing_summary(save_path: str, summary: Dict[str, Any]) -> None:
    path = _timing_path(save_path, int(summary["layer_idx"]))
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _collect_layer_timing_summaries(save_path: str):
    summaries = []
    if not os.path.isdir(save_path):
        return summaries

    for fn in os.listdir(save_path):
        if not fn.endswith("_timing.json"):
            continue
        with open(os.path.join(save_path, fn), "r") as f:
            summaries.append(json.load(f))

    summaries.sort(key=lambda x: int(x["layer_idx"]))
    return summaries


def _log_timing_summary(save_path: str, wall_clock_s: float) -> None:
    summaries = _collect_layer_timing_summaries(save_path)
    if not summaries:
        glog.warning("No layer timing summaries found; skipping timing aggregate.")
        return

    module_names = ("v", "q", "k", "o", "up", "gate", "down")
    module_totals = {k: 0.0 for k in module_names}
    module_counts = {k: 0 for k in module_names}
    layer_block_total_s = 0.0

    for s in summaries:
        layer_block_total_s += float(s["layer_block_calibration_time_s"])
        for name, t in s["module_calibration_times_s"].items():
            if name in module_totals:
                module_totals[name] += float(t)
                module_counts[name] += 1

    glog.info("==== Calibration timing summary ====")
    for name in module_names:
        if module_counts[name] > 0:
            avg_s = module_totals[name] / module_counts[name]
            glog.info(
                f'average calibration time for module type {name}: {avg_s:.2f}s '
                f'over {module_counts[name]} layers'
            )

    avg_layer_block_s = layer_block_total_s / len(summaries)
    glog.info(
        f'average calibration time per layer block: {avg_layer_block_s:.2f}s '
        f'over {len(summaries)} layers'
    )
    glog.info(
        f'total summed calibration time across layer blocks: {layer_block_total_s:.2f}s'
    )
    glog.info(f'total wall-clock time spent: {wall_clock_s:.2f}s')


parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_cpu_threads', default=8, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--devset_size', default=384, type=int)
parser.add_argument('--ctx_size', default=4096, type=int)
parser.add_argument('--save_path', type=str)
parser.add_argument('--in_hess_path', type=str)
parser.add_argument('--base_model', type=str)
parser.add_argument('--sigma_reg', default=1e-2, type=float)
parser.add_argument('--sigma_reg2', default=1e-2, type=float)
parser.add_argument('--scale_override', default=-1, type=float)
parser.add_argument('--codebook', type=str)
parser.add_argument('--use_fp64', action='store_true')
parser.add_argument('--no_use_buffered', action='store_true')
parser.add_argument('--sample_proc', default=1, type=int)
parser.add_argument('--lowmem_ldlq', action='store_true')
parser.add_argument('--ft_lr', default=3e-6, type=float)
parser.add_argument('--ft_bs', default=4, type=int)
parser.add_argument('--ft_update_freq', default=1, type=int)
parser.add_argument('--ft_epochs', default=5, type=int)
parser.add_argument('--ft_valid_freq', default=1, type=int)
parser.add_argument('--ft_valid_size', default=128, type=float)
parser.add_argument('--ft_early_stop', default=5, type=int)
parser.add_argument('--ft_grad_ckpt', action='store_true')
parser.add_argument('--td_x', default=16, type=int)
parser.add_argument('--td_y', default=16, type=int)
parser.add_argument('--L', default=16, type=int)
parser.add_argument('--K', default=2, type=int)
parser.add_argument('--V', default=2, type=int)
parser.add_argument('--tlut_bits', default=0, type=int)
parser.add_argument('--decode_mode', default='lut', type=str)
parser.add_argument('--ft_train_lut', action='store_true')
parser.add_argument('--split_for_tp', action='store_true')
parser.add_argument('--tp_rank', default=8, type=int)
parser.add_argument('--skip_list', default=None, type=str)

parser.add_argument('--incoh_mode', default='had', choices=['had', 'harp'])

parser.add_argument('--harp_b', default=8, type=int)
parser.add_argument('--harp_max_b', default=8, type=int)
parser.add_argument('--harp_passes', default=1, type=int)
parser.add_argument('--harp_ordering_mode', default='stride', choices=['stride'])
parser.add_argument('--harp_fixed_mixer', default='had_or_qr',
                    choices=['had_or_ident', 'identity', 'had_or_qr'])
parser.add_argument('--harp_kron_fallback', action='store_true')
parser.add_argument('--harp_use_givens_b2', action='store_true')

parser.add_argument('--harp_steps', default=600, type=int)
parser.add_argument('--harp_strategy', default='proxy', choices=['proxy'])
parser.add_argument('--harp_lr_u', default=3e-2, type=float)
parser.add_argument('--harp_lr_v', default=3e-2, type=float)
parser.add_argument('--harp_grad_clip', default=None, type=float)
parser.add_argument('--harp_reg_theta', default=0.0, type=float)
parser.add_argument('--harp_theta_init_scale', default=0.0, type=float)
parser.add_argument('--harp_theta_clip', default=None, type=float)
parser.add_argument('--harp_hbd_lambda', default=0.1, type=float)
parser.add_argument('--harp_hbd_block', default=16, type=int)
parser.add_argument('--harp_chunk_size', default=4096, type=int)

parser.add_argument('--harp_q_recompute_every', default=6, type=int)

parser.add_argument('--harp_q_recompute_every_final', default=2, type=int)

parser.add_argument('--harp_q_refresh_parts', default=8, type=int)

parser.add_argument('--harp_early_stop_window', default=0, type=int)
parser.add_argument('--harp_early_stop_min_rel_improve', default=0.005, type=float)

parser.add_argument('--ft_susv_lr', default=None, type=float)
parser.add_argument('--ft_harp_lr', default=0.0, type=float)

parser.add_argument('--layer_idx', default=None, type=int,
                    help='If set, only quantize this decoder layer index')


def check_exist(idx, args):
    suffix = ['q', 'k', 'v', 'o', 'up', 'down', 'layernorm']
    for _ in suffix:
        test = f'{args.save_path}/{idx}_{_}.pt'
        if not os.path.exists(test):
            return False
    return True


def quantize_llama_decoder(layer, idx, cb, args, device, pre_orig_emb,
                           orig_emb, model_config, skip_list):
    if check_exist(idx, args):
        return

    layer_t0 = time.perf_counter()

    if skip_list is None:
        skip_list = []

    quant_order = []
    for thing in [('self_attn.v_proj', 'v', 'qkv', 'v', 'col'),
                  ('self_attn.q_proj', 'q', 'qkv', 'q', 'col'),
                  ('self_attn.k_proj', 'k', 'qkv', 'k', 'col'),
                  ('self_attn.o_proj', 'o', 'o', 'o', 'row'),
                  ('mlp.up_proj', 'up', 'up', 'up', 'col'),
                  ('mlp.gate_proj', 'gate', 'up', 'gate', 'col'),
                  ('mlp.down_proj', 'down', 'down', 'down', 'row')]:
        if f'{idx}_{thing[1]}' not in skip_list:
            quant_order.append(thing)
        else:
            attrgetter(thing[0])(layer).weight.requires_grad = False
            glog.info(f'skipping {idx}_{thing[1]}')

    timing_summary = finetune.quantize_finetune_decoder_layer(
        layer,
        quant_order,
        idx,
        cb,
        args,
        device,
        pre_orig_emb,
        orig_emb,
    )

    torch.save(
        {
            'input_layernorm': layer.input_layernorm.weight,
            'post_attention_layernorm': layer.post_attention_layernorm.weight,
        },
        f'{args.save_path}/{idx}_layernorm.pt'
    )

    if timing_summary is not None:
        timing_summary["layer_wall_time_s"] = float(time.perf_counter() - layer_t0)
        glog.info(
            f'layer {idx} whole layer-block calibration time: '
            f'{timing_summary["layer_block_calibration_time_s"]:.2f}s'
        )
        glog.info(
            f'layer {idx} wall time (including wrapping/saving): '
            f'{timing_summary["layer_wall_time_s"]:.2f}s'
        )
        _save_layer_timing_summary(args.save_path, timing_summary)


@torch.no_grad()
def compute_layer_io(model, devset, layer_idx, batch_size, ctx_size, device):
    # devset: [S, ctx]
    tok_emb = model.model.embed_tokens(devset)
    cur = tok_emb.to(device)

    S = devset.shape[0]
    position_ids_full = torch.arange(
        ctx_size, dtype=torch.int32, device=device
    )[None, :]

    for li in range(layer_idx):
        layer = model.model.layers[li].to(device)
        out = []
        for j in range(0, S, batch_size):
            x = cur[j:j + batch_size]
            bsz = x.shape[0]
            pid = position_ids_full.expand(bsz, -1).contiguous()
            am = _prepare_4d_causal_attention_mask(
                None, (bsz, ctx_size), x, 0
            )
            out.append(
                layer(
                    x,
                    position_ids=pid,
                    attention_mask=am,
                    use_cache=False,
                    output_attentions=False
                )[0].detach()
            )
        cur = torch.cat(out, dim=0)
        layer.cpu()
        torch.cuda.empty_cache()

    pre = cur.cpu()

    layer = model.model.layers[layer_idx].to(device)
    out = []
    for j in range(0, S, batch_size):
        x = cur[j:j + batch_size]
        bsz = x.shape[0]
        pid = position_ids_full.expand(bsz, -1).contiguous()
        am = _prepare_4d_causal_attention_mask(
            None, (bsz, ctx_size), x, 0
        )
        out.append(
            layer(
                x,
                position_ids=pid,
                attention_mask=am,
                use_cache=False,
                output_attentions=False
            )[0].detach()
        )
    post = torch.cat(out, dim=0).cpu()
    layer.cpu()
    torch.cuda.empty_cache()

    return pre, post


def wait_for_available_device(active_procs):
    while True:
        for dev, proc in active_procs.items():
            if proc is None:
                return dev
            if not proc.is_alive():
                proc.join()
                active_procs[dev] = None
                return dev
        time.sleep(0.2)


def main(args):
    overall_t0 = time.perf_counter()
    spawn_ctx = mp.get_context("spawn")

    if args.skip_list is not None:
        args.skip_list = args.skip_list.split(',')

    if args.ft_susv_lr is None:
        args.ft_susv_lr = args.ft_lr

    if args.incoh_mode == 'harp':
        args.harp_cfg = make_harp_cfg_from_args(args)
    else:
        args.harp_cfg = None

    glog.info("==== Arguments ====")
    glog.info(f'{args}')

    cb = bitshift.bitshift_codebook(
        L=args.L,
        K=args.K,
        V=args.V,
        tlut_bits=args.tlut_bits,
        decode_mode=args.decode_mode
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype='auto',
        low_cpu_mem_usage=True
    )

    do_ft = int(args.ft_epochs) > 0

    all_config = {'quant_args': args, 'model_config': model.config}
    quip_params = {
        'codebook': args.codebook,
        'codebook_version': cb.version,
        'L': args.L,
        'K': args.K,
        'V': args.V,
        'tlut_bits': args.tlut_bits,
        'decode_mode': args.decode_mode,
        'td_x': args.td_x,
        'td_y': args.td_y,
        'split_for_tp': args.split_for_tp,
        'tp_rank': args.tp_rank,
        'skip_list': args.skip_list,
        'incoh_mode': args.incoh_mode,
        'quant_layers': [int(args.layer_idx)] if args.layer_idx is not None else None,
    }
    if args.incoh_mode == 'harp':
        quip_params['harp'] = dict(args.harp_cfg)

    all_config['model_config'].update({'quip_params': quip_params})
    torch.save(all_config, os.path.join(args.save_path, 'config.pt'))

    glog.info("==== Full Config ====")
    glog.info(f'{all_config}')
    glog.info('loaded model')

    nproc = torch.cuda.device_count()

    devset = None
    orig_emb_cache = None
    position_ids = None
    attention_mask = None
    cur_device = 0
    proc_list = None
    active_procs = None

    if do_ft:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        tokenizer.pad_token = tokenizer.eos_token

        devset = utils.sample_rp1t(
            tokenizer, args.devset_size, args.ctx_size, args.sample_proc
        )
        glog.info('loaded dataset and devset')

        orig_emb_cache = [model.model.embed_tokens(devset)]
        for _ in range(nproc):
            orig_emb_cache.append(
                torch.zeros(
                    orig_emb_cache[0].shape,
                    dtype=orig_emb_cache[0].dtype,
                    device=orig_emb_cache[0].device
                )
            )

        position_ids = torch.arange(args.ctx_size, dtype=torch.int32)[None, :] + \
            torch.zeros(args.batch_size, args.ctx_size, dtype=torch.int32)
        attention_mask = _prepare_4d_causal_attention_mask(
            None, (args.batch_size, args.ctx_size),
            orig_emb_cache[0][:args.batch_size], 0
        )

        proc_list = [None for _ in range(nproc)]
    else:
        active_procs = {dev: None for dev in range(nproc)}

    if args.layer_idx is not None:
        idx = int(args.layer_idx)
        assert 0 <= idx < len(model.model.layers)

        if do_ft:
            pre_emb, post_emb = compute_layer_io(
                model, devset, idx, args.batch_size, args.ctx_size, device=0
            )
        else:
            pre_emb, post_emb = None, None

        quantize_llama_decoder(
            model.model.layers[idx],
            idx,
            cb,
            args,
            device=0,
            pre_orig_emb=pre_emb,
            orig_emb=post_emb,
            model_config=all_config['model_config'],
            skip_list=args.skip_list,
        )

        total_wall_clock_s = time.perf_counter() - overall_t0
        _log_timing_summary(args.save_path, total_wall_clock_s)
        return

    if do_ft:
        for i in range(len(model.model.layers)):
            glog.info(f'layer {i} gpu {cur_device}')

            if proc_list[cur_device] is not None:
                proc_list[cur_device][0].join()
                model.model.layers[proc_list[cur_device][1]] = None
                utils.clean()
                if cur_device == 0:
                    orig_emb_cache[0].copy_(orig_emb_cache[-1])

            if cur_device + 1 < nproc and proc_list[cur_device + 1] is not None:
                proc_list[cur_device + 1][0].join()

            utils.clean()
            st = time.time()

            position_ids = position_ids.to(cur_device)
            attention_mask = attention_mask.to(cur_device)
            model.model.layers[i].to(cur_device)

            for j in range(args.devset_size // args.batch_size):
                utils.clean()
                orig_emb_cache[cur_device + 1][
                    args.batch_size * j: args.batch_size * (j + 1)
                ] = model.model.layers[i](
                    orig_emb_cache[cur_device][
                        args.batch_size * j: args.batch_size * (j + 1)
                    ].to(cur_device),
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_attentions=False
                )[0].cpu()

            model.model.layers[i].cpu()

            orig_msv = orig_emb_cache[cur_device].float().norm()**2 / \
                orig_emb_cache[cur_device].numel()
            target_msv = orig_emb_cache[cur_device + 1].float().norm()**2 / \
                orig_emb_cache[cur_device + 1].numel()

            position_ids = position_ids.cpu()
            attention_mask = attention_mask.cpu()
            utils.clean()

            glog.info(
                'computed original embedding for layer {} in {}s, pre msv {}, post msv {}'.format(
                    i, time.time() - st, orig_msv, target_msv
                )
            )

            proc = spawn_ctx.Process(
                target=quantize_llama_decoder,
                args=(
                    model.model.layers[i],
                    i,
                    cb,
                    args,
                    cur_device,
                    orig_emb_cache[cur_device],
                    orig_emb_cache[cur_device + 1],
                    all_config['model_config'],
                    args.skip_list,
                )
            )
            proc.start()
            proc_list[cur_device] = (proc, i)

            cur_device = (cur_device + 1) % nproc

        for p in proc_list:
            p[0].join()
    else:
        for i in range(len(model.model.layers)):
            device = wait_for_available_device(active_procs)
            glog.info(f'layer {i} gpu {device}')
            utils.clean()

            p = spawn_ctx.Process(
                target=quantize_llama_decoder,
                args=(
                    model.model.layers[i],
                    i,
                    cb,
                    args,
                    device,
                    None,
                    None,
                    all_config['model_config'],
                    args.skip_list,
                )
            )
            p.start()
            active_procs[device] = p

        for p in active_procs.values():
            if p is not None:
                p.join()

    total_wall_clock_s = time.perf_counter() - overall_t0
    _log_timing_summary(args.save_path, total_wall_clock_s)


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    mp.set_sharing_strategy('file_system')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    main(args)