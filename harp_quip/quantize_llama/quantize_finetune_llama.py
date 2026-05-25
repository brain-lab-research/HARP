import argparse
import os
import time
from typing import Any, Dict

import glog
import json
import subprocess
import sys

os.environ['PYTORCH_ALLOC_CONF'] = 'max_split_size_mb:512'

import torch

import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_attn_mask_utils import \
    _prepare_4d_causal_attention_mask

from lib import codebook, utils
from lib.algo import finetune, quip
from lib.linear import FusedLinear
from model.llama import LlamaDecoderLayer

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

    module_names = ("qkv", "o", "up", "down")
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

def _layer_required_artifacts(idx: int):
    return [f"{idx}_{name}.pt" for name in ("qkv", "o", "up", "down", "layernorm")]


def _upload_layer_artifacts(save_path: str, idx: int, args) -> None:
    if not getattr(args, "upload_after_layer", False):
        return

    if not getattr(args, "upload_repo_id", None):
        raise ValueError("--upload_after_layer requires --upload_repo_id")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    upload_script = getattr(args, "upload_script", None) or os.path.join(
        repo_root, "scripts", "upload_layer.py"
    )

    cmd = [
        sys.executable,
        upload_script,
        "--repo-id", args.upload_repo_id,
        "--repo-type", args.upload_repo_type,
        "--folder", save_path,
        "--layer", str(idx),
        "--token-env", args.upload_token_env,
        "--lock-path", args.upload_lock_path,
    ]

    glog.info(f"uploading completed layer {idx} artifacts to {args.upload_repo_id}")
    subprocess.check_call(cmd)


def _hf_complete_layers(args, n_layers: int):
    if not getattr(args, "skip_hf_existing", False):
        return set()

    if not getattr(args, "upload_repo_id", None):
        raise ValueError("--skip_hf_existing requires --upload_repo_id")

    token = os.environ.get(args.upload_token_env)
    if not token:
        raise RuntimeError(
            f"--skip_hf_existing requires upload token env var {args.upload_token_env}"
        )

    from huggingface_hub import HfApi

    remote_files = set(
        HfApi().list_repo_files(
            repo_id=args.upload_repo_id,
            repo_type=args.upload_repo_type,
            token=token,
        )
    )

    done = {
        i for i in range(n_layers)
        if all(name in remote_files for name in _layer_required_artifacts(i))
    }

    glog.info(
        f"HF resume: found {len(done)}/{n_layers} complete layers already in "
        f"{args.upload_repo_id}"
    )
    return done


def _join_checked(proc, label: str):
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(f"{label} failed with exit code {proc.exitcode}")

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_cpu_threads', default=8, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--devset_size', default=384, type=int)
parser.add_argument('--ctx_size', default=4096, type=int)
parser.add_argument('--save_path', type=str)
parser.add_argument('--hessian_path', type=str)
parser.add_argument('--base_model', type=str)
parser.add_argument('--sigma_reg', default=1e-2, type=float)
parser.add_argument('--sigma_reg2', default=1e-2, type=float)
parser.add_argument('--lora_rank',
                    default=0,
                    type=int,
                    help='if <=0 then turned off')
parser.add_argument('--scale_override', default=-1, type=float)
parser.add_argument('--resid_scale_override', default=-1, type=float)
parser.add_argument('--codebook', type=str)
parser.add_argument('--quip_tune_iters', default=10, type=int)
parser.add_argument('--use_fp64', action='store_true')
parser.add_argument('--full_svd', action='store_true')
parser.add_argument('--no_use_buffered', action='store_true')
parser.add_argument('--rescale_WH', action='store_true')
parser.add_argument('--sample_proc', default=1, type=int)
parser.add_argument('--lowmem_ldlq', action='store_true')
parser.add_argument('--ft_lr', default=5e-5, type=float)
parser.add_argument('--ft_susv_lr', default=5e-4, type=float)
parser.add_argument('--ft_harp_lr', default=5e-7, type=float)
parser.add_argument('--ft_bs', default=8, type=int)
parser.add_argument('--ft_update_freq', default=2, type=int)
parser.add_argument('--ft_epochs', default=5, type=int)
parser.add_argument('--ft_valid_freq', default=1, type=int)
parser.add_argument('--ft_valid_size', default=128, type=float)
parser.add_argument('--ft_early_stop', default=3, type=int)
parser.add_argument('--ft_train_mode', action='store_true')
parser.add_argument('--ft_grad_ckpt', action='store_true')
parser.add_argument('--xsamples_path', default=None, type=str,
                    help='Folder containing x_samples files named like the hessian files (e.g. 0_qkv.pt)')
parser.add_argument('--incoh_mode', default='had', type=str, choices=['had','kron','harp'])
# HARP
parser.add_argument('--harp_b', default=8, type=int)
parser.add_argument('--harp_max_b', default=8, type=int)
parser.add_argument('--harp_passes', default=1, type=int)

parser.add_argument('--harp_ordering_mode', default='stride', type=str,
                    choices=['stride'])

parser.add_argument('--harp_fixed_mixer', default='had_or_qr', type=str,
                    choices=['had_or_ident', 'identity', 'had_or_qr'])

parser.add_argument('--harp_kron_fallback', action='store_true')
parser.add_argument('--harp_use_givens_b2', action='store_true')

parser.add_argument('--harp_steps', default=1200, type=int)
parser.add_argument('--harp_strategy', default='proxy', type=str,
                    choices=['proxy'])

parser.add_argument('--harp_lr_u', default=3e-2, type=float)
parser.add_argument('--harp_lr_v', default=3e-2, type=float)

parser.add_argument('--harp_grad_clip', default=None, type=float)
parser.add_argument('--harp_reg_theta', default=0.0, type=float)

parser.add_argument('--harp_theta_init_scale', default=0.0, type=float)
parser.add_argument('--harp_theta_clip', default=None, type=float)

parser.add_argument('--harp_hbd_lambda', default=0.1, type=float)
parser.add_argument('--harp_hbd_block', default=8, type=int)
parser.add_argument('--harp_chunk_size', default=131072, type=int)
parser.add_argument('--layer_idx', default=None, type=int,
                    help='If set, only quantize this decoder layer index')
parser.add_argument('--harp_q_recompute_every',
                    default=1,
                    type=int,
                    help='Recompute Q(W_tilde) every k steps during HARP fitting. 1 = old behavior.')

parser.add_argument('--harp_early_stop_window',
                    default=0,
                    type=int,
                    help='Early-stop window for HARP. 0 disables early stopping.')

parser.add_argument('--harp_early_stop_min_rel_improve',
                    default=0.01,
                    type=float,
                    help='Minimum relative improvement required over the early-stop window.')

parser.add_argument("--resume_from_layer", default=0, type=int)
parser.add_argument("--stop_before_layer", default=None, type=int)

parser.add_argument("--skip_hf_existing", action="store_true")
parser.add_argument("--upload_after_layer", action="store_true")
parser.add_argument("--upload_repo_id", default=None, type=str)
parser.add_argument("--upload_repo_type", default="model", type=str)
parser.add_argument("--upload_token_env", default="HF_UPLOAD_TOKEN", type=str)
parser.add_argument("--upload_lock_path", default="/tmp/harp_hf_upload.lock", type=str)
parser.add_argument("--upload_script", default=None, type=str)



def check_exist(idx, args):
    suffix = ['qkv', 'o', 'up', 'down', 'layernorm']
    for _ in suffix:
        test = f'{args.save_path}/{idx}_{_}.pt'
        if not os.path.exists(test):
            return False
    return True


def quantize_llama_layer(layer, idx, cb, args, device, pre_orig_emb, orig_emb,
                         model_config):
    if check_exist(idx, args):
        glog.info(f"layer {idx} already complete locally")
        return

    layer_t0 = time.perf_counter()

    mixed_layer = LlamaDecoderLayer(model_config, idx).cpu()
    with torch.no_grad():
        weights = [
            layer.self_attn.q_proj.weight, layer.self_attn.k_proj.weight,
            layer.self_attn.v_proj.weight
        ]

        fused_qkv_proj = FusedLinear(-1, [_.shape[0] for _ in weights],
                                     weights[0].shape[1],
                                     sum([_.shape[0] for _ in weights]),
                                     bias=False)
        cur = 0
        for w in weights:
            fused_qkv_proj.weight[cur:cur + w.shape[0]].copy_(w)
            cur += w.shape[0]

        mixed_layer.self_attn.qkv_proj = fused_qkv_proj

        mixed_layer.self_attn.o_proj = layer.self_attn.o_proj

        weights = [layer.mlp.up_proj.weight, layer.mlp.gate_proj.weight]
        fused_upgate_proj = FusedLinear(-1, [_.shape[0] for _ in weights],
                                        weights[0].shape[1],
                                        sum([_.shape[0] for _ in weights]),
                                        bias=False)
        cur = 0
        for w in weights:
            fused_upgate_proj.weight[cur:cur + w.shape[0]].copy_(w)
            cur += w.shape[0]
        mixed_layer.mlp.upgate_proj = fused_upgate_proj

        mixed_layer.mlp.down_proj = layer.mlp.down_proj

        mixed_layer.input_layernorm.weight.copy_(layer.input_layernorm.weight)
        mixed_layer.post_attention_layernorm.weight.copy_(
            layer.post_attention_layernorm.weight)

    timing_summary = finetune.quantize_finetune_decoder_layer(
        mixed_layer,
        [('self_attn.qkv_proj', 'qkv'),
         ('self_attn.o_proj', 'o'),
         ('mlp.upgate_proj', 'up'),
         ('mlp.down_proj', 'down')],
        idx,
        cb,
        args,
        device,
        pre_orig_emb,
        orig_emb,
    )

    torch.save(
        {
            'input_layernorm':
            mixed_layer.input_layernorm.weight,
            'post_attention_layernorm':
            mixed_layer.post_attention_layernorm.weight,
        }, f'{args.save_path}/{idx}_layernorm.pt')
    del mixed_layer

    if timing_summary is not None:
        timing_summary["layer_wall_time_s"] = float(time.perf_counter() -
                                                    layer_t0)
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
    tok_emb = model.model.embed_tokens(devset)  # on CPU/GPU depending on model
    # move working tensor to device
    cur = tok_emb.to(device)

    position_ids = torch.arange(ctx_size, dtype=torch.int32, device=device)[None, :]
    position_ids = position_ids.expand(batch_size, -1).contiguous()

    from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
    attn_mask = _prepare_4d_causal_attention_mask(
        None, (batch_size, ctx_size), cur[:batch_size], device
    )

    # run up to layer_idx to get pre
    S = devset.shape[0]
    for li in range(layer_idx):
        layer = model.model.layers[li].to(device)
        out = []
        for j in range(0, S, batch_size):
            x = cur[j:j+batch_size]
            bsz = x.shape[0]
            pid = position_ids[:bsz]
            am = _prepare_4d_causal_attention_mask(None, (bsz, ctx_size), x, device)
            out.append(layer(x, position_ids=pid, attention_mask=am,
                             use_cache=False, output_attentions=False)[0].detach())
        cur = torch.cat(out, dim=0)
        layer.cpu()
        torch.cuda.empty_cache()

    pre = cur.cpu()  # input to layer_idx

    # run layer_idx once to get post
    layer = model.model.layers[layer_idx].to(device)
    out = []
    for j in range(0, S, batch_size):
        x = cur[j:j+batch_size]
        bsz = x.shape[0]
        pid = position_ids[:bsz]
        am = _prepare_4d_causal_attention_mask(None, (bsz, ctx_size), x, device)
        out.append(layer(x, position_ids=pid, attention_mask=am,
                         use_cache=False, output_attentions=False)[0].detach())
    post = torch.cat(out, dim=0).cpu()
    layer.cpu()
    torch.cuda.empty_cache()

    return pre, post

def _finish_worker_on_device(dev, active_procs, active_layers, args):
    proc = active_procs[dev]
    layer_idx = active_layers.get(dev)

    _join_checked(proc, f"worker on GPU {dev}, layer {layer_idx}")

    active_procs[dev] = None
    active_layers.pop(dev, None)

    if layer_idx is not None:
        glog.info(f"GPU {dev} finished layer {layer_idx}; uploading from parent process")
        _upload_layer_artifacts(args.save_path, layer_idx, args)
        glog.info(f"GPU {dev} layer {layer_idx} upload done; slot is free")


def wait_for_finished_device(active_procs, active_layers, args):
    while True:
        for dev, proc in active_procs.items():
            if proc is not None and not proc.is_alive():
                _finish_worker_on_device(dev, active_procs, active_layers, args)
                return dev
        time.sleep(0.2)


def wait_for_available_device(active_procs, active_layers, args):
    for dev, proc in active_procs.items():
        if proc is None:
            return dev

    return wait_for_finished_device(active_procs, active_layers, args)


def drain_active_workers(active_procs, active_layers, args):
    while any(proc is not None for proc in active_procs.values()):
        wait_for_finished_device(active_procs, active_layers, args)

def main(args):
    overall_t0 = time.perf_counter()
    spawn_ctx = mp.get_context("spawn")

    if args.incoh_mode == "harp":
        args.harp_cfg = make_harp_cfg_from_args(args)
    else:
        args.harp_cfg = None

    glog.info("==== Arguments ====")
    glog.info(f'{args}')

    dtype_ = torch.float64 if args.use_fp64 else torch.float32

    cb = codebook.get_codebook(args.codebook)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype='auto',
        low_cpu_mem_usage=False
    )

    do_ft = int(args.ft_epochs) > 0


    all_config = {'quant_args': args, 'model_config': model.config}
    quip_params = {
        'lora_rank': args.lora_rank,
        'rescale_WH': args.rescale_WH,
        'codebook': args.codebook,
        'codebook_version': cb.version,
        'codesz': cb.codesz,
        'idx_dtype': str(cb.idx_dtype),
        'packsz': cb.packsz,
        'incoh_mode': args.incoh_mode,
        'resid_scale_override': args.resid_scale_override,
        'quant_layers': [int(args.layer_idx)] if args.layer_idx is not None else None,
    }

    if args.incoh_mode == "harp":
        assert args.harp_cfg is not None
        quip_params["harp"] = dict(args.harp_cfg)

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
    current_emb = None

    if do_ft:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        tokenizer.pad_token = tokenizer.eos_token

        devset = utils.sample_rp1t(tokenizer, args.devset_size, args.ctx_size,
                                   args.sample_proc)
        glog.info('loaded dataset and devset')

        orig_emb_cache = [model.model.embed_tokens(devset)]
        for _ in range(nproc):
            orig_emb_cache.append(
                torch.zeros(orig_emb_cache[0].shape,
                            dtype=orig_emb_cache[0].dtype,
                            device=orig_emb_cache[0].device))

        position_ids = torch.arange(args.ctx_size, dtype=torch.int32)[None, :] + \
            torch.zeros(args.batch_size, args.ctx_size, dtype=torch.int32)
        attention_mask = _prepare_4d_causal_attention_mask(
            None, (args.batch_size, args.ctx_size),
            orig_emb_cache[0][:args.batch_size], 0)

        proc_list = [None for _ in range(nproc)]
    else:
        # No finetuning: no shared activation ring, no barrier needed.
        active_procs = {dev: None for dev in range(nproc)}
        active_layers = {}

    remote_done_layers = _hf_complete_layers(args, len(model.model.layers))

    start_layer = max(0, int(args.resume_from_layer))
    stop_before_layer = (
        len(model.model.layers)
        if args.stop_before_layer is None
        else min(int(args.stop_before_layer), len(model.model.layers))
    )

    layer_indices = [
        i for i in range(start_layer, stop_before_layer)
        if i not in remote_done_layers
    ]

    if do_ft and (args.resume_from_layer != 0 or args.skip_hf_existing):
        raise ValueError(
            "This resume/remote-skip path is intended for --ft_epochs 0. "
            "For ft_epochs > 0, use --layer_idx or implement activation replay carefully."
        )

    if args.layer_idx is not None:
        idx = int(args.layer_idx)
        assert 0 <= idx < len(model.model.layers)

        if idx in remote_done_layers:
            glog.info(f"layer {idx} already complete on HF; skipping")
            return

        if do_ft:
            pre_emb, post_emb = compute_layer_io(
                model, devset, idx, args.batch_size, args.ctx_size, device=0
            )
        else:
            pre_emb, post_emb = None, None

        quantize_llama_layer(
            model.model.layers[idx],
            idx,
            cb,
            args,
            device=0,
            pre_orig_emb=pre_emb,
            orig_emb=post_emb,
            model_config=all_config['model_config'],
        )

        total_wall_clock_s = time.perf_counter() - overall_t0
        _log_timing_summary(args.save_path, total_wall_clock_s)
        return


    if do_ft:
        # Synchronized scheduling when ft_epochs > 0,
        # because the shared activation ring buffers are reused across layers.
        for i in range(len(model.model.layers)):
            glog.info(f'layer {i} gpu {cur_device}')
            if proc_list[cur_device] is not None:
                proc_list[cur_device].join()
                if cur_device == 0:
                    orig_emb_cache[0].copy_(orig_emb_cache[-1])
            if cur_device + 1 < nproc and proc_list[cur_device + 1] is not None:
                proc_list[cur_device + 1].join()
            utils.clean()

            st = time.time()
            position_ids = position_ids.to(cur_device)
            attention_mask = attention_mask.to(cur_device)
            model.model.layers[i].to(cur_device)
            for j in range(args.devset_size // args.batch_size):
                orig_emb_cache[cur_device + 1][
                    args.batch_size * j : args.batch_size * (j + 1)] = \
                    model.model.layers[i](
                        orig_emb_cache[cur_device][
                            args.batch_size * j : args.batch_size * (j + 1)].to(cur_device),
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_attentions=False)[0].cpu()
            model.model.layers[i].cpu()
            orig_msv = orig_emb_cache[cur_device].float().norm(
            )**2 / orig_emb_cache[cur_device].numel()
            target_msv = orig_emb_cache[cur_device + 1].float().norm(
            )**2 / orig_emb_cache[cur_device + 1].numel()
            position_ids = position_ids.cpu()
            attention_mask = attention_mask.cpu()
            utils.clean()
            glog.info(
                'computed original embedding for layer {} in {}s, pre msv {}, post msv {}'
                .format(i,
                        time.time() - st, orig_msv, target_msv))

            proc_list[cur_device] = spawn_ctx.Process(target=quantize_llama_layer,
                                          args=(
                                              model.model.layers[i],
                                              i,
                                              cb,
                                              args,
                                              cur_device,
                                              orig_emb_cache[cur_device],
                                              orig_emb_cache[cur_device + 1],
                                              all_config['model_config'],
                                          ))
            proc_list[cur_device].start()

            cur_device = (cur_device + 1) % nproc

        for p in proc_list:
            p.join()
    else:
        # No finetuning: use dynamic scheduling with no cross-GPU barrier.
        for i in layer_indices:
            device = wait_for_available_device(active_procs, active_layers, args)
            glog.info(f'layer {i} gpu {device}')
            utils.clean()

            p = spawn_ctx.Process(
                target=quantize_llama_layer,
                args=(
                    model.model.layers[i],
                    i,
                    cb,
                    args,
                    device,
                    None,
                    None,
                    all_config['model_config'],
                ))
            p.start()
            active_procs[device] = p
            active_layers[device] = i
            glog.info(f"launched layer {i} on GPU {device}")

        drain_active_workers(active_procs, active_layers, args)
    
    total_wall_clock_s = time.perf_counter() - overall_t0
    _log_timing_summary(args.save_path, total_wall_clock_s)


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    mp.set_sharing_strategy('file_system')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    main(args)