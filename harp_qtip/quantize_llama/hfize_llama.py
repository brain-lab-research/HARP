import argparse
import os
import time

import glog
import torch
from transformers import AutoTokenizer

from lib import utils
from lib.utils.unsafe_import import model_from_hf_path
from model.llama import LlamaForCausalLM
from transformers import LlamaForCausalLM as OrigLlama

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--quantized_path', type=str)
parser.add_argument('--hf_output_path', type=str)
parser.add_argument('--skip_list', default=None, type=str)
parser.add_argument(
    '--quantize_harp_theta',
    action='store_true',
    help='Quantize HARP theta parameters before bit counting / save_pretrained.'
)


def count_bits_for_quant_linear(mod, quantize_harp_theta=False):
    """
    Count bits for one QTIP QuantizedLinear.

    Quantized tensor bits:
        m * n * K
    because in QTIP, K is bits-per-weight.

    HARP bits:
        delegated to mod.harp.get_storage_bits()

    If quantize_harp_theta=True and mod.harp exists, we call
    mod.harp.quantize_theta_8bit() first (expected to be idempotent).
    """
    m = mod.out_features
    n = mod.in_features
    num_params = m * n

    bits_quant = num_params * int(mod.K)

    harp = getattr(mod, "harp", None)
    bits_harp = 0

    if harp is not None:
        if quantize_harp_theta:
            qfn = getattr(harp, "quantize_theta_8bit", None)
            if not callable(qfn):
                raise ValueError(
                    "quantize_harp_theta was requested, but this HARPProcessor "
                    "does not implement quantize_theta_8bit()."
                )
            qfn()  # should be idempotent

        gb = getattr(harp, "get_storage_bits", None)
        if not callable(gb):
            raise ValueError(
                "HARPProcessor is present, but get_storage_bits() is missing."
            )
        bits_harp = int(gb())

    return num_params, bits_quant + bits_harp


def maybe_load_quant_proj(layer_mod, orig_mod, path, cpu, skip_name,
                          skip_list_union, total_params, total_bits,
                          quantize_harp_theta):
    """
    Load one quantized projection if not skipped, otherwise restore dense original.
    Returns updated (total_params, total_bits).
    """
    if skip_name not in skip_list_union:
        saved_layer = torch.load(path, map_location=cpu, weights_only=False)
        utils.unpack_quip(layer_mod, saved_layer)

        nparams, nbits = count_bits_for_quant_linear(
            layer_mod,
            quantize_harp_theta=quantize_harp_theta,
        )
        total_params += nparams
        total_bits += nbits
    else:
        return False, total_params, total_bits

    return True, total_params, total_bits


def main(args):
    total_params = 0
    total_bits = 0

    assert os.path.exists(args.quantized_path)
    saved_config = torch.load(
        os.path.join(args.quantized_path, 'config.pt'),
        weights_only=False,
    )
    model_config = saved_config['model_config']
    glog.info(model_config)

    tokenizer = AutoTokenizer.from_pretrained(model_config._name_or_path)

    model = LlamaForCausalLM.from_pretrained(
        model_config._name_or_path,
        torch_dtype='auto',
        low_cpu_mem_usage=False,
        config=model_config,
    )

    orig_model = OrigLlama.from_pretrained(
        model_config._name_or_path,
        torch_dtype='auto',
        low_cpu_mem_usage=False,
        config=model_config,
    )

    if model_config.quip_params['skip_list'] is None:
        model_config.quip_params['skip_list'] = []

    cpu = torch.device('cpu')
    if os.path.exists(f'{args.quantized_path}/lmhead.pt'):
        lmhead_data = torch.load(
            f'{args.quantized_path}/lmhead.pt',
            map_location=cpu,
            weights_only=False,
        )
        model.lm_head.weight.copy_(lmhead_data['lm_head'].to(
            model.lm_head.weight.dtype))
        model.model.norm.weight.copy_(lmhead_data['norm'].to(
            model.model.norm.weight.dtype))

    if args.skip_list is not None:
        args.skip_list = args.skip_list.split(',')
    else:
        args.skip_list = []

    skip_list_union = [*args.skip_list, *model_config.quip_params['skip_list']]
    model.config.quip_params['skip_list'] = skip_list_union

    for ii in range(len(model.model.layers)):
        layer = model.model.layers[ii]
        orig_layer = orig_model.model.layers[ii]

        if os.path.exists(f'{args.quantized_path}/{ii}_layernorm.pt'):
            ln_data = torch.load(
                f'{args.quantized_path}/{ii}_layernorm.pt',
                map_location=cpu,
                weights_only=False,
            )
            layer.input_layernorm.weight.copy_(ln_data['input_layernorm'].to(
                layer.input_layernorm.weight.dtype))
            layer.post_attention_layernorm.weight.copy_(
                ln_data['post_attention_layernorm'].to(
                    layer.post_attention_layernorm.weight.dtype))

        loaded, total_params, total_bits = maybe_load_quant_proj(
            layer.self_attn.q_proj,
            orig_layer.self_attn.q_proj,
            f'{args.quantized_path}/{ii}_q.pt',
            cpu,
            f'{ii}_q',
            skip_list_union,
            total_params,
            total_bits,
            args.quantize_harp_theta,
        )
        if not loaded:
            layer.self_attn.q_proj = orig_layer.self_attn.q_proj

        loaded, total_params, total_bits = maybe_load_quant_proj(
            layer.self_attn.k_proj,
            orig_layer.self_attn.k_proj,
            f'{args.quantized_path}/{ii}_k.pt',
            cpu,
            f'{ii}_k',
            skip_list_union,
            total_params,
            total_bits,
            args.quantize_harp_theta,
        )
        if not loaded:
            layer.self_attn.k_proj = orig_layer.self_attn.k_proj

        loaded, total_params, total_bits = maybe_load_quant_proj(
            layer.self_attn.v_proj,
            orig_layer.self_attn.v_proj,
            f'{args.quantized_path}/{ii}_v.pt',
            cpu,
            f'{ii}_v',
            skip_list_union,
            total_params,
            total_bits,
            args.quantize_harp_theta,
        )
        if not loaded:
            layer.self_attn.v_proj = orig_layer.self_attn.v_proj

        loaded, total_params, total_bits = maybe_load_quant_proj(
            layer.self_attn.o_proj,
            orig_layer.self_attn.o_proj,
            f'{args.quantized_path}/{ii}_o.pt',
            cpu,
            f'{ii}_o',
            skip_list_union,
            total_params,
            total_bits,
            args.quantize_harp_theta,
        )
        if not loaded:
            layer.self_attn.o_proj = orig_layer.self_attn.o_proj

        loaded, total_params, total_bits = maybe_load_quant_proj(
            layer.mlp.up_proj,
            orig_layer.mlp.up_proj,
            f'{args.quantized_path}/{ii}_up.pt',
            cpu,
            f'{ii}_up',
            skip_list_union,
            total_params,
            total_bits,
            args.quantize_harp_theta,
        )
        if not loaded:
            layer.mlp.up_proj = orig_layer.mlp.up_proj

        loaded, total_params, total_bits = maybe_load_quant_proj(
            layer.mlp.gate_proj,
            orig_layer.mlp.gate_proj,
            f'{args.quantized_path}/{ii}_gate.pt',
            cpu,
            f'{ii}_gate',
            skip_list_union,
            total_params,
            total_bits,
            args.quantize_harp_theta,
        )
        if not loaded:
            layer.mlp.gate_proj = orig_layer.mlp.gate_proj

        loaded, total_params, total_bits = maybe_load_quant_proj(
            layer.mlp.down_proj,
            orig_layer.mlp.down_proj,
            f'{args.quantized_path}/{ii}_down.pt',
            cpu,
            f'{ii}_down',
            skip_list_union,
            total_params,
            total_bits,
            args.quantize_harp_theta,
        )
        if not loaded:
            layer.mlp.down_proj = orig_layer.mlp.down_proj

        glog.info(f'loaded layer {ii}')

    glog.info('saving model...')
    model.save_pretrained(args.hf_output_path, safe_serialization=True)

    del model

    model, _ = model_from_hf_path(args.hf_output_path)

    glog.info('successfully loaded hfized model')
    glog.info('generating some text...')

    start = time.time()
    prompt = 'It is a truth universally acknowledged that'
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(
        input_ids=inputs['input_ids'].cuda(),
        attention_mask=inputs['attention_mask'].cuda(),
        max_new_tokens=64,
        return_dict_in_generate=True,
    )
    token = outputs.sequences[0, :]
    output_str = tokenizer.decode(token)
    glog.info(output_str)

    if total_params > 0:
        bpp = total_bits / total_params
    else:
        bpp = 0.0

    glog.info("==== QTIP BPP REPORT ====")
    glog.info(f"Total quantized parameters: {total_params:,}")
    glog.info(f"Total bits (QTIP trellis + HARP): {total_bits:,}")
    glog.info(f"Average BPP: {bpp:.4f}")

    glog.info(f'elapsed: {time.time() - start}')


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    args = parser.parse_args()
    main(args)