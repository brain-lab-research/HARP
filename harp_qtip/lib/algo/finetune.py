"""
Utilities for fine tuning
"""
import copy
import math
import time
from contextlib import contextmanager
from operator import attrgetter

import glog
import torch
from torch import multiprocessing as mp
from torch import nn
from transformers import AutoModelForCausalLM

from lib import codebook, utils
from lib.linear import QuantizedLinear
from lib.incoherence.harp import HARPProcessor

from . import ldlq


@contextmanager
def use_tf32():
    fp32_matmul_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('high')
    yield
    torch.set_float32_matmul_precision(fp32_matmul_precision)

def _collect_ft_param_groups(module, args):
    susv_params = []
    harp_params = []
    tlut_params = []
    other_params = []

    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue

        parts = name.split(".")
        last = parts[-1]

        if last in ("SU", "SV"):
            susv_params.append(param)
        elif "harp" in parts:
            harp_params.append(param)
        elif last == "tlut":
            tlut_params.append(param)
        else:
            other_params.append(param)

    # sanity check: no duplicates across groups
    seen = set()
    for group in (susv_params, harp_params, tlut_params, other_params):
        for p in group:
            pid = id(p)
            if pid in seen:
                raise RuntimeError("Parameter appears in multiple optimizer groups.")
            seen.add(pid)

    param_groups = []

    if len(susv_params) > 0:
        param_groups.append({"params": susv_params, "lr": args.ft_lr})

    if len(harp_params) > 0:
        harp_lr = getattr(args, "ft_harp_lr", args.ft_lr)
        if harp_lr > 0:
            param_groups.append({"params": harp_params, "lr": harp_lr})

    if len(tlut_params) > 0:
        # keep LUT params on ft_lr unless you want a separate knob
        param_groups.append({"params": tlut_params, "lr": args.ft_lr})

    if len(other_params) > 0:
        param_groups.append({"params": other_params, "lr": args.ft_lr})

    if len(param_groups) == 0:
        raise RuntimeError("No trainable parameters found for finetuning.")

    return param_groups

def _make_harp_from_args(n, m, device, args, rcp, has_kernel):
    from lib.incoherence.harp import HARPProcessor

    cfg = dict(getattr(args, "harp_cfg", {}) or {})
    for k in (
        "n", "m", "device",
        "split_for_tp", "rcp", "tp_rank",
        "td_x", "td_y", "L", "K", "V",
        "tlut_bits", "decode_mode", "for_kernel",
    ):
        cfg.pop(k, None)

    split_for_tp = bool(
        getattr(args, "split_for_tp", False)
        and getattr(args, "tp_rank", 1) > 1
        and rcp in ("row", "col")
    )

    if isinstance(device, int):
        device_str = f"cuda:{device}"
    else:
        device_str = str(device)

    return HARPProcessor(
        n=n,
        m=m,
        device=device_str,
        split_for_tp=split_for_tp,
        rcp=rcp if split_for_tp else None,
        tp_rank=int(getattr(args, "tp_rank", 1)),
        td_x=int(args.td_x),
        td_y=int(args.td_y),
        L=int(args.L),
        K=int(args.K),
        V=int(args.V),
        tlut_bits=int(args.tlut_bits),
        decode_mode=str(args.decode_mode),
        for_kernel=bool(has_kernel),
        **cfg,
    )

def finetune_decoder_layer(layer, name, device, train_dl, valid_dl, orig_dtype,
                           args):
    with use_tf32():
        layer = layer.to(device)

        source = next(iter(train_dl))[0]
        position_ids = torch.arange(source.shape[1], device=device).unsqueeze(0)
        # manifest tensor parallel attributes in layer
        output = layer(source.to(device),
                       position_ids=position_ids)[0]
        
        best_sd = {k: v.cpu() for k, v in layer.state_dict().items()}
        utils.clean()

        optim = torch.optim.Adam(_collect_ft_param_groups(layer, args))
        best_loss = utils.calculate_mse_loss(layer, valid_dl, device)
        glog.info(f'layer {name} initial loss {best_loss}')
        scaler = torch.cuda.amp.GradScaler(enabled=(orig_dtype==torch.float16))
        worse_ct = 0

        for epoch in range(args.ft_epochs):
            for bidx, (source, targets) in enumerate(train_dl):
                targets = targets.to(device, non_blocking=True)
                with torch.autocast(device_type='cuda',
                                    dtype=orig_dtype,
                                    enabled=True):
                    output = layer(source.to(device),
                                   position_ids=position_ids)[0]
                    loss = nn.MSELoss()(output, targets)
                scaler.scale(loss).backward()
                if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(
                        train_dl) - 1:
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad()

            if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
                test_loss = utils.calculate_mse_loss(layer, valid_dl, device)
                if test_loss < best_loss:
                    glog.info(
                        f'layer {name} @ epoch {epoch} new loss {test_loss} old loss {best_loss} BETTER'
                    )
                    best_loss = test_loss
                    best_sd = {k: v.cpu() for k, v in layer.state_dict().items()}
                    utils.clean()
                    worse_ct = 0
                else:
                    glog.info(
                        f'layer {name} @ epoch {epoch} new loss {test_loss} old loss {best_loss} WORSE'
                    )
                    worse_ct += 1
                    if worse_ct >= args.ft_early_stop:
                        break

    del optim, train_dl, valid_dl

    layer = layer.cpu()
    layer.load_state_dict(best_sd)
    utils.clean()


def quantize_finetune_decoder_layer(mixed_layer, quant_order, idx, cb, args,
                                    device, pre_orig_emb, orig_emb):
    torch.manual_seed(idx)
    torch.set_num_threads(args.num_cpu_threads)
    torch.set_grad_enabled(False)

    dtype_ = torch.float64 if args.use_fp64 else torch.float32
    orig_dtype = None
    for p in mixed_layer.parameters():
        orig_dtype = p.dtype
        break
    mixed_layer = mixed_layer.float()

    do_ft = int(args.ft_epochs) > 0
    if do_ft:
        if pre_orig_emb is None or orig_emb is None:
            raise ValueError(
                "pre_orig_emb/orig_emb must be provided when ft_epochs > 0"
            )
        train_dl, valid_dl = utils.split_data(pre_orig_emb, orig_emb, args)
    else:
        train_dl, valid_dl = None, None

    has_kernel = utils.has_kernel(args.decode_mode, args.L, args.K, args.V,
                                  args.tlut_bits, args.td_x, args.td_y)

    module_calibration_times_s = {}

    for quant_i, (linear_attr, name, in_hess_name, out_hess_name,
                  rcp) in enumerate(quant_order):
        module_t0 = time.perf_counter()

        utils.clean()
        cb = cb.to(device).to(orig_dtype)

        orig_linear = attrgetter(linear_attr)(mixed_layer)
        W = orig_linear.weight.to(dtype_)
        del orig_linear

        (m, n) = W.shape
        SU = (torch.randn(n, device=device).sign() + 1e-5).sign().to(dtype_)
        SV = (torch.randn(m, device=device).sign() + 1e-5).sign().to(dtype_)

        in_hess_path = f'{args.in_hess_path}/{idx}_{in_hess_name}.pt'
        H_data = torch.load(in_hess_path, map_location=torch.device('cpu'))
        HR = utils.flat_to_sym(H_data['flatH'], H_data['n'])
        if 'mu' in H_data:
            mu = H_data['mu']
            HR += mu[None, :] * mu[:, None]
            del mu
        del H_data

        HR = utils.regularize_H(HR, args.sigma_reg)

        harp = None

        if args.incoh_mode == "had":
            if args.split_for_tp:
                if rcp == 'col':
                    Wr = utils.matmul_hadUt(
                        utils.matmul_hadUt((W.T.to(device) * SV).reshape(
                            n * args.tp_rank, m // args.tp_rank)).reshape(
                                W.T.shape).T * SU)
                    HRr = utils.matmul_hadUt(
                        utils.matmul_hadUt(HR.to(device) * SU).T * SU)

                    Wscale = Wr.reshape(
                        args.tp_rank, m * n // args.tp_rank).square().mean(
                            dim=-1).sqrt() / (cb.lut.to(
                                torch.float64).square().mean().sqrt().float() *
                                              args.scale_override)
                    Wr = Wr.reshape(args.tp_rank,
                                    m * n // args.tp_rank) / Wscale.unsqueeze(-1)
                    Wr = Wr.reshape(m, n)

                elif rcp == 'row':
                    Wr = utils.matmul_hadUt(
                        (utils.matmul_hadUt(W.T.to(device) * SV).T * SU).reshape(
                            m * args.tp_rank, n // args.tp_rank)).reshape(W.shape)
                    HRr = utils.matmul_hadUt(
                        (utils.matmul_hadUt((HR.to(device) * SU).reshape(
                            n * args.tp_rank, n // args.tp_rank)).reshape(n, n).T *
                         SU).reshape(n * args.tp_rank,
                                     n // args.tp_rank)).reshape(n, n)
                    Wscale = Wr.reshape(
                        m, args.tp_rank,
                        n // args.tp_rank).transpose(0, 1).reshape(
                            args.tp_rank, m * n // args.tp_rank).square().mean(
                                dim=-1).sqrt() / (cb.lut.to(
                                    torch.float64).square().mean().sqrt().float() *
                                                  args.scale_override)
                    Wr = Wr.reshape(m, args.tp_rank, n // args.tp_rank).transpose(
                        0, 1).reshape(args.tp_rank,
                                      m * n // args.tp_rank) / Wscale.unsqueeze(-1)
                    Wr = Wr.reshape(args.tp_rank, m,
                                    n // args.tp_rank).transpose(0, 1).reshape(m, n)
                else:
                    raise ValueError(f"Unknown rcp={rcp}")
            else:
                Wr = utils.matmul_hadUt(
                    utils.matmul_hadUt(W.T.to(device) * SV).T * SU)
                HRr = utils.matmul_hadUt(
                    utils.matmul_hadUt(HR.to(device) * SU).T * SU)

                Wscale = Wr.square().mean().sqrt() / (
                    cb.lut.to(torch.float64).square().mean().sqrt().float() *
                    args.scale_override)
                Wr /= Wscale

        elif args.incoh_mode == "harp":
            Wr0 = W.to(device) * SV[:, None] * SU[None, :]
            HR0 = HR.to(device) * SU[None, :] * SU[:, None]

            harp = _make_harp_from_args(
                n=n,
                m=m,
                device=device,
                args=args,
                rcp=rcp if args.split_for_tp else None,
                has_kernel=has_kernel,
            ).to(device)

            torch.cuda.empty_cache()
            with torch.enable_grad():
                harp.fit(HR0, Wr0, cb=cb, args=args)

            HRr, Wr = harp.transform(HR0, Wr0)

            if args.split_for_tp:
                if rcp == 'col':
                    Wscale = Wr.reshape(
                        args.tp_rank, m * n // args.tp_rank).square().mean(
                            dim=-1).sqrt() / (cb.lut.to(
                                torch.float64).square().mean().sqrt().float() *
                                              args.scale_override)
                    Wr = Wr.reshape(args.tp_rank,
                                    m * n // args.tp_rank) / Wscale.unsqueeze(-1)
                    Wr = Wr.reshape(m, n)

                elif rcp == 'row':
                    Wscale = Wr.reshape(
                        m, args.tp_rank,
                        n // args.tp_rank).transpose(0, 1).reshape(
                            args.tp_rank, m * n // args.tp_rank).square().mean(
                                dim=-1).sqrt() / (cb.lut.to(
                                    torch.float64).square().mean().sqrt().float() *
                                                  args.scale_override)
                    Wr = Wr.reshape(m, args.tp_rank, n // args.tp_rank).transpose(
                        0, 1).reshape(args.tp_rank,
                                      m * n // args.tp_rank) / Wscale.unsqueeze(-1)
                    Wr = Wr.reshape(args.tp_rank, m,
                                    n // args.tp_rank).transpose(0, 1).reshape(m, n)
                else:
                    raise ValueError(f"Unknown rcp={rcp}")
            else:
                Wscale = Wr.square().mean().sqrt() / (
                    cb.lut.to(torch.float64).square().mean().sqrt().float() *
                    args.scale_override)
                Wr /= Wscale

        else:
            raise ValueError(f"Unsupported incoh_mode={args.incoh_mode}")

        LRr, _ = utils.block_LDL(HRr, args.td_y)
        diag = torch.arange(n, device=LRr.device)
        LRr[diag, diag] = 0

        hatWr, Qidxs = ldlq.LDLQ(Wr, LRr, cb, args, for_kernel=has_kernel)

        Qidxs = Qidxs.cpu()
        packed = cb.pack_trellis(
            Qidxs.reshape(m // args.td_x, args.td_x, n // args.td_y,
                          args.td_y // args.V).transpose(1, 2).reshape(
                              -1, args.td_x * args.td_y // args.V))

        if has_kernel:
            packed = packed.view(torch.uint8).view(-1, 2).flip(
                (-1, )).reshape(m // 16 // 2, 2, n // 16 // 2, 2, 16 * 16 // 8,
                                args.K).permute(0, 2, 4, 3, 1, 5).flip(
                                    (-1, )).contiguous().flatten().view(
                                        torch.int16).reshape(packed.shape)
        else:
            packed = packed.view(torch.int16)

        if rcp == 'col':
            Wr = (Wr.reshape(args.tp_rank, m * n // args.tp_rank) *
                  Wscale.unsqueeze(-1)).reshape(m, n)
            hatWr = (hatWr.reshape(args.tp_rank, m * n // args.tp_rank) *
                     Wscale.unsqueeze(-1)).reshape(m, n)
        elif rcp == 'row':
            Wr = Wr.reshape(m, args.tp_rank, n // args.tp_rank).transpose(
                0, 1).reshape(args.tp_rank, -1) * Wscale.unsqueeze(-1)
            Wr = Wr.reshape(args.tp_rank, m,
                            n // args.tp_rank).transpose(0, 1).reshape(m, n)
            hatWr = hatWr.reshape(m, args.tp_rank,
                                  n // args.tp_rank).transpose(0, 1).reshape(
                                      args.tp_rank, -1) * Wscale.unsqueeze(-1)
            hatWr = hatWr.reshape(args.tp_rank, m,
                                  n // args.tp_rank).transpose(0, 1).reshape(
                                      m, n)
        else:
            Wr *= Wscale
            hatWr *= Wscale

        err = torch.trace(
            (Wr - hatWr) @ HRr @ (Wr - hatWr).T) / torch.trace(Wr @ HRr @ Wr.T)
        print(
            f'{idx}_{name} proxy err {err.item()} tr(WHW.T) {torch.trace(Wr @ HRr @ Wr.T)}'
        )

        save_path = f'{args.save_path}/{idx}_{name}.pt'

        rcp_int = 0
        if args.split_for_tp:
            rcp_int = 1 if rcp == 'row' else 2

        save_obj = {
            'trellis': packed.cpu(),
            'SU': SU.to(orig_dtype).cpu(),
            'SV': SV.to(orig_dtype).cpu(),
            'Wscale': Wscale,
            'proxy_err': err.item(),
            'tlut': cb.tlut.data.to(orig_dtype).cpu()
                    if hasattr(cb, 'tlut') and cb.tlut is not None else None,
            'rcp': rcp_int,
            'tp_rank': args.tp_rank,
            'incoh_mode': args.incoh_mode,
        }

        if args.incoh_mode == "harp":
            save_obj['harp_cfg'] = dict(getattr(args, "harp_cfg", {}) or {})
            save_obj['harp_state'] = {
                k: v.detach().cpu() for k, v in harp.state_dict().items()
            }

        torch.save(save_obj, save_path)

        module_elapsed_s = time.perf_counter() - module_t0
        module_calibration_times_s[name] = float(module_elapsed_s)
        glog.info(
            f'layer {idx} module {name} calibration time: {module_elapsed_s:.2f}s'
        )

        del HRr, Wr, hatWr, LRr, Qidxs
        utils.clean()

        q_linear = QuantizedLinear(
            n,
            m,
            args.td_x,
            args.td_y,
            args.L,
            args.K,
            args.V,
            args.tlut_bits,
            args.decode_mode,
            mode='train-recons' if args.ft_train_lut else 'train-fixW',
            dtype=orig_dtype,
            grad_ckpt=args.ft_grad_ckpt,
            incoh_mode=args.incoh_mode,
            harp_cfg=getattr(args, "harp_cfg", None),
            split_for_tp=args.split_for_tp,
            rcp=rcp if args.split_for_tp else None,
            tp_rank=args.tp_rank,
        )
        q_linear.trellis.copy_(packed)
        q_linear.SU.copy_(SU)
        q_linear.SV.copy_(SV)
        q_linear.rcp.copy_(torch.tensor(rcp_int))
        q_linear.tp_rank.copy_(torch.tensor(args.tp_rank))
        q_linear = q_linear.to(device).float()

        if args.incoh_mode == "harp":
            if q_linear.harp is None:
                raise RuntimeError(
                    "QuantizedLinear was created with incoh_mode='harp' but harp module is None"
                )
            q_linear.harp.load_state_dict(harp.state_dict(), strict=True)
            for p in q_linear.harp.parameters():
                p.requires_grad_(getattr(args, "ft_harp_lr", 0.0) > 0)

        del packed, SU, SV
        utils.clean()

        if rcp == 'row':
            q_linear.SU = nn.Parameter(
                (q_linear.SU.reshape(args.tp_rank, -1) *
                 Wscale.unsqueeze(-1)).reshape(q_linear.SU.shape),
                requires_grad=True)
            q_linear.SV = nn.Parameter(q_linear.SV, requires_grad=True)
        elif rcp == 'col':
            q_linear.SU = nn.Parameter(q_linear.SU, requires_grad=True)
            q_linear.SV = nn.Parameter(
                (q_linear.SV.reshape(args.tp_rank, -1) *
                 Wscale.unsqueeze(-1)).reshape(q_linear.SV.shape),
                requires_grad=True)
        else:
            q_linear.SU = nn.Parameter(q_linear.SU, requires_grad=True)
            q_linear.SV = nn.Parameter(q_linear.SV * Wscale,
                                       requires_grad=True)

        if q_linear.tlut is not None:
            q_linear.tlut.copy_(cb.tlut.data)
            q_linear.tlut.requires_grad = args.ft_train_lut

        split_attr = linear_attr.split('.')
        setattr(
            attrgetter('.'.join(split_attr[:-1]))(mixed_layer), split_attr[-1],
            q_linear)

        if do_ft:
            with torch.enable_grad():
                finetune_decoder_layer(
                    mixed_layer,
                    f'{idx}_{name}',
                    device,
                    train_dl,
                    valid_dl,
                    orig_dtype,
                    args,
                )

        cb = cb.cpu()
        utils.clean()

    for quant_i, (linear_attr, name, in_hess_name, out_hess_name,
                  rcp) in enumerate(quant_order):
        quant_linear = attrgetter(linear_attr)(mixed_layer)
        save_path = f'{args.save_path}/{idx}_{name}.pt'
        data = torch.load(save_path, weights_only=False)

        if rcp == 'row':
            data['SU'] = (
                ((quant_linear.SU.data).reshape(args.tp_rank, -1) /
                 data['Wscale'].to(quant_linear.SU.device).unsqueeze(-1)
                 ).reshape(quant_linear.SU.data.shape)).to(orig_dtype).cpu()
            data['SV'] = quant_linear.SV.data.to(orig_dtype).cpu()
        elif rcp == 'col':
            data['SU'] = quant_linear.SU.data.to(orig_dtype).cpu()
            data['SV'] = (
                ((quant_linear.SV.data).reshape(args.tp_rank, -1) /
                 data['Wscale'].to(quant_linear.SV.device).unsqueeze(-1)
                 ).reshape(quant_linear.SV.data.shape)).to(orig_dtype).cpu()
        else:
            data['SU'] = quant_linear.SU.data.to(orig_dtype).cpu()
            data['SV'] = (quant_linear.SV.data / data['Wscale'].to(
                quant_linear.SV.device)).to(orig_dtype).cpu()

        if quant_linear.tlut is not None:
            data['tlut'] = quant_linear.tlut.data.to(orig_dtype).cpu()

        if getattr(quant_linear, "incoh_mode", "had") == "harp":
            data['incoh_mode'] = 'harp'
            data['harp_cfg'] = quant_linear.harp_cfg
            data['harp_state'] = {
                k: v.detach().cpu()
                for k, v in quant_linear.harp.state_dict().items()
            }

        torch.save(data, save_path)

    mixed_layer = mixed_layer.to(orig_dtype).cpu()

    utils.clean()
    torch.set_grad_enabled(False)

    layer_block_calibration_time_s = sum(module_calibration_times_s.values())
    return {
        "layer_idx": int(idx),
        "module_calibration_times_s": {
            k: float(v) for k, v in module_calibration_times_s.items()
        },
        "layer_block_calibration_time_s": float(layer_block_calibration_time_s),
    }


def infer(args, end_dev, n_layers, in_q, out_q):
    with torch.no_grad():
        fake_dev_map = {
            'model.embed_tokens': 0,
            'model.rotary_emb': 0,
            'model.norm': end_dev - 1,
            'lm_head': end_dev - 1
        }
        per_dev = math.ceil(n_layers / end_dev)
        for i in range(n_layers):
            fake_dev_map[f'model.layers.{i}'] = (i + 1) // per_dev

        model = AutoModelForCausalLM.from_pretrained(args.base_model,
                                                     torch_dtype='auto',
                                                     device_map=fake_dev_map,
                                                     low_cpu_mem_usage=False)
        while True:
            data = in_q.get()
            if data is None:
                return
            out_q.put(
                model(data.to(0))['logits'][:, :-1].contiguous().softmax(
                    dim=-1).cpu())


def finetune_susv_e2e(quant_model, start_dev, devset, orig_dtype, args):

    in_q = mp.Queue()
    out_q = mp.Queue()
    p = mp.Process(target=infer,
                   args=(args, start_dev, len(quant_model.model.layers), in_q,
                         out_q))
    p.start()

    train_dl, valid_dl = utils.split_data(devset, devset, args)

    optim = torch.optim.Adam(_collect_ft_param_groups(quant_model, args))

    best_loss = utils.calculate_ce_loss_model(quant_model, valid_dl, start_dev,
                                              in_q, out_q)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best_sd = copy.deepcopy(quant_model.state_dict())
    glog.info(f'initial loss {best_loss}')
    worse_ct = 0
    for epoch in range(args.ft_epochs):
        for bidx, (source, _) in enumerate(train_dl):
            in_q.put(source)
            with torch.autocast(device_type='cuda',
                                dtype=orig_dtype,
                                enabled=True):
                output = quant_model(
                    source.to(start_dev))['logits'][:, :-1].contiguous()
                target = out_q.get().to(output.device)
                target = target.view(-1, target.shape[-1])
                loss = nn.CrossEntropyLoss()(output.view(-1, output.shape[-1]),
                                             target)
            scaler.scale(loss).backward()
            if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(
                    train_dl) - 1:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

        if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
            test_loss = utils.calculate_ce_loss_model(quant_model, valid_dl,
                                                      start_dev, in_q, out_q)
            if test_loss < best_loss:
                glog.info(
                    f'epoch {epoch} new loss {test_loss} old loss {best_loss} BETTER'
                )
                best_loss = test_loss
                best_sd = copy.deepcopy(quant_model.state_dict())
                worse_ct = 0
            else:
                glog.info(
                    f'epoch {epoch} new loss {test_loss} old loss {best_loss} WORSE'
                )
                worse_ct += 1
                if worse_ct >= args.ft_early_stop:
                    break

    in_q.put(None)
    p.join()
    with torch.no_grad():
        quant_model.load_state_dict(best_sd)
