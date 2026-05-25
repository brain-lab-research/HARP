import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm

from .base import IncoherenceProcessor
from .loss_utils import quantize_block_hard_ste_chunked
from .harp_lib import HARPRotationPreconditioned
from lib.utils.math_utils import block_LDL
import copy
import glog


class HARPProcessor(IncoherenceProcessor):
    def __init__(
        self,
        n: int,
        m: int,
        harp_b: int = 8,
        harp_max_b: int = 8,
        harp_stage_bs: Optional[List[int]] = None,
        harp_passes: int = 1,
        ordering_mode: str = "stride",
        device: str = "cuda",
        steps: int = 0,
        kron_fallback: bool = False,
        use_givens_b2: bool = False,
        theta_init_scale: float = 0.0,
        theta_clip: Optional[float] = None,
        fixed_mixer: str = "had_or_qr",
        fixed_seed: int = 0,
        
        strategy: str = "proxy",
        lr_u: float = 3e-2,
        lr_v: float = 3e-2,
        grad_clip: Optional[float] = None,
        reg_theta: float = 0.0,
        hbd_lambda: float = 0.1,
        hbd_block: int = 8,
        chunk_size: int = 131072,
        q_recompute_every: int = 1,
        early_stop_window: int = 0,
        early_stop_min_rel_improve: float = 0.01,
    ):
        super().__init__()
        self.n = int(n)
        self.m = int(m)
        self.device = str(device)
        self.steps = int(steps)
        self.strategy = str(strategy)
        self.ordering_mode = str(ordering_mode)

        self.lr_u = float(lr_u)
        self.lr_v = float(lr_v)
        self.grad_clip = float(grad_clip) if grad_clip is not None else None
        self.reg_theta = float(reg_theta)

        self.hbd_lambda = float(hbd_lambda)
        self.hbd_block = int(hbd_block)
        self.chunk_size = int(chunk_size)

        self.q_recompute_every = max(1, int(q_recompute_every))
        self.early_stop_window = int(early_stop_window)
        self.early_stop_min_rel_improve = float(early_stop_min_rel_improve)

        self.harp_v = HARPRotationPreconditioned(
            d=self.n,
            base_b=int(harp_b),
            max_b=int(harp_max_b),
            stage_bs=harp_stage_bs,
            n_passes=int(max(1, harp_passes)),
            ordering_mode=self.ordering_mode,
            fixed_mixer=fixed_mixer,
            fixed_seed=fixed_seed,
            theta_init_scale=theta_init_scale,
            theta_clip=theta_clip,
            device=device,
            dtype=torch.float32,
            kron_fallback=kron_fallback,
            use_givens_b2=use_givens_b2
        )
        self.harp_u = HARPRotationPreconditioned(
            d=self.m,
            base_b=int(harp_b),
            max_b=int(harp_max_b),
            stage_bs=harp_stage_bs,
            n_passes=int(max(1, harp_passes)),
            ordering_mode=self.ordering_mode,
            fixed_mixer=fixed_mixer,
            fixed_seed=fixed_seed,
            theta_init_scale=theta_init_scale,
            theta_clip=theta_clip,
            device=device,
            dtype=torch.float32,
            kron_fallback=kron_fallback,
            use_givens_b2=use_givens_b2
        )

    def _auto_h_chunk_rows(self, H: torch.Tensor) -> Optional[int]:
        n = H.shape[0]
        if n >= 28672:
            return 14336
        return None

    def _transform_rows_chunked(self, rot, X: torch.Tensor, chunk_rows: int | None):
        if chunk_rows is None or X.shape[0] <= chunk_rows:
            return rot.transform(X)
        return torch.cat(
            [rot.transform(X[s:s + chunk_rows]) for s in range(0, X.shape[0], chunk_rows)],
            dim=0,
        )

    def _transform_H_two_sided_exact(self, H: torch.Tensor) -> torch.Tensor:
        chunk_rows = self._auto_h_chunk_rows(H)
        if chunk_rows is None:
            H_right = self.harp_v.transform(H)
            return self.harp_v.transform(H_right.T).T

        H_right = self._transform_rows_chunked(self.harp_v, H, chunk_rows)
        H_tilde_T = self._transform_rows_chunked(self.harp_v, H_right.T, chunk_rows)
        return H_tilde_T.T
    
    @torch.no_grad()
    def _transform_H_diag_only_gpu_no_grad(self, H: torch.Tensor) -> torch.Tensor:
        """
        Exact diagonal of H_tilde = V^T H V.
        GPU-only, no CPU offload, no autograd graph.
        """
        H_right = self.harp_v.transform(H) # H @ V
        H_tilde_T = self.harp_v.transform(H_right.T) # (V^T H V)^T
        h_diag = H_tilde_T.diagonal().clone()
        del H_right, H_tilde_T
        torch.cuda.empty_cache()
        return h_diag

    def transform(self, H: torch.Tensor, W: torch.Tensor):
        """
        Returns:
          H_tilde = V^T H V
          W_tilde = U^T W V
        """
        with torch.no_grad():
            H_tilde = self._transform_H_two_sided_exact(H)

            W_right = self.harp_v.transform(W)
            W_tilde = self.harp_u.transform(W_right.T).T
        return H_tilde, W_tilde

    def inverse_transform_weights(self, W_tilde: torch.Tensor):
        """
        Invert W_tilde = U^T W V -> W = U W_tilde V^T
        """
        tmp = self.harp_u.inverse_transform(W_tilde.T).T
        W = self.harp_v.inverse_transform(tmp)
        return W

    def forward_pre_transform(self, x: torch.Tensor):
        return self.harp_v.transform(x)

    def forward_post_transform(self, y: torch.Tensor):
        return self.harp_u.inverse_transform(y)
    
    def _compute_proxy_scale(self, W_tilde, codebook, args):
        scale = (W_tilde.pow(2).mean().sqrt() + 1e-8)
        if getattr(args, "scale_override", -1) and args.scale_override > 0:
            scale = scale / float(args.scale_override)
        else:
            scale = scale / float(codebook.opt_scale)
        return scale.detach()

    def _proxy_quant_kwargs(self, args):
        quant_kwargs = {}
        resid_scale_override = getattr(args, "resid_scale_override", -1) if args is not None else -1
        if resid_scale_override is not None and resid_scale_override > 0:
            quant_kwargs["resid_scale_override"] = float(resid_scale_override)
        return quant_kwargs

    def _compute_proxy_quant_target(self, W_tilde, codebook, args):
        scale = self._compute_proxy_scale(W_tilde, codebook, args)
        W_norm = (W_tilde / scale).to(torch.float32)

        m_dim, n_dim = W_norm.shape
        codesz = int(codebook.codesz)

        pad_len = (codesz - (n_dim % codesz)) % codesz
        W_in = F.pad(W_norm, (0, pad_len)) if pad_len else W_norm
        W_flat = W_in.reshape(-1, codesz).contiguous()

        W_q_flat = quantize_block_hard_ste_chunked(
            W_flat,
            codebook,
            chunk=self.chunk_size,
            **self._proxy_quant_kwargs(args),
        ).detach()

        W_q = W_q_flat.view(W_in.shape)[:, :n_dim].detach()
        return scale, W_q

    def _proxy_loss_from_target(self, W_tilde, h_diag, W_q, scale):
        W_norm = (W_tilde / scale).to(torch.float32)
        error = W_norm - W_q

        h_diag = h_diag.abs().detach().to(torch.float32)
        h_diag = h_diag / (h_diag.mean() + 1e-8)

        weighted_error = (error ** 2) * h_diag.unsqueeze(0)
        return torch.mean(weighted_error)

    def _should_early_stop(self, loss_history):
        w = self.early_stop_window
        if w <= 0 or len(loss_history) <= w:
            return False, None

        prev_best = min(loss_history[:-w])
        recent_best = min(loss_history[-w:])
        rel_improve = (prev_best - recent_best) / max(abs(prev_best), 1e-12)

        return rel_improve < self.early_stop_min_rel_improve, (
            prev_best,
            recent_best,
            rel_improve,
        )

    def _proxy_loss_from_hdiag(self, W_tilde, h_diag, codebook, args):
        scale, W_q = self._compute_proxy_quant_target(W_tilde, codebook, args)
        return self._proxy_loss_from_target(W_tilde, h_diag, W_q, scale)

    def _proxy_loss(self, W_tilde, H_tilde, codebook, args):
        return self._proxy_loss_from_hdiag(
            W_tilde, H_tilde.diagonal(), codebook, args
        )
    
    def _disable_hbd_for_this_layer(self) -> bool:
        return self.n >= 24672

    def _h_blockdiag_loss(self, H_tilde: torch.Tensor, block: int = 8) -> torch.Tensor:
        """
        Loss = sum_{i!=j} ||H_ij||_F^2 over block partitions.
        """
        n = H_tilde.shape[0]
        b = int(block)
        if n % b != 0:
            off = H_tilde - torch.diag(torch.diagonal(H_tilde))
            return off.pow(2).mean()

        nb = n // b
        H4 = H_tilde.view(nb, b, nb, b).permute(0, 2, 1, 3) # [nb, nb, b, b]
        total = H4.pow(2).sum()
        diag_blocks = torch.diagonal(H4, dim1=0, dim2=1) # [nb, b, b]
        diag_sum = diag_blocks.pow(2).sum()
        off_sum = total - diag_sum
        denom = H4.numel()
        return off_sum / denom

    @staticmethod
    def _set_req_grad(mod: nn.Module, flag: bool):
        for p in mod.parameters(recurse=True):
            if isinstance(p, nn.Parameter) and p.numel() > 0:
                p.requires_grad_(flag)

    def trainable_param_groups(self) -> Dict[str, List[nn.Parameter]]:
        def collect(mod: nn.Module) -> List[nn.Parameter]:
            return [p for p in mod.parameters(recurse=True) if isinstance(p, nn.Parameter) and p.numel() > 0]
        return {"harp_u": collect(self.harp_u), "harp_v": collect(self.harp_v)}

    def fit(self, H: torch.Tensor, W: torch.Tensor, codebook=None, x_samples: Optional[torch.Tensor] = None, args=None):
        if codebook is None:
            raise ValueError("HARPProcessor.fit requires codebook.")
        if torch.is_inference_mode_enabled():
            raise RuntimeError("HARPProcessor fit can't run under torch.inference_mode().")

        H = H.detach()
        W = W.detach()
        if x_samples is not None:
            x_samples = x_samples.detach()

        codebook = copy.deepcopy(codebook).to(device=W.device, dtype=torch.float32)

        self.train(True)
        self.harp_u.train(True)
        self.harp_v.train(True)
        self._set_req_grad(self.harp_u, True)
        self._set_req_grad(self.harp_v, True)

        groups = self.trainable_param_groups()
        params_u = groups["harp_u"]
        params_v = groups["harp_v"]
        if len(params_u) == 0 or len(params_v) == 0:
            raise RuntimeError("No HARP parameters to fit.")

        optimizer = optim.Adam(
            [{"params": params_v, "lr": self.lr_v},
             {"params": params_u, "lr": self.lr_u}]
        )

        need_hbd = (self.hbd_lambda > 0) and (not self._disable_hbd_for_this_layer())

        pbar = tqdm(desc=f"HARP({self.strategy})", total=self.steps, leave=False)

        cached_scale = None
        cached_q_target = None
        loss_history = []

        for step in range(self.steps):
            optimizer.zero_grad(set_to_none=True)

            H_tilde = None

            refresh_q = (
                cached_q_target is None
                or ((step % self.q_recompute_every) == 0)
            )

            if self.strategy == "proxy" and not need_hbd:
                h_diag = self._transform_H_diag_only_gpu_no_grad(H)

                W_right = self.harp_v.transform(W)
                W_tilde = self.harp_u.transform(W_right.T).T

                if refresh_q:
                    cached_scale, cached_q_target = self._compute_proxy_quant_target(
                        W_tilde, codebook, args=args
                    )

                loss = self._proxy_loss_from_target(
                    W_tilde, h_diag, cached_q_target, cached_scale
                )

            else:
                W_right = self.harp_v.transform(W)
                W_tilde = self.harp_u.transform(W_right.T).T

                if self.strategy == "proxy":
                    H_tilde = self._transform_H_two_sided_exact(H)

                    if refresh_q:
                        cached_scale, cached_q_target = self._compute_proxy_quant_target(
                            W_tilde, codebook, args=args
                        )

                    loss = self._proxy_loss_from_target(
                        W_tilde, H_tilde.diagonal(), cached_q_target, cached_scale
                    )
                else:
                    raise ValueError(f"Unsupported HARP strategy: {self.strategy}")

                if self.hbd_lambda > 0:
                    if H_tilde is None:
                        H_tilde = self._transform_H_two_sided_exact(H)
                    loss = loss + self.hbd_lambda * self._h_blockdiag_loss(H_tilde, block=self.hbd_block)


            # optional L2 regularization
            if self.reg_theta > 0:
                reg = 0.0
                for p in params_v + params_u:
                    reg = reg + p.float().pow(2).mean()
                loss = loss + self.reg_theta * reg

            loss.backward()
            loss_history.append(float(loss.detach().cpu()))

            if self.grad_clip is not None and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params_v + params_u, self.grad_clip)

            optimizer.step()

            if (step + 1) % 50 == 0:
                pbar.set_postfix_str(f"loss={loss.item():.3e}", refresh=False)
                pbar.update(50)
            
            should_stop, stop_info = self._should_early_stop(loss_history)
            if should_stop:
                prev_best, recent_best, rel_improve = stop_info
                glog.info(
                    f'HARP early stopping triggered at step {step + 1}: '
                    f'prev_best={prev_best:.4e}, recent_best={recent_best:.4e}, '
                    f'rel_improve={100.0 * rel_improve:.2f}%'
                )
                break

        pbar.close()

    def get_storage_bits(self):
        return self.harp_u.get_storage_bits() + self.harp_v.get_storage_bits()
    
    @torch.no_grad()
    def quantize_theta_8bit(self, scale_dtype: torch.dtype = torch.float16):
        if hasattr(self.harp_u, "quantize_theta_8bit"):
            self.harp_u.quantize_theta_8bit(scale_dtype=scale_dtype)
        if hasattr(self.harp_v, "quantize_theta_8bit"):
            self.harp_v.quantize_theta_8bit(scale_dtype=scale_dtype)
