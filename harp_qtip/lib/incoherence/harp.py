import copy
from typing import Dict, List, Optional, Tuple

import glog
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from .base import IncoherenceProcessor
from .harp_lib import HARPRotationPreconditioned
from .loss_utils import quantize_blocks_hard_chunked

_PERMUTE = torch.arange(256).reshape(2, 8, 2, 4, 2).permute(1, 3, 2, 0, 4).flatten()
_INV_PERMUTE = torch.zeros(256, dtype=torch.int64)
_INV_PERMUTE[_PERMUTE] = torch.arange(256)


class HARPProcessor(IncoherenceProcessor):
    """
    QTIP-specific HARP processor.
    """

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
        hbd_block: int = 16,
        chunk_size: int = 4096,
        q_recompute_every: int = 6,
        q_recompute_every_final: int = 2,
        q_refresh_parts: int = 8,
        early_stop_window: int = 0,
        early_stop_min_rel_improve: float = 0.01,

        split_for_tp: bool = False,
        rcp: Optional[str] = None,  # None / 'row' / 'col'
        tp_rank: int = 1,

        td_x: int = 16,
        td_y: int = 16,
        L: int = 16,
        K: int = 2,
        V: int = 2,
        tlut_bits: int = 9,
        decode_mode: str = "quantlut_sym",
        for_kernel: bool = False,
    ):
        super().__init__()
        self.n = int(n)
        self.m = int(m)
        self.device = str(device)

        self.steps = int(steps)
        self.strategy = str(strategy)

        self.lr_u = float(lr_u)
        self.lr_v = float(lr_v)
        self.grad_clip = float(grad_clip) if grad_clip is not None else None
        self.reg_theta = float(reg_theta)

        self.hbd_lambda = float(hbd_lambda)
        self.hbd_block = int(hbd_block)

        self.chunk_size = int(chunk_size)

        self.q_recompute_every = max(1, int(q_recompute_every))
        self.q_recompute_every_final = max(1, int(q_recompute_every_final))
        self.q_refresh_parts = max(1, int(q_refresh_parts))

        self.early_stop_window = int(early_stop_window)
        self.early_stop_min_rel_improve = float(early_stop_min_rel_improve)

        self.td_x = int(td_x)
        self.td_y = int(td_y)
        self.L = int(L)
        self.K = int(K)
        self.V = int(V)
        self.tlut_bits = int(tlut_bits)
        self.decode_mode = str(decode_mode)
        self.for_kernel = bool(for_kernel)

        self.split_for_tp = bool(split_for_tp and (tp_rank > 1) and (rcp is not None))
        self.rcp = None if not self.split_for_tp else str(rcp)
        self.tp_rank = int(tp_rank)

        if self.split_for_tp and self.rcp not in ("row", "col"):
            raise ValueError(f"split_for_tp=True but rcp={rcp}. Expected 'row' or 'col'.")

        if self.split_for_tp:
            if self.rcp == "row" and (self.n % self.tp_rank != 0):
                raise ValueError(
                    f"row-split HARP requires n divisible by tp_rank, got n={self.n}, tp_rank={self.tp_rank}"
                )
            if self.rcp == "col" and (self.m % self.tp_rank != 0):
                raise ValueError(
                    f"col-split HARP requires m divisible by tp_rank, got m={self.m}, tp_rank={self.tp_rank}"
                )

        if (self.m % self.td_x) != 0 or (self.n % self.td_y) != 0:
            raise ValueError(
                f"HARP proxy for QTIP requires m % td_x == 0 and n % td_y == 0, "
                f"got m={self.m}, td_x={self.td_x}, n={self.n}, td_y={self.td_y}"
            )

        common_kwargs = dict(
            base_b=int(harp_b),
            max_b=int(harp_max_b),
            stage_bs=harp_stage_bs,
            n_passes=int(max(1, harp_passes)),
            ordering_mode=str(ordering_mode),
            fixed_mixer=str(fixed_mixer),
            fixed_seed=int(fixed_seed),
            theta_init_scale=float(theta_init_scale),
            theta_clip=theta_clip,
            device=device,
            dtype=torch.float32,
            kron_fallback=bool(kron_fallback),
            use_givens_b2=bool(use_givens_b2),
        )

        self.harp_u: Optional[HARPRotationPreconditioned] = None
        self.harp_v: Optional[HARPRotationPreconditioned] = None
        self.harp_u_shards = nn.ModuleList()
        self.harp_v_shards = nn.ModuleList()

        if not self.split_for_tp:
            self.harp_v = HARPRotationPreconditioned(d=self.n, **common_kwargs)
            self.harp_u = HARPRotationPreconditioned(d=self.m, **common_kwargs)

        elif self.rcp == "row":
            # Full output transform U, shard-local input transform V
            self.harp_u = HARPRotationPreconditioned(d=self.m, **common_kwargs)
            n_shard = self.n // self.tp_rank
            for _ in range(self.tp_rank):
                self.harp_v_shards.append(
                    HARPRotationPreconditioned(d=n_shard, **common_kwargs)
                )

        elif self.rcp == "col":
            # Full input transform V, shard-local output transform U
            self.harp_v = HARPRotationPreconditioned(d=self.n, **common_kwargs)
            m_shard = self.m // self.tp_rank
            for _ in range(self.tp_rank):
                self.harp_u_shards.append(
                    HARPRotationPreconditioned(d=m_shard, **common_kwargs)
                )

    def _modules_for_u(self) -> List[nn.Module]:
        if self.harp_u is not None:
            return [self.harp_u]
        return list(self.harp_u_shards)

    def _modules_for_v(self) -> List[nn.Module]:
        if self.harp_v is not None:
            return [self.harp_v]
        return list(self.harp_v_shards)

    @staticmethod
    def _set_req_grad(mod: nn.Module, flag: bool):
        for p in mod.parameters(recurse=True):
            if isinstance(p, nn.Parameter) and p.numel() > 0:
                p.requires_grad_(flag)

    def trainable_param_groups(self) -> Dict[str, List[nn.Parameter]]:
        def collect(mods: List[nn.Module]) -> List[nn.Parameter]:
            out = []
            for mod in mods:
                out.extend([
                    p for p in mod.parameters(recurse=True)
                    if isinstance(p, nn.Parameter) and p.numel() > 0
                ])
            return out

        return {
            "harp_u": collect(self._modules_for_u()),
            "harp_v": collect(self._modules_for_v()),
        }

    def _apply_input_transform_right(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply right-multiplication by V:
          X -> X V
        X: [*, n]
        """
        if not self.split_for_tp:
            return self.harp_v.transform(X)

        if self.rcp == "col":
            return self.harp_v.transform(X)

        # row split: block-diagonal V over input shards
        n_shard = self.n // self.tp_rank
        parts = []
        for i, rot in enumerate(self.harp_v_shards):
            xi = X[..., i * n_shard:(i + 1) * n_shard]
            parts.append(rot.transform(xi))
        return torch.cat(parts, dim=-1)

    def _apply_input_inverse_right(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply right-multiplication by V^T:
          X -> X V^T
        """
        if not self.split_for_tp:
            return self.harp_v.inverse_transform(X)

        if self.rcp == "col":
            return self.harp_v.inverse_transform(X)

        # row split: block-diagonal inverse over input shards
        n_shard = self.n // self.tp_rank
        parts = []
        for i, rot in enumerate(self.harp_v_shards):
            xi = X[..., i * n_shard:(i + 1) * n_shard]
            parts.append(rot.inverse_transform(xi))
        return torch.cat(parts, dim=-1)

    def _apply_output_transform_left(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply left-multiplication by U^T:
          X -> U^T X
        X: [m, *]
        """
        if not self.split_for_tp:
            return self.harp_u.transform(X.T).T

        if self.rcp == "row":
            return self.harp_u.transform(X.T).T

        # col split: block-diagonal U over output shards
        m_shard = self.m // self.tp_rank
        parts = []
        for i, rot in enumerate(self.harp_u_shards):
            xi = X[i * m_shard:(i + 1) * m_shard, :]
            parts.append(rot.transform(xi.T).T)
        return torch.cat(parts, dim=0)

    def _apply_output_inverse_left(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply left-multiplication by U:
          X -> U X
        """
        if not self.split_for_tp:
            return self.harp_u.inverse_transform(X.T).T

        if self.rcp == "row":
            return self.harp_u.inverse_transform(X.T).T

        # col split: block-diagonal inverse over output shards
        m_shard = self.m // self.tp_rank
        parts = []
        for i, rot in enumerate(self.harp_u_shards):
            xi = X[i * m_shard:(i + 1) * m_shard, :]
            parts.append(rot.inverse_transform(xi.T).T)
        return torch.cat(parts, dim=0)

    def _transform_H_two_sided_exact(self, H: torch.Tensor) -> torch.Tensor:
        """
        Exact H_tilde = V^T H V
        """
        H_right = self._apply_input_transform_right(H) # H V
        H_tilde_T = self._apply_input_transform_right(H_right.T) # (V^T H V)^T
        return H_tilde_T.T

    @torch.no_grad()
    def _transform_H_diag_only_gpu_no_grad(self, H: torch.Tensor) -> torch.Tensor:
        """
        Exact diagonal of H_tilde = V^T H V, without building an autograd graph.
        """
        H_right = self._apply_input_transform_right(H)
        H_tilde_T = self._apply_input_transform_right(H_right.T)
        h_diag = H_tilde_T.diagonal().clone()
        del H_right, H_tilde_T
        return h_diag

    def transform(self, H: torch.Tensor, W: torch.Tensor):
        """
        Returns:
          H_tilde = V^T H V
          W_tilde = U^T W V
        """
        with torch.no_grad():
            H_tilde = self._transform_H_two_sided_exact(H)
            W_right = self._apply_input_transform_right(W)
            W_tilde = self._apply_output_transform_left(W_right)
        return H_tilde, W_tilde

    def inverse_transform_weights(self, W_tilde: torch.Tensor):
        """
        Invert:
          W_tilde = U^T W V
        so:
          W = U W_tilde V^T
        """
        tmp = self._apply_output_inverse_left(W_tilde)
        W = self._apply_input_inverse_right(tmp)
        return W

    def forward_pre_transform(self, x: torch.Tensor):
        """
        Activation-side input transform:
          x -> x V
        """
        return self._apply_input_transform_right(x)

    def forward_post_transform(self, y: torch.Tensor):
        """
        Activation-side output inverse transform:
          y_tilde -> y_tilde U^T
        """
        if not self.split_for_tp:
            return self.harp_u.inverse_transform(y)

        if self.rcp == "row":
            return self.harp_u.inverse_transform(y)

        # col split: block-diagonal inverse over output shards
        m_shard = self.m // self.tp_rank
        parts = []
        for i, rot in enumerate(self.harp_u_shards):
            yi = y[..., i * m_shard:(i + 1) * m_shard]
            parts.append(rot.inverse_transform(yi))
        return torch.cat(parts, dim=-1)

    def _normalize_W(self, W_tilde: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        if not self.split_for_tp:
            return W_tilde / scale

        if self.rcp == "col":
            Wn = W_tilde.reshape(self.tp_rank, self.m * self.n // self.tp_rank)
            Wn = Wn / scale.unsqueeze(-1)
            return Wn.reshape(self.m, self.n)

        # row split: shard over input dimension
        Wn = W_tilde.reshape(self.m, self.tp_rank, self.n // self.tp_rank).transpose(0, 1)
        Wn = Wn.reshape(self.tp_rank, self.m * self.n // self.tp_rank)
        Wn = Wn / scale.unsqueeze(-1)
        Wn = Wn.reshape(self.tp_rank, self.m, self.n // self.tp_rank).transpose(0, 1)
        return Wn.reshape(self.m, self.n)

    def _compute_proxy_scale(self, W_tilde: torch.Tensor, cb, args) -> torch.Tensor:
        lut_norm = cb.lut.to(torch.float64).square().mean().sqrt().float()
        scale_override = float(args.scale_override)

        if not self.split_for_tp:
            scale = W_tilde.pow(2).mean().sqrt()
            scale = scale / (lut_norm * scale_override)
            return scale.detach()

        if self.rcp == "col":
            scale = W_tilde.reshape(self.tp_rank, self.m * self.n // self.tp_rank)
            scale = scale.square().mean(dim=-1).sqrt()
            scale = scale / (lut_norm * scale_override)
            return scale.detach()

        # row split
        scale = W_tilde.reshape(
            self.m, self.tp_rank, self.n // self.tp_rank
        ).transpose(0, 1).reshape(self.tp_rank, self.m * self.n // self.tp_rank)
        scale = scale.square().mean(dim=-1).sqrt()
        scale = scale / (lut_norm * scale_override)
        return scale.detach()

    def _reshape_W_to_blocks(self, W_norm: torch.Tensor) -> torch.Tensor:
        """
        QTIP block layout:
          [m, n] -> [num_blocks, td_x * td_y]
        """
        return (
            W_norm.view(self.m // self.td_x, self.td_x, self.n // self.td_y, self.td_y)
                  .transpose(1, 2)
                  .reshape(-1, self.td_x * self.td_y)
                  .contiguous()
        )

    def _quantize_block_matrix(self, blocks: torch.Tensor, cb) -> torch.Tensor:
        """
        Quantize a [num_blocks, td_x * td_y] matrix using the proxy backend.
        """
        if self.for_kernel:
            if self.td_x * self.td_y != 256:
                raise ValueError(
                    f"for_kernel=True requires td_x * td_y == 256, got {self.td_x} * {self.td_y}"
                )
            perm = _PERMUTE.to(device=blocks.device)
            blocks = blocks[:, perm]

        q_blocks = quantize_blocks_hard_chunked(
            blocks,
            codebook=cb,
            chunk=self.chunk_size,
        )

        if self.for_kernel:
            inv_perm = _INV_PERMUTE.to(device=q_blocks.device)
            q_blocks = q_blocks[:, inv_perm]

        return q_blocks.detach()

    @torch.no_grad()
    def _compute_proxy_quant_cache_full(self, W_tilde: torch.Tensor, cb, args):
        """
        Build the full cached proxy target:
          cached_scale
          cached_q_blocks: [num_blocks, td_x * td_y]
        """
        scale = self._compute_proxy_scale(W_tilde, cb, args)
        W_norm = self._normalize_W(W_tilde.detach(), scale).to(torch.float32)
        blocks = self._reshape_W_to_blocks(W_norm)
        q_blocks = self._quantize_block_matrix(blocks, cb)
        return scale, q_blocks

    @torch.no_grad()
    def _refresh_proxy_quant_cache_partition(
        self,
        W_tilde: torch.Tensor,
        cb,
        cached_scale: torch.Tensor,
        cached_q_blocks: torch.Tensor,
        part_idx: int,
        num_parts: int,
    ) -> torch.Tensor:
        if num_parts <= 1:
            return cached_q_blocks

        W_norm = self._normalize_W(W_tilde.detach(), cached_scale).to(torch.float32)
        blocks = self._reshape_W_to_blocks(W_norm)

        num_blocks = blocks.shape[0]
        idx = torch.arange(part_idx, num_blocks, num_parts, device=blocks.device)
        if idx.numel() == 0:
            return cached_q_blocks

        q_part = self._quantize_block_matrix(blocks.index_select(0, idx), cb)
        cached_q_blocks.index_copy_(0, idx, q_part)
        return cached_q_blocks

    def _block_column_weights(self, h_diag: torch.Tensor, device, dtype) -> torch.Tensor:
        """
        Convert h_diag [n] into per-block column weights [num_blocks, td_y].
        """
        h_diag = h_diag.abs().detach().to(device=device, dtype=dtype)
        h_diag = h_diag / (h_diag.mean() + 1e-8)

        # [n_blk, td_y]
        h_cols = h_diag.view(self.n // self.td_y, self.td_y)

        # expand over row-blocks -> [m_blk, n_blk, td_y] -> [num_blocks, td_y]
        h_cols = h_cols.unsqueeze(0).expand(self.m // self.td_x, -1, -1).reshape(-1, self.td_y)
        return h_cols

    def _proxy_loss_from_cached_blocks(
        self,
        W_tilde: torch.Tensor,
        h_diag: torch.Tensor,
        cached_q_blocks: torch.Tensor,
        cached_scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full-matrix proxy loss, but against a cached block target.
        """
        W_norm = self._normalize_W(W_tilde, cached_scale).to(torch.float32)
        blocks = self._reshape_W_to_blocks(W_norm)

        err = (blocks - cached_q_blocks).view(-1, self.td_x, self.td_y)
        h_cols = self._block_column_weights(h_diag, device=err.device, dtype=err.dtype)

        weighted_error = (err ** 2) * h_cols.unsqueeze(1)
        return weighted_error.mean()

    def _disable_hbd_for_this_layer(self) -> bool:
        return self.n >= 28672

    def _h_blockdiag_loss(self, H_tilde: torch.Tensor, block: int = 16) -> torch.Tensor:
        """
        Loss = sum_{i!=j} ||H_ij||_F^2 over block partitions.
        """
        n = H_tilde.shape[0]
        b = int(block)
        if n % b != 0:
            off = H_tilde - torch.diag(torch.diagonal(H_tilde))
            return off.pow(2).mean()

        nb = n // b
        H4 = H_tilde.view(nb, b, nb, b).permute(0, 2, 1, 3)  # [nb, nb, b, b]
        total = H4.pow(2).sum()
        diag_blocks = torch.diagonal(H4, dim1=0, dim2=1)      # [nb, b, b]
        diag_sum = diag_blocks.pow(2).sum()
        off_sum = total - diag_sum
        return off_sum / H4.numel()

    def _should_early_stop(self, loss_history: List[float]):
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

    def _q_schedule_state(self, step: int) -> Tuple[int, int, int]:
        """
        Refresh schedule for the cached proxy target.
        """
        start_interval = max(self.q_recompute_every, self.q_recompute_every_final)
        final_interval = self.q_recompute_every_final
        start_parts = max(1, self.q_refresh_parts)

        if self.steps <= 1:
            return final_interval, 1, 3

        progress = float(step) / float(max(self.steps - 1, 1))

        if progress < 0.50:
            return start_interval, start_parts, 0

        if progress < 0.80:
            mid_interval = max(final_interval + 1, int(round((start_interval + final_interval) / 2.0)))
            mid_parts = max(2, start_parts // 2)
            return mid_interval, mid_parts, 1

        if progress < 0.95:
            return final_interval, 2, 2

        return final_interval, 1, 3

    def fit(self, H: torch.Tensor, W: torch.Tensor, cb=None, args=None):
        if cb is None:
            raise ValueError("HARPProcessor.fit requires QTIP codebook `cb`.")
        if args is None:
            raise ValueError("HARPProcessor.fit requires args.")
        if torch.is_inference_mode_enabled():
            raise RuntimeError("HARPProcessor.fit cannot run under torch.inference_mode().")

        H = H.detach()
        W = W.detach()

        cb = copy.deepcopy(cb).to(device=W.device)

        self.train(True)
        for mod in self._modules_for_u():
            mod.train(True)
            self._set_req_grad(mod, True)
        for mod in self._modules_for_v():
            mod.train(True)
            self._set_req_grad(mod, True)

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
        cached_q_blocks = None

        next_refresh_step = 0
        refresh_part_idx = 0
        last_phase = None

        loss_history: List[float] = []
        last_pbar_step = 0

        for step in range(self.steps):
            optimizer.zero_grad(set_to_none=True)

            if self.strategy != "proxy":
                raise ValueError(f"Unsupported HARP strategy: {self.strategy}")

            W_right = self._apply_input_transform_right(W)
            W_tilde = self._apply_output_transform_left(W_right)

            H_tilde = None
            if need_hbd:
                H_tilde = self._transform_H_two_sided_exact(H)
                h_diag = H_tilde.diagonal()
            else:
                h_diag = self._transform_H_diag_only_gpu_no_grad(H)

            refresh_interval, refresh_parts, phase = self._q_schedule_state(step)

            force_full_refresh = (cached_q_blocks is None) or (phase != last_phase)

            if force_full_refresh or (step >= next_refresh_step):
                if force_full_refresh or (refresh_parts <= 1):
                    cached_scale, cached_q_blocks = self._compute_proxy_quant_cache_full(
                        W_tilde, cb, args
                    )
                    refresh_part_idx = 0
                else:
                    cached_q_blocks = self._refresh_proxy_quant_cache_partition(
                        W_tilde=W_tilde,
                        cb=cb,
                        cached_scale=cached_scale,
                        cached_q_blocks=cached_q_blocks,
                        part_idx=refresh_part_idx,
                        num_parts=refresh_parts,
                    )
                    refresh_part_idx = (refresh_part_idx + 1) % refresh_parts

                next_refresh_step = step + refresh_interval
                last_phase = phase

            loss = self._proxy_loss_from_cached_blocks(
                W_tilde=W_tilde,
                h_diag=h_diag,
                cached_q_blocks=cached_q_blocks,
                cached_scale=cached_scale,
            )

            if need_hbd:
                loss = loss + self.hbd_lambda * self._h_blockdiag_loss(
                    H_tilde, block=self.hbd_block
                )

            if self.reg_theta > 0:
                reg = 0.0
                for p in params_u + params_v:
                    reg = reg + p.float().pow(2).mean()
                loss = loss + self.reg_theta * reg

            loss.backward()

            if self.grad_clip is not None and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params_u + params_v, self.grad_clip)

            optimizer.step()

            loss_val = float(loss.item())
            loss_history.append(loss_val)

            cur_step = step + 1
            if (cur_step - last_pbar_step) >= 50 or cur_step == self.steps:
                pbar.set_postfix_str(
                    f"loss={loss_val:.3e}, qint={refresh_interval}, qparts={refresh_parts}",
                    refresh=False,
                )
                pbar.update(cur_step - last_pbar_step)
                last_pbar_step = cur_step

            should_stop, stop_info = self._should_early_stop(loss_history)
            if should_stop:
                prev_best, recent_best, rel_improve = stop_info
                glog.info(
                    f"HARP early stopping at step {step + 1}: "
                    f"prev_best={prev_best:.4e}, recent_best={recent_best:.4e}, "
                    f"rel_improve={100.0 * rel_improve:.2f}%"
                )
                break

        pbar.close()

    def get_storage_bits(self):
        bits = 0
        for mod in self._modules_for_u():
            if hasattr(mod, "get_storage_bits"):
                bits += int(mod.get_storage_bits())
        for mod in self._modules_for_v():
            if hasattr(mod, "get_storage_bits"):
                bits += int(mod.get_storage_bits())
        return bits

    @torch.no_grad()
    def quantize_theta_8bit(self, scale_dtype: torch.dtype = torch.float16):
        for mod in self._modules_for_u():
            if hasattr(mod, "quantize_theta_8bit"):
                mod.quantize_theta_8bit(scale_dtype=scale_dtype)
        for mod in self._modules_for_v():
            if hasattr(mod, "quantize_theta_8bit"):
                mod.quantize_theta_8bit(scale_dtype=scale_dtype)