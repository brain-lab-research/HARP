from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import torch
import torch.nn as nn
from typing import Optional

def _is_pow2(n: int) -> bool:
    return (n > 0) and ((n & (n - 1)) == 0)


class KronHadKFallbackRotation(nn.Module):
    """
    Kronecker fallback wrapper:

      d = K * n2, where get_hadK(d) returns hadK (KxK) and K, and n2 is pow2.

    Transform (row-vector convention):
      x: [B, d] -> view [B, K, n2]
      apply axis_pow2 on n2 for each of the K slices
      scale by 1/sqrt(K)
      mix across K using hadK: out[b,i,n] = sum_j hadK[i,j] * curr[b,j,n]
    """
    def __init__(
        self,
        d: int,
        axis_pow2: nn.Module,                 
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.d = int(d)
        self.axis_pow2 = axis_pow2

        from lib.utils.matmul_had import get_hadK  # type: ignore

        hadK, K = get_hadK(self.d, transpose=True)
        self.K = int(K)
        if self.d % self.K != 0:
            raise ValueError(f"get_hadK returned incompatible K={self.K} for d={self.d}")

        self.n2 = int(self.d // self.K)
        if not _is_pow2(self.n2):
            raise ValueError(f"Kron fallback requires n2 pow2, got n2={self.n2} (d={self.d}, K={self.K})")

        if self.K > 1:
            if hadK is None:
                raise RuntimeError("Expected hadK for K>1")
            self.register_buffer("hadK", hadK.to(torch.float32), persistent=False)
            self.register_buffer("invK", torch.tensor(1.0 / float(self.K), dtype=torch.float32), persistent=False)
            self.register_buffer("scaleK", torch.tensor(1.0 / math.sqrt(self.K), dtype=torch.float32), persistent=False)
        else:
            self.register_buffer("hadK", torch.empty(0, 0), persistent=False)
            self.register_buffer("invK", torch.tensor(1.0, dtype=torch.float32), persistent=False)
            self.register_buffer("scaleK", torch.tensor(1.0, dtype=torch.float32), persistent=False)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x = x.reshape(-1, self.d)
        B = x.shape[0]

        if self.K == 1:
            out = self.axis_pow2.transform(x)
            return out.view(orig_shape)

        curr = x.view(B, self.K, self.n2)

        flat = curr.reshape(B * self.K, self.n2)
        flat = self.axis_pow2.transform(flat)
        curr = flat.view(B, self.K, self.n2)

        curr = curr * self.scaleK.to(device=curr.device, dtype=curr.dtype)

        H = self.hadK.to(device=curr.device, dtype=curr.dtype)
        out = torch.einsum("ij,bjn->bin", H, curr)

        return out.reshape(B, self.d).view(orig_shape)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x = x.reshape(-1, self.d)
        B = x.shape[0]

        if self.K == 1:
            out = self.axis_pow2.inverse_transform(x)
            return out.view(orig_shape)

        curr = x.view(B, self.K, self.n2)

        Ht = self.hadK.t().to(device=curr.device, dtype=curr.dtype)
        curr = torch.einsum("ij,bjn->bin", Ht, curr) * self.invK.to(device=curr.device, dtype=curr.dtype)

        curr = curr / self.scaleK.to(device=curr.device, dtype=curr.dtype)

        flat = curr.reshape(B * self.K, self.n2)
        flat = self.axis_pow2.inverse_transform(flat)
        curr = flat.view(B, self.K, self.n2)

        out = curr.reshape(B, self.d)
        return out.view(orig_shape)

    def get_storage_bits(self) -> int:
        if hasattr(self.axis_pow2, "get_storage_bits"):
            return int(self.axis_pow2.get_storage_bits())
        return 0
    
    @torch.no_grad()
    def quantize_theta_8bit(self, *args, **kwargs):
        if hasattr(self.axis_pow2, "quantize_theta_8bit"):
            return self.axis_pow2.quantize_theta_8bit(*args, **kwargs)

def _is_factor(a: int, b: int) -> bool:
    return (a % b) == 0

def _factor_schedule_greedy(n: int, base_b: int = 8, max_b: int = 8) -> List[int]:
    """
    Produce a stage schedule [b1, b2, ...] such that prod(bi) == n and each bi <= max_b.
    Preference: divide by base_b repeatedly, then divide by remaining small factors.
    """
    if n <= 1:
        return [1]
    rem = int(n)
    out: List[int] = []

    while rem % base_b == 0 and rem > 1:
        out.append(int(base_b))
        rem //= int(base_b)

    for f in range(min(max_b, rem), 1, -1):
        while rem % f == 0 and rem > 1:
            out.append(int(f))
            rem //= int(f)

    if rem != 1:
        out.append(int(rem))

    prod = 1
    for b in out:
        prod *= b
    if prod != n:
        raise RuntimeError(f"Internal schedule error: prod(schedule)={prod} != n={n}. schedule={out}")
    return out

def _make_sylvester_hadamard(b: int, device, dtype) -> torch.Tensor:
    """
    Normalized Sylvester Hadamard H_b / sqrt(b), b must be power-of-two.
    """
    if not ((b & (b - 1) == 0) and b > 0):
        raise ValueError(f"Hadamard Sylvester needs pow2, got b={b}")

    H = torch.tensor([[1.0]], device=device, dtype=dtype)
    steps = int(math.log2(b))
    for _ in range(steps):
        top = torch.cat([H, H], dim=1)
        bot = torch.cat([H, -H], dim=1)
        H = torch.cat([top, bot], dim=0)
    return H / math.sqrt(b)

def _make_fixed_orthogonal(b: int, device, dtype, kind: str = "had_or_qr", seed: int = 0) -> torch.Tensor:
    """
    Return a fixed orthogonal matrix M_b.

    kind:
      - "had_or_qr": pow2 -> Hadamard; else -> deterministic QR random orthogonal
      - "had_or_ident": pow2 -> Hadamard; else -> Identity
      - "qr": always QR orthogonal
      - "identity": identity
    """
    kind = str(kind)
    if kind == "identity":
        return torch.eye(b, device=device, dtype=dtype)

    if kind in ("had_or_qr", "had_or_ident"):
        if (b & (b - 1) == 0) and b > 0:
            return _make_sylvester_hadamard(b, device=device, dtype=dtype)
        if kind == "had_or_ident":
            return torch.eye(b, device=device, dtype=dtype)

    # QR random orthogonal (deterministic by seed+b)
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) + 10007 * int(b))
    A = torch.randn(b, b, generator=g, device="cpu", dtype=torch.float32)
    Q, R = torch.linalg.qr(A)
    d = torch.sign(torch.diag(R))
    d[d == 0] = 1.0
    Q = Q * d.unsqueeze(0)
    return Q.to(device=device, dtype=dtype)

def _cayley_batch_from_skew(skew: torch.Tensor) -> torch.Tensor:
    """
    Cayley transform on a batch of skew-symmetric matrices:
        Q = (I + S)^{-1} (I - S)
    skew: [B, b, b]
    returns Q: [B, b, b]
    """
    skew = skew.to(torch.float32)
    b = skew.shape[-1]
    I = torch.eye(b, device=skew.device, dtype=torch.float32).unsqueeze(0).expand(skew.shape[0], b, b)
    return torch.linalg.solve(I + skew, I - skew)

def _skew_from_raw(theta_raw: torch.Tensor, clip: Optional[float]) -> torch.Tensor:
    """
    Make skew-symmetric from raw parameter tensor [B,b,b].
    Optionally clip with tanh for stability.
    """
    x = theta_raw
    if clip is not None and clip > 0:
        x = torch.tanh(x) * float(clip)
    skew = 0.5 * (x - x.transpose(-1, -2))
    return skew

class FactorizedOrthogonalRotation(nn.Module):
    """
    Orthogonal rotation R in O(n) applied on the right (row-vector convention):
        x [..., n] -> x @ R

    It is a product of stages. Each stage mixes a radix b_t along a stride tensorization.

    Each stage kernel per block:
        K = Cayley(skew(theta)) @ M_b
    where M_b is a fixed orthogonal mixer.
    """
    def __init__(
        self,
        n: int,
        base_b: int = 8,
        max_b: int = 8,
        stage_bs: Optional[List[int]] = None,
        n_passes: int = 1,
        ordering_mode: str = "stride",
        fixed_mixer: str = "had_or_qr",
        fixed_seed: int = 0,
        theta_init_scale: float = 0.0,
        theta_clip: Optional[float] = 0.05,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        use_givens_b2: bool = False,
    ):
        super().__init__()
        self.n = int(n)
        self.base_b = int(base_b)
        self.max_b = int(max_b)
        self.n_passes = int(max(1, n_passes))
        self.ordering_mode = str(ordering_mode)
        self.fixed_mixer = str(fixed_mixer)
        self.fixed_seed = int(fixed_seed)
        self.theta_init_scale = float(theta_init_scale)
        self.theta_clip = float(theta_clip) if theta_clip is not None else None
        self.use_givens_b2 = bool(use_givens_b2)

        def _v2(x: int) -> int:
            c = 0
            while x > 0 and (x & 1) == 0:
                c += 1
                x >>= 1
            return c

        if stage_bs is None:
            stage_bs = _factor_schedule_greedy(self.n, base_b=self.base_b, max_b=self.max_b)
        else:
            prod = 1
            for b in stage_bs:
                b = int(b)
                if b <= 1:
                    raise ValueError(f"Invalid stage size {b}. Must be >=2.")
                prod *= b
            if prod != self.n:
                raise ValueError(f"prod(stage_bs) must equal n. Got prod={prod}, n={self.n}. stage_bs={stage_bs}")

        self.stage_bs: List[int] = [int(b) for b in stage_bs]
        self.m: int = len(self.stage_bs)

        self.stage_is_givens: List[bool] = [
            (self.use_givens_b2 and (b_t == 2)) for b_t in self.stage_bs
        ]
        
        for t, b_t in enumerate(self.stage_bs):
            M = _make_fixed_orthogonal(
                b_t, device=(device or "cpu"), dtype=dtype,
                kind=self.fixed_mixer, seed=self.fixed_seed
            )
            self.register_buffer(f"M_fixed_{t}", M, persistent=False)
            idx = torch.triu_indices(b_t, b_t, offset=1)  # [2, tcount]
            self.register_buffer(f"triu_idx_{t}", idx, persistent=False)

        thetas: List[nn.ParameterList] = []
        for p in range(self.n_passes):
            plist: List[nn.Parameter] = []
            for t, b_t in enumerate(self.stage_bs):
                D_t = self.n // b_t
                if self.stage_is_givens[t]:
                    param = torch.zeros(D_t, device=device, dtype=dtype)
                    if self.theta_init_scale > 0:
                        param = param.normal_(mean=0.0, std=self.theta_init_scale)
                    plist.append(nn.Parameter(param))
                else:
                    param = torch.zeros(D_t, b_t, b_t, device=device, dtype=dtype)
                    if self.theta_init_scale > 0:
                        param = param.normal_(mean=0.0, std=self.theta_init_scale)
                    plist.append(nn.Parameter(param))
            thetas.append(nn.ParameterList(plist))
        self.theta_raw = nn.ModuleList(thetas)

        self.register_buffer("perm_seeds", torch.empty(0, dtype=torch.int64), persistent=False)
        self._cached_device = None
        self._cached_perm_fwd = None
        self._cached_perm_bwd = None

        self._blocks_cache: Dict[Tuple[torch.device, torch.dtype, int, int], torch.Tensor] = {}
        self.register_buffer("theta_is_quantized", torch.tensor(0, dtype=torch.uint8), persistent=True)

        for p in range(self.n_passes):
            for t, b_t in enumerate(self.stage_bs):
                D_t = self.n // b_t
                T = 1 if self.stage_is_givens[t] else (b_t * (b_t - 1) // 2)

                qname = self._qname(p, t)
                sname = self._sname(p, t)

                if not hasattr(self, qname):
                    self.register_buffer(
                        qname,
                        torch.zeros(D_t, T, dtype=torch.int8, device=(device or "cpu")),
                        persistent=True,
                    )
                if not hasattr(self, sname):
                    self.register_buffer(
                        sname,
                        torch.ones(D_t, dtype=torch.float16, device=(device or "cpu")),
                        persistent=True,
                    )


    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._blocks_cache.clear()
        return self

    @torch.no_grad()
    def _invert_perm(self, p: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(p)
        inv[p] = torch.arange(p.numel(), device=p.device, dtype=p.dtype)
        return inv

    def _qname(self, p: int, t: int) -> str:
        return f"theta_q_p{p}_t{t}"

    def _sname(self, p: int, t: int) -> str:
        return f"theta_s_p{p}_t{t}"

    def is_theta_quantized(self) -> bool:
        return bool(self.theta_is_quantized.item())
    
    def _effective_angle(self, theta: torch.Tensor) -> torch.Tensor:
        if self.theta_clip is not None and self.theta_clip > 0:
            return torch.tanh(theta) * float(self.theta_clip)
        return theta

    @torch.no_grad()
    def quantize_theta_8bit(self, scale_dtype: torch.dtype = torch.float16):
        """
        Replace theta_raw parameters with packed int8 + per-block scales.
        """
        if self.is_theta_quantized():
            return

        self._blocks_cache.clear()

        for p in range(self.n_passes):
            for t, b_t in enumerate(self.stage_bs):
                if self.stage_is_givens[t]:
                    theta = self.theta_raw[p][t].detach().to(dtype=torch.float32)
                    theta_eff = self._effective_angle(theta)
                    packed = theta_eff.view(theta_eff.shape[0], 1)  # [D,1]
                else:
                    theta = self.theta_raw[p][t].detach().to(dtype=torch.float32)
                    skew = _skew_from_raw(theta, self.theta_clip)
                    idx = getattr(self, f"triu_idx_{t}").to(device=skew.device)
                    iu, ju = idx[0], idx[1]
                    packed = skew[:, iu, ju]  # [D,T]

                max_abs = packed.abs().amax(dim=1)  # [D]
                scale = torch.where(max_abs > 0, max_abs / 127.0, torch.ones_like(max_abs))
                q = torch.clamp(torch.round(packed / scale.unsqueeze(1)), -127, 127).to(torch.int8)

                qname = self._qname(p, t)
                sname = self._sname(p, t)

                qb = getattr(self, qname)
                sb = getattr(self, sname)

                if qb.shape != q.shape:
                    raise RuntimeError(f"{qname} shape mismatch: buf={tuple(qb.shape)} vs new={tuple(q.shape)}")
                if sb.shape != scale.shape:
                    raise RuntimeError(f"{sname} shape mismatch: buf={tuple(sb.shape)} vs new={tuple(scale.shape)}")

                qb.copy_(q.to(device=qb.device))
                sb.copy_(scale.to(device=sb.device, dtype=sb.dtype))

        self.theta_is_quantized.fill_(1)

        dev0 = self.theta_raw[0][0].device
        for p in range(self.n_passes):
            for t in range(self.m):
                self.theta_raw[p][t] = nn.Parameter(torch.empty(0, device=dev0), requires_grad=False)

    def _get_givens_angles(self, p: int, t: int, device: torch.device) -> torch.Tensor:
        """
        Returns angles theta [D] (for b=2 givens stages).
        """
        quant = self.is_theta_quantized()

        if not quant:
            theta = self.theta_raw[p][t].to(device=device, dtype=torch.float32)  # [D]
            return self._effective_angle(theta)

        q = getattr(self, self._qname(p, t)).to(device=device)          # [D,1] int8 (or [D])
        s = getattr(self, self._sname(p, t)).to(device=device).float()  # [D]
        if q.dim() == 2:
            qv = q.float().squeeze(1)
        else:
            qv = q.float()
        return qv * s  # [D]

    def _get_skew(self, p: int, t: int, device: torch.device) -> torch.Tensor:
        """
        Returns skew matrices [D,b,b].
        """
        b_t = self.stage_bs[t]

        quant = self.is_theta_quantized()

        if not quant:
            theta = self.theta_raw[p][t].to(device=device, dtype=torch.float32)
            return _skew_from_raw(theta, self.theta_clip)

        q = getattr(self, self._qname(p, t)).to(device=device)
        s = getattr(self, self._sname(p, t)).to(device=device).float()
        vals = q.float() * s.unsqueeze(1)

        D = vals.shape[0]
        skew = torch.zeros(D, b_t, b_t, device=device, dtype=torch.float32)
        idx = getattr(self, f"triu_idx_{t}").to(device=device)
        iu, ju = idx[0], idx[1]

        skew[:, iu, ju] = vals
        skew[:, ju, iu] = -vals
        return skew

    def _compute_blocks(self, p: int, t: int, device: torch.device, out_dtype: torch.dtype) -> torch.Tensor:
        """
        blocks: [D_t, b_t, b_t] in out_dtype
        """
        b_t = self.stage_bs[t]

        if self.stage_is_givens[t]:
            # b_t == 2
            theta = self._get_givens_angles(p, t, device=device)
            c = torch.cos(theta)
            s = torch.sin(theta)
            Q = torch.stack(
                [
                    torch.stack([c, -s], dim=-1),
                    torch.stack([s,  c], dim=-1),
                ],
                dim=-2,
            )
        else:
            skew = self._get_skew(p, t, device=device)
            Q = _cayley_batch_from_skew(skew)

        M = getattr(self, f"M_fixed_{t}").to(device=device, dtype=torch.float32)
        blocks = torch.matmul(Q, M)
        return blocks.to(dtype=out_dtype)

    def _get_blocks(self, p: int, t: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not self.training:
            key = (device, dtype, p, t)
            if key not in self._blocks_cache:
                self._blocks_cache[key] = self._compute_blocks(p, t, device, dtype)
            return self._blocks_cache[key]
        return self._compute_blocks(p, t, device, dtype)
    
    def _apply_stride_pass(self, x2d: torch.Tensor, p: int, inverse: bool) -> torch.Tensor:
        """
        Apply one pass in stride mode. x2d is [B,n].
        """
        B, n = x2d.shape
        assert n == self.n
        curr = x2d

        if not inverse:
            step = 1
            for t, b_t in enumerate(self.stage_bs):
                assert self.n % (b_t * step) == 0, "Invalid stride schedule"
                blocks = self.n // (b_t * step)

                curr = curr.view(B, blocks, b_t, step).transpose(2, 3).contiguous()

                K = self._get_blocks(p, t, device=curr.device, dtype=curr.dtype)  # [D_t,b,b]
                K = K.view(blocks, step, b_t, b_t) # [blocks, step, b, b]

                curr = torch.einsum("bcsi,csij->bcsj", curr, K.transpose(-1, -2))
                curr = curr.transpose(2, 3).contiguous().view(B, self.n)

                step *= b_t
        else:
            steps: List[int] = []
            step = 1
            for b_t in self.stage_bs:
                steps.append(step)
                step *= b_t

            for t in reversed(range(self.m)):
                b_t = self.stage_bs[t]
                step = steps[t]
                blocks = self.n // (b_t * step)

                curr = curr.view(B, blocks, b_t, step).transpose(2, 3).contiguous()
                K = self._get_blocks(p, t, device=curr.device, dtype=curr.dtype).view(blocks, step, b_t, b_t)
                curr = torch.einsum("bcsi,csij->bcsj", curr, K)
                curr = curr.transpose(2, 3).contiguous().view(B, self.n)

        return curr

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply x @ R.
        """
        orig_shape = x.shape
        x2d = x.reshape(-1, self.n)
        curr = x2d

        for p in range(self.n_passes):
            if self.ordering_mode == "stride":
                curr = self._apply_stride_pass(curr, p=p, inverse=False)
            else:
                raise ValueError(f"Unknown ordering_mode={self.ordering_mode}")

        return curr.view(orig_shape)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply x @ R^T.
        """
        orig_shape = x.shape
        x2d = x.reshape(-1, self.n)
        curr = x2d

        for p in reversed(range(self.n_passes)):
            if self.ordering_mode == "stride":
                curr = self._apply_stride_pass(curr, p=p, inverse=True)
            else:
                raise ValueError(f"Unknown ordering_mode={self.ordering_mode}")

        return curr.view(orig_shape)

    def get_storage_bits(self) -> int:
        bits = 0
        n = self.n

        if self.is_theta_quantized():
            for p in range(self.n_passes):
                for t, b_t in enumerate(self.stage_bs):
                    q = getattr(self, self._qname(p, t))
                    s = getattr(self, self._sname(p, t))
                    bits += int(q.numel()) * 8
                    bits += int(s.numel()) * (16 if s.dtype == torch.float16 else 32)
        else:
            for _p in range(self.n_passes):
                for b_t in self.stage_bs:
                    params = n * (b_t - 1) // 2
                    bits += int(params) * 16

        return bits
    
    def state_dict(self, destination=None, prefix: str = "", keep_vars: bool = False):
        sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        if self.is_theta_quantized():
            drop_pref = prefix + "theta_raw."
            for k in [k for k in list(sd.keys()) if k.startswith(drop_pref)]:
                sd.pop(k, None)

        return sd

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        flag_key = prefix + "theta_is_quantized"

        quant_in_sd = False
        if flag_key in state_dict:
            quant_in_sd = bool(state_dict[flag_key].item())
        else:
            for p in range(self.n_passes):
                for t in range(self.m):
                    if (prefix + self._qname(p, t)) in state_dict:
                        quant_in_sd = True
                        break
                if quant_in_sd:
                    break

        self.theta_is_quantized.fill_(1 if quant_in_sd else 0)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

        self.theta_is_quantized.fill_(1 if quant_in_sd else 0)

        if quant_in_sd:
            for p in range(self.n_passes):
                for t in range(self.m):
                    k = prefix + f"theta_raw.{p}.{t}"
                    if k in missing_keys:
                        missing_keys.remove(k)
        else:
            for p in range(self.n_passes):
                for t in range(self.m):
                    qk = prefix + self._qname(p, t)
                    sk = prefix + self._sname(p, t)
                    if qk in missing_keys:
                        missing_keys.remove(qk)
                    if sk in missing_keys:
                        missing_keys.remove(sk)

        if flag_key in missing_keys:
            missing_keys.remove(flag_key)


class HARPRotationPreconditioned(nn.Module):
    def __init__(
        self,
        d: int,
        base_b: int = 8,
        max_b: int = 8,
        stage_bs: Optional[List[int]] = None,
        n_passes: int = 1,
        ordering_mode: str = "stride",
        fixed_mixer: str = "had_or_qr",
        fixed_seed: int = 0,
        theta_init_scale: float = 0.0,
        theta_clip: Optional[float] = 0.0,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        kron_fallback: bool = False,
        use_givens_b2: bool = False,
    ):
        super().__init__()
        self.d = int(d)
        self.rot = FactorizedOrthogonalRotation(
            n=self.d,
            base_b=base_b,
            max_b=max_b,
            stage_bs=stage_bs,
            n_passes=n_passes,
            ordering_mode=ordering_mode,
            fixed_mixer=fixed_mixer,
            fixed_seed=fixed_seed,
            theta_init_scale=theta_init_scale,
            theta_clip=theta_clip,
            device=device,
            dtype=dtype,
            use_givens_b2=use_givens_b2,
        )

        if bool(kron_fallback) and (not _is_pow2(self.d)):
            from lib.utils.matmul_had import get_hadK  # type: ignore
            _, K = get_hadK(self.d, transpose=False)
            K = int(K)
            if K > 1 and (self.d % K == 0):
                n2 = self.d // K
                if _is_pow2(int(n2)):
                    axis_pow2 = FactorizedOrthogonalRotation(
                        n=int(n2),
                        base_b=base_b,
                        max_b=max_b,
                        stage_bs=None,
                        n_passes=n_passes,
                        ordering_mode=ordering_mode,
                        fixed_mixer=fixed_mixer,
                        fixed_seed=fixed_seed,
                        theta_init_scale=theta_init_scale,
                        theta_clip=theta_clip,
                        device=device,
                        dtype=dtype,
                        use_givens_b2=use_givens_b2,
                    )
                    self.rot = KronHadKFallbackRotation(
                        d=self.d,
                        axis_pow2=axis_pow2,
                        device=device,
                        dtype=dtype,
                    )

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.rot.transform(x)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.rot.inverse_transform(x)

    def get_storage_bits(self) -> int:
        return self.rot.get_storage_bits()

    @torch.no_grad()
    def quantize_theta_8bit(self, scale_dtype: torch.dtype = torch.float16):
        if hasattr(self.rot, "quantize_theta_8bit"):
            self.rot.quantize_theta_8bit(scale_dtype=scale_dtype)
