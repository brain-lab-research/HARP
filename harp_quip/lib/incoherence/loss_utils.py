from typing import Optional
import torch
import math

def quantize_block_hard(x: torch.Tensor, codebook, **quant_kwargs) -> torch.Tensor:
    vals, _ = codebook.quantize(x, return_idx=True, **quant_kwargs)
    return vals

def quantize_block_hard_ste_chunked(x: torch.Tensor, codebook, chunk: int = 131072, **quant_kwargs):
    """
    x: [N, codesz]
    Forward: exact codebook quantize
    """
    x_in = x
    x = x.contiguous()

    outs = []
    with torch.no_grad():
        for i in range(0, x.shape[0], chunk):
            xb = x[i:i+chunk]
            qb = codebook.quantize(xb, return_idx=False, **quant_kwargs)
            outs.append(qb)
        q = torch.cat(outs, dim=0)

    return x_in + (q - x_in).detach()
