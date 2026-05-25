from typing import List

import torch


@torch.no_grad()
def quantize_blocks_hard_chunked(
    x: torch.Tensor,
    codebook,
    chunk: int = 4096,
) -> torch.Tensor:
    """
    Chunked hard quantization helper.

    Args:
        x: [N, block_dim]
        codebook: QTIP bitshift_codebook-like object with .quantize(...)
        chunk: number of blocks per chunk

    Returns:
        Quantized values with the same shape as x.
    """
    x = x.contiguous()
    outs: List[torch.Tensor] = []
    for i in range(0, x.shape[0], chunk):
        xb = x[i:i + chunk]
        qb, _ = codebook.quantize(xb)
        outs.append(qb.detach())
    return torch.cat(outs, dim=0)