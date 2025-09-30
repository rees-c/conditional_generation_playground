import math

import torch
import torch.nn as nn

Tensor: type = torch.Tensor


class FourierTimeEmbeddings(nn.Module):
    """https://github.com/jiaor17/DiffCSP-PP/blob/55099b51fe8ebb6695faa141ada5d43d5b83fe39/diffcsp/pl_modules/diffusion.py#L51C1-L64C26"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: Tensor):
        """
        Args:
            time: torch.float
                shape (n,)

        Returns:
            (n, dim)
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings  # (n, dim)
