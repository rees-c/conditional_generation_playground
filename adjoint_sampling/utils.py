import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from aliases import *


class FourierTimeEmbeddings(nn.Module):
    """https://github.com/jiaor17/DiffCSP-PP/blob/55099b51fe8ebb6695faa141ada5d43d5b83fe39/diffcsp/pl_modules/diffusion.py#L51C1-L64C26"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time):
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


class GeometricNoiseScheduler(nn.Module):
    def __init__(self, sigma_min=0.1, sigma_max=1.0):
        """
        t=1 is a clean sample. t=0 is a noisy sample.

        sigma(t) =
            sigma_min *
            (sigma_max / sigma_min).pow(1-t) *
            sqrt(2 * log[sigma_max / sigma_min])
        """
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # Eqn 50 in Adjoint Sampling paper (appendix B)
        self.base_dist_std: float = (
            sigma_max * math.sqrt(1. - (sigma_min / sigma_max) ** 2)
        )

    @torch.no_grad()
    def get_sigma(self, t: Tensor) -> Tensor:
        assert torch.all((t >= 0) & (t <= 1))
        max_to_min = torch.tensor(self.sigma_max / self.sigma_min, device=t.device)
        sigmas = self.sigma_min * (max_to_min).pow(1-t) * torch.sqrt(2 * max_to_min.log())
        return sigmas

    @torch.no_grad()
    def get_nu_t_given_1(self, t: Tensor) -> Tensor:
        assert torch.all((t >= 0) & (t <= 1))
        min_to_max = torch.tensor(self.sigma_min / self.sigma_max, device=t.device)
        return (self.sigma_max ** 2) * (min_to_max.pow(2) - min_to_max.pow(2 * t))

    @torch.no_grad()
    def get_alpha_t_given_1(self, t: Tensor) -> Tensor:
        assert torch.all((t >= 0) & (t <= 1))
        min_to_max = torch.tensor(self.sigma_min / self.sigma_max, device=t.device)
        return (min_to_max.pow(2 * t) - 1) / (min_to_max.pow(2) - 1)

    @staticmethod
    def uniform_sample_timestep(batch_size: int, device: torch.device):
        return torch.rand(batch_size, device=device)


class PreBatchedDataset(Dataset):
    def __init__(self, batched_data):
        """
        Args:
          batched_data (list of tuples): Each tuple is a batch, typically (inputs, targets)
        """
        self.batched_data = batched_data

    def __len__(self):
        return len(self.batched_data)

    def __getitem__(self, index):
        # Return the pre-batched data
        return self.batched_data[index]


class ReplayBuffer:
    """Store the most recent objects."""
    def __init__(self, max_size: int = 10_000):
        self.max_size = max_size
        self.buffer = []
        # self.pointer = 0  # points to the next index to insert at

    def add(self, states, energy_grads):
        """
        states: (batch_size, 2)
        energy_grads: (batch_size, 2)
        """
        # batch = [(state, grad) for state, grad in zip(states, energy_grads)]
        # self.buffer.extend(batch)
        self.buffer.append((states, energy_grads))
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]

    def get_data_loader(self):
        dataset = PreBatchedDataset(self.buffer)
        return DataLoader(dataset, batch_size=None, shuffle=True)

    def reset(self):
        self.buffer = []
        # self.pointer = 0

    def __len__(self):
        return len(self.buffer)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
