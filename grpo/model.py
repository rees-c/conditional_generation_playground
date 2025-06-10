import torch
import torch.nn as nn
import torch.nn.functional as F

from aliases import *


class DiscreteModel(nn.Module):
    """Autoregressive model for rewards.DiscreteEnergyModel"""
    def __init__(self, n_bins_per_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.universe_size_per_dim = n_bins_per_dim
        self.x_emb = nn.Parameter(
            torch.randn(self.universe_size_per_dim, hidden_dim), requires_grad=True
        )
        self.y_emb = nn.Parameter(
            torch.randn(self.universe_size_per_dim, hidden_dim), requires_grad=True
        )
        self.x_logit_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.y_logit_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_indices: Tensor, y_indices: Tensor) -> Tensor:
        """
        Args:
            x_indices: torch.long
                shape (batch_size,)
            y_indices: torch.long
                shape (batch_size,)

        Returns:
            shape (batch_size,)
        """
        batch_size = x_indices.shape[0]
        device = x_indices.device

        x_probs = F.softmax(self.x_logit_head(self.x_emb), dim=0).squeeze(dim=-1)
        # (universe_size_per_dim,)
        emb = torch.cat(
            [
                self.x_emb[x_indices][:, None, :].expand(
                    batch_size, self.universe_size_per_dim, -1
                ),
                self.y_emb[None, :, :].expand(
                    batch_size, self.universe_size_per_dim, -1
                ),
            ], dim=-1
        )
        # (batch_size, universe_size_per_dim, 2 * hidden_dim)
        y_probs = F.softmax(self.y_logit_head(emb).squeeze(dim=-1), dim=1)
        # (batch_size, universe_size_per_dim)

        x_probs = x_probs[x_indices]
        y_probs = y_probs[torch.arange(batch_size, device=device), y_indices]
        probs = x_probs * y_probs
        return probs  # (batch_size,)

    def sample_and_probs(self, batch_size: int = 1) -> Tuple[Tensor, Tensor, Tensor]:
        device = self.x_emb.data.device

        x_probs = F.softmax(self.x_logit_head(self.x_emb), dim=0).squeeze(dim=-1)
        # (universe_size_per_dim,)
        x_indices = torch.multinomial(x_probs, batch_size, replacement=True)
        # (batch_size,)

        emb = torch.cat(
            [
                self.x_emb[x_indices][:, None, :].expand(
                    batch_size, self.universe_size_per_dim, -1
                ),
                self.y_emb[None, :, :].expand(
                    batch_size, self.universe_size_per_dim, -1
                ),
            ], dim=-1
        )
        # (batch_size, universe_size_per_dim, 2 * hidden_dim)
        y_probs = F.softmax(self.y_logit_head(emb).squeeze(dim=-1), dim=1)
        # (batch_size, universe_size_per_dim)
        y_indices = torch.multinomial(y_probs, 1, replacement=True)
        # (batch_size,)

        x_probs = x_probs[x_indices]
        y_probs = y_probs[torch.arange(batch_size, device=device), y_indices]
        probs = x_probs * y_probs
        return x_indices, y_indices, probs


class DummyUniformModel(nn.Module):
    def __init__(self, n_bins_per_dim: int, n_dims: int = 2):
        super().__init__()
        self.n_bins_per_dim = n_bins_per_dim
        self.n_dims = n_dims
        self.n_bins = n_bins_per_dim * n_dims

    def forward(self, batch_size: int, device: torch.device) -> Tensor:
        return (1.0 / self.n_bins) * torch.ones(batch_size, device=device)

    def sample_and_probs(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor]:
        x_indices = torch.randint(low=0, high=self.n_bins, size=(batch_size,))
        y_indices = torch.randint(low=0, high=self.n_bins, size=(batch_size,))
        probs = self(batch_size)
        return x_indices, y_indices, probs
