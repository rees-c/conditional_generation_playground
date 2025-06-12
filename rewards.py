import torch
import torch.nn as nn
from torch.distributions import (
    MultivariateNormal, Normal, MixtureSameFamily, Independent, Categorical
)
import matplotlib.pyplot as plt

from aliases import *


def get_energy_function(
    device: torch.device,
    sparse: bool = False,
    discrete_energy: bool = False,
    n_bins_per_dim: int = 20,  # for discrete_energy == True
) -> nn.Module:
    if sparse:
        loc = torch.cartesian_prod(
            torch.tensor([-0.3, 0.0, 0.3], device=device),
            torch.tensor([-0.3, 0.0, 0.3], device=device)
        )
        scale = 0.07 * torch.rand_like(loc) + 0.05
        n_components = scale.shape[0]
    else:
        n_components = 3
        loc = torch.stack(
          [
              torch.linspace(-0.21, 0.2, steps=n_components, device=device),
              torch.linspace(-0.21, 0.2, steps=n_components, device=device),
          ], dim=-1
        )
        scale = 0.3 * torch.tensor(
            [[0.5028, 0.4018],
              [0.2751, 0.6016],
              [0.3351, 0.4889]]
        )
    components = Independent(Normal(loc, scale), 1)  # [batch_shape: (5, 2), event_shape: (,)] -> [batch_shape: (5,), event_shape: (2,)]
    mixture_probs = Categorical(torch.ones(n_components, device=device))
    gmm = MixtureSameFamily(mixture_probs, components)
    if discrete_energy:
        return DiscreteEnergyModel(gmm, n_bins_per_dim)
    else:
        return EnergyModel(gmm)


class EnergyModel(nn.Module):
    """GMM differentiable reward"""
    def __init__(self, gmm: MixtureSameFamily):
        super().__init__()
        self.gmm = gmm

    def forward(self, x: Tensor):
        """
        x: shape (n, 2)
        """
        return -self.gmm.log_prob(x)

    def grad(self, x):
        with torch.enable_grad():
            x.requires_grad_(True)
            grad = torch.autograd.grad(self(x).sum(), x)[0]
            x.requires_grad_(False)
        return grad


class DiscreteEnergyModel(nn.Module):
    """Discretize a gmm"""
    @torch.no_grad()
    def __init__(self, gmm: MixtureSameFamily, n_bins_per_dim: int):
        super().__init__()
        self.gmm = gmm
        self.n_bins_per_dim = n_bins_per_dim

        grid = torch.cartesian_prod(
            torch.linspace(-0.5, 0.5, steps=self.n_bins_per_dim),
            torch.linspace(-0.5, 0.5, steps=self.n_bins_per_dim)
        )  # (100, 2)
        energies = -gmm.log_prob(grid).reshape(self.n_bins_per_dim, self.n_bins_per_dim)
        self.register_buffer("energies", energies)
        # (self.n_bins_per_dim, self.n_bins_per_dim)
        self.register_buffer("normalizing_constant", torch.exp(-self.energies).sum())

    def forward(self, x_indices: Tensor, y_indices: Tensor) -> Tensor:
        """
        Args:
            x_indices: torch.long
                shape (batch_size,)

        Returns:
            shape (batch_size,)
        """
        return self.energies[x_indices, y_indices]


@torch.no_grad()
def plot_continuous_reward(device):
    n_components = 3
    loc = torch.stack(
      [
          torch.linspace(-0.21, 0.2, steps=n_components, device=device),
          torch.linspace(-0.21, 0.2, steps=n_components, device=device),
      ], dim=-1
    )
    # scale = 0.5 * torch.rand(n_components, 2) + 0.2
    scale = 0.3 * torch.tensor(
        [[0.5028, 0.4018],
          [0.2751, 0.6016],
          [0.3351, 0.4889]]
    )
    components = Independent(Normal(loc, scale), 1)  # [batch_shape: (5, 2), event_shape: (,)] -> [batch_shape: (5,), event_shape: (2,)]
    mixture_probs = Categorical(torch.ones(n_components, device=device))
    gmm = MixtureSameFamily(mixture_probs, components)

    print(loc)
    print(scale)

    grid_x, grid_y = torch.meshgrid(torch.linspace(-0.5, 0.5, steps=25), torch.linspace(-0.5, 0.5, steps=25))
    grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1).to(device)
    energies = gmm.log_prob(grid).exp()

    plt.contourf(grid_x.cpu(), grid_y.cpu(), energies.view(25, 25).cpu())
    plt.show()


if __name__ == "__main__":
    plot_continuous_reward("cpu")
