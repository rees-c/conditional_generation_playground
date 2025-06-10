from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from rewards import get_energy_function
from grpo.model import DiscreteModel, DummyUniformModel


@torch.no_grad()
def plot_current_policy(
    n_bins_per_dim: int,
    energy_fn: nn.Module,
    device: torch.device,
    policy: nn.Module,
    plot_name: str = None
):
    grid_x, grid_y = torch.meshgrid(
        torch.linspace(-0.5, 0.5, steps=n_bins_per_dim),
        torch.linspace(-0.5, 0.5, steps=n_bins_per_dim)
    )
    grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1).to(device)
    grid2 = torch.cartesian_prod(
        torch.linspace(-0.5, 0.5, steps=n_bins_per_dim),
        torch.linspace(-0.5, 0.5, steps=n_bins_per_dim)
    )  # (100, 2)
    assert torch.isclose(grid, grid2).all()
    target_dist = energy_fn.gmm.log_prob(grid).exp()
    # x_indices, y_indices, _ = policy.sample_and_probs(100)

    fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
    axs[0].pcolormesh(
        grid_x,
        grid_y,
        target_dist.view(n_bins_per_dim, n_bins_per_dim).cpu(),
        vmin=0.0, vmax=target_dist.max(),
    )
    axs[0].set_title("Target policy")

    grid_x_indices, grid_y_indices = torch.meshgrid(
        torch.arange(n_bins_per_dim), torch.arange(n_bins_per_dim)
    )
    probs = policy(grid_x_indices.reshape(-1), grid_y_indices.reshape(-1))
    axs[1].pcolormesh(
        grid_x, grid_y, probs.view(n_bins_per_dim, n_bins_per_dim).cpu(),
        vmin=0.0, vmax=target_dist.max(),
    )
    axs[1].set_title("Current policy")
    if plot_name is not None:
        plt.savefig(Path(plot_name).with_suffix(".png"))
    else:
        plt.show()
    plt.close()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # hyperparameters
    lr = 1e-3
    num_iterations = 10_000
    batch_size = 64
    hidden_dim = 64
    reward_n_bins_per_dim = 20
    use_sparse_reward = True

    energy_fn: nn.Module = get_energy_function(
        device,
        sparse=use_sparse_reward,
        discrete_energy=True,
        n_bins_per_dim=reward_n_bins_per_dim,
    )
    reference_policy = DummyUniformModel(
        n_bins_per_dim=reward_n_bins_per_dim, n_dims=2
    )
    policy = DiscreteModel(reward_n_bins_per_dim, hidden_dim=hidden_dim)
    policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    plot_current_policy(
        reward_n_bins_per_dim, energy_fn, device, policy,
        plot_name="discrete_grpo_epoch0"
    )

    for iter in range(num_iterations):
        loss = ...


if __name__ == "__main__":
    raise NotImplementedError
    main()
