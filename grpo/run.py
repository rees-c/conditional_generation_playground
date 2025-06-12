from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from aliases import *
from rewards import get_energy_function
from common_utils import AverageMeter
from grpo.model import DiscreteModel, DummyUniformModel
from grpo.utils import update_old_policy, compute_grpo_loss


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
    probs = policy((grid_x_indices.reshape(-1), grid_y_indices.reshape(-1)))
    probs = probs.prod(dim=-1)
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
    num_iterations = 3_001
    batch_size = 64
    hidden_dim = 64
    reward_n_bins_per_dim = 20
    use_sparse_reward = True
    reward_temperature = 1.0
    use_ema = True
    old_policy_update_freq = 1 if use_ema else 100
    epsilon = 0.1
    beta = 0.01
    n_log_prints = 10
    log_freq = int(num_iterations / n_log_prints)

    # -- Reward
    energy_fn: nn.Module = get_energy_function(
        device,
        sparse=use_sparse_reward,
        discrete_energy=True,
        n_bins_per_dim=reward_n_bins_per_dim,
    )
    reward_fn: Callable = (
        lambda samples: torch.exp(-energy_fn(samples[0], samples[1]) / reward_temperature)
    )

    # -- Model
    reference_policy = DummyUniformModel(
        n_bins_per_dim=reward_n_bins_per_dim, n_dims=2, device=device
    )
    policy = DiscreteModel(reward_n_bins_per_dim, hidden_dim=hidden_dim)
    policy.to(device)

    old_policy = DiscreteModel(reward_n_bins_per_dim, hidden_dim=hidden_dim)
    old_policy.load_state_dict(policy.state_dict())
    old_policy.eval()
    for p in old_policy.parameters():
        p.requires_grad = False
    old_policy.to(device)

    # -- Optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # -- Visualization
    plot_current_policy(
        reward_n_bins_per_dim, energy_fn, device, policy,
        plot_name="discrete_grpo_iter0"
    )

    # -- Train
    n_tokens_per_sample = torch.tensor([2] * batch_size, dtype=torch.long, device=device)
    meters = {"loss": AverageMeter(), "reward": AverageMeter()}
    for iter in range(num_iterations):
        samples, probs = old_policy.sample_and_probs(batch_size)
        # (batch_size,), (batch_size,), (batch_size, n_tokens=2)
        rewards = reward_fn(samples)
        # (batch_size,)

        # outcome supervision (see 4.1.2)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

        grpo_loss = compute_grpo_loss(
            samples,
            n_tokens_per_sample,
            advantages,
            policy,
            old_policy,
            reference_policy,
            epsilon,
            beta,
        )
        grpo_loss.backward()
        torch.nn.utils.clip_grad_value_(policy.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        meters["loss"].update(grpo_loss.detach().cpu())
        meters["reward"].update(rewards.mean().detach().cpu())
        if iter % old_policy_update_freq == 0:
            update_old_policy(old_policy, policy, use_ema=use_ema)
        if iter % log_freq == 0:
            print(f"Iter [{iter}/{num_iterations}]: loss {meters['loss'].avg:0.4f}, reward {meters['reward'].avg:0.4f}")
            for meter in meters.values():
                meter.reset()

    plot_current_policy(
        reward_n_bins_per_dim, energy_fn, device, policy,
        plot_name=f"discrete_grpo_iter{iter}"
    )


if __name__ == "__main__":
    main()
