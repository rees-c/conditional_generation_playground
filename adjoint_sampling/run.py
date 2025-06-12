import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from rewards import get_energy_function
from common_utils import AverageMeter
from adjoint_sampling.utils import ReplayBuffer, cycle
from adjoint_sampling.model import DiffusionModel


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Hyperparameters
    lr = 1e-3
    num_epochs = 301
    num_batches_per_epoch = 100
    batch_size = 96
    buffer_size = 5_000
    n_log_prints = 10
    log_freq = int(num_epochs / n_log_prints)
    energy_temperature = 0.07  #0.01
    max_target_score_norm = 20.0
    sparse_reward = True

    energy_fn: nn.Module = get_energy_function(device, sparse=sparse_reward)
    model = DiffusionModel(sigma_min=0.01, sigma_max=0.4)
    model.to(device)
    buffer = ReplayBuffer(int(buffer_size / batch_size))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    meter = AverageMeter()

    # Plot samples from initial model
    grid_x, grid_y = torch.meshgrid(torch.linspace(-0.5, 0.5, steps=25), torch.linspace(-0.5, 0.5, steps=25))
    grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1).to(device)

    target_dist = torch.exp(-energy_fn(grid)) #energy_fn.gmm.log_prob(grid).exp()
    plt.contourf(grid_x.cpu(), grid_y.cpu(), target_dist.view(25, 25).cpu())
    samples = model.sample(100).clamp(min=-0.5, max=0.5)
    samples = samples.cpu().numpy()
    plt.plot(samples[:, 0], samples[:, 1], "rx", alpha=0.5)
    plt.savefig(f"adjoint_sampling_{'sparse' if sparse_reward else ''}_epoch0.png")
    plt.close()

    for epoch in range(num_epochs):
        # Sample from model and store in buffer
        samples = model.sample(batch_size)
        energy_grads = energy_fn.grad(samples) / energy_temperature
        buffer.add(samples, energy_grads)

        # Train on some samples from the buffer
        loader = iter(cycle(buffer.get_data_loader()))
        for i in range(num_batches_per_epoch):
            batch_points, batch_energy_grads = next(loader)
            batch_energy_grads = batch_energy_grads
            loss = model.compute_loss(
                batch_points,
                batch_energy_grads,
                max_target_score_norm,
            )
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad()

            meter.update(loss.detach().cpu())

        if epoch % log_freq == 0:
            print(f"Epoch {epoch}: {meter.avg}")
        meter.reset()

    # Plot result
    grid_x, grid_y = torch.meshgrid(torch.linspace(-0.5, 0.5, steps=25), torch.linspace(-0.5, 0.5, steps=25))
    grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1).to(device)
    target_dist = torch.exp(-energy_fn(grid)) #energy_fn.gmm.log_prob(grid).exp()
    plt.contourf(grid_x.cpu(), grid_y.cpu(), target_dist.view(25, 25).cpu())

    samples = model.sample(100).clamp(min=-0.5, max=0.5)
    samples = samples.cpu().numpy()
    plt.plot(samples[:, 0], samples[:, 1], "rx", alpha=0.5)
    plt.savefig(f"adjoint_sampling_{'sparse' if sparse_reward else ''}_epoch{epoch}.png")
    plt.close()


if __name__ == "__main__":
    main()
