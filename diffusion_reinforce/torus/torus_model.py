seed = 42
import torch
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)
import numpy as np
np.random.seed(seed)
import random
random.seed(seed)

import json
import math
from typing import *
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import os
import functools
from queue import Empty

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt

from common_utils import AverageMeter
from rewards import get_energy_function
from grpo.utils import update_old_policy
from diffusion_reinforce.model_utils import FourierTimeEmbeddings

Tensor: type = torch.Tensor

parent_dir = Path("torus")
output_dir = parent_dir / Path(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
pretrained_model_path = parent_dir / "ckpt.pt"

# Model hyperparams
num_timesteps = 100
hidden_dim = 64
spatial_dim = 1
model_kwargs = {
    "num_timesteps": num_timesteps, "hidden_dim": hidden_dim,
    "spatial_dim": spatial_dim
}
assert spatial_dim in [1, 2]


m = 3
translations_1d = torch.arange(-m, m+1, device=device)
translations = torch.cartesian_prod(*([translations_1d] * spatial_dim))
@torch.no_grad()
def wrapped_reward_fn(
    x: Tensor, energy_fn: Callable, reward_temperature: float = 1.0
) -> Tensor:
    """x: shape (B, d)"""
    spatial_dim = x.shape[-1]
    device = x.device

    # (n_translations, d)
    xs = (x % 1.0)[:, None, :] + translations[None, ...].to(x.dtype)
    # (B, n_translations, d)
    reward = 10.0 * torch.exp(
        -energy_fn(xs.view(-1, spatial_dim) - 0.5)
        / reward_temperature
    ).view(x.shape[0], translations.shape[0], spatial_dim)
    reward = reward.mean(dim=1)
    return reward


def wrapped_gaussian_log_pdf(x: Tensor, mean: Tensor, std: Tensor, m: int = 5) -> Tensor:
    """
    Get log probability + log (unknown) normalizing constant

    Args:
        x: (batch_size, 2)
        mean: (batch_size, 2)
        std: (,)

    Returns:
        (batch_size,)
    """
    ndim = x.shape[-1]
    device = x.device

    translations_1d = torch.arange(-m, m+1, device=device)
    translations = torch.cartesian_prod(translations_1d, translations_1d)
    # (n_translations, 2)
    mean = mean[:, None, :] % 1.0 + translations[None, ...]
    # (batch_size, n_translations, 2)

    std = std.clamp(min=1e-6)
    x = x % 1.0

    log_pdf = -0.5 * (
        ndim * math.log(2. * math.pi)
        + 2 * ndim * torch.log(std)
        + (x[:, None, :] - mean).pow(2).sum(dim=-1) / std.pow(2)
    )
    log_pdf = torch.logsumexp(log_pdf, dim=1)
    return log_pdf


def p_wrapped_multivariate_normal(x, sigma, N=10, T=1.0):
    # x: [..., d]
    sigma = sigma.clamp(min=1e-6)
    dims = x.shape[-1]
    coordinates = [torch.arange(-N, N+1, device=x.device)] * dims
    meshgrid = torch.meshgrid(coordinates)
    meshpoints = torch.cat([_.reshape(-1,1) for _ in meshgrid],dim=-1)
    # (n_translations, d)
    p_ = 0.
    for point in meshpoints:
        p_ += torch.exp(-((x + T * point) ** 2).sum(dim=-1) / 2 / sigma[...,0] ** 2)
    return p_


@torch.no_grad()
def d_log_p_wrapped_multivariate_normal(x, sigma, N=5, T=1.0):
    # x: [..., d]
    dims = x.shape[-1]
    coordinates = [torch.arange(-N, N+1, device=x.device, dtype=torch.float)] * dims
    meshgrid = torch.meshgrid(coordinates)
    meshpoints = torch.cat([_.reshape(-1,1) for _ in meshgrid],dim=-1) # [(2N + 1) ^ d, d]
    p_ = 0.
    for point in meshpoints:
        p_ += (x + T * point) / sigma ** 2 * torch.exp(-((x + T * point) ** 2).sum(dim=-1, keepdim=True) / 2 / sigma ** 2)

    return -p_ / p_wrapped_multivariate_normal(x, sigma, N, T).unsqueeze(-1)


@torch.no_grad()
def sigma_norm(sigma, T=1.0, sn=5000, dims=1):
    sigmas = sigma[None, :, None].repeat(sn, 1, dims)
    x_sample = sigmas * torch.randn_like(sigmas)
    x_sample = x_sample % T
    normal_ = d_log_p_wrapped_multivariate_normal(x_sample, sigmas, T=T)
    return (normal_ ** 2).sum(dim=-1).mean(dim=0)


def kl_divergence(log_p: Tensor, log_q: Tensor) -> Tensor:
    # todo: use the lower-variance Rao-Blackwellized estimator instead
    #  (https://arxiv.org/pdf/2504.10637)
    #  KL divergence between Gaussians can be computed exactly
    return log_p.exp() * (log_p - log_q)


class NoiseScheduler(nn.Module):
    def __init__(
        self,
        num_timesteps: int,
        sigma_min: float = 0.01,
        sigma_max: float = 0.5,
        device: torch.device = "cpu",
    ):
        """
        Exponential scheduler:
            sigma_0 = 0                                             if t = 0
            sigma_t = sigma_1 * (sigma_T / sigma_1)^(t-1 / T-1)     if t > 0
        """
        super().__init__()
        self.num_timesteps = num_timesteps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(sigma_min), np.log(sigma_max), num_timesteps)),
            dtype=torch.float, device=device
        )
        # (num_timesteps,)
        sigmas_norm_ = sigma_norm(sigmas)

        self.register_buffer(
            "sigmas", torch.cat([torch.zeros([1], device=sigmas.device), sigmas], dim=0)
        )  # (1 + num_timesteps,)
        self.register_buffer(
            "sigmas_norm", torch.cat([torch.ones([1]), sigmas_norm_], dim=0)
        )

    def uniform_sample_timestep(self, batch_size: int, device: torch.device):
        return torch.randint(
            low=1,
            high=self.num_timesteps+1,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )


class TorusMLP(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_freqs: int = 16, ndim: int = 2):
        super().__init__()
        self.register_buffer("freqs", torch.linspace(1, num_freqs, num_freqs))
        self.layers = nn.Sequential(
            nn.Linear(2 * ndim * num_freqs, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, x: Tensor):
        """x: (batch_size, ndim)"""
        f = self.freqs[None, None, :] * x[..., None]
        # (batch_size, ndim, freqs)
        f = torch.cat([f.sin(), f.cos()], dim=-1)
        # (batch_size, ndim, 2 * freqs)
        return self.layers(f.view(x.shape[0], -1))  # (batch_size, hidden_dim)


class DiffusionModel(nn.Module):
    def __init__(
        self,
        num_timesteps: int = 1000,
        hidden_dim: int = 64,
        time_dim: int = 64,
        spatial_dim: int = 2,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.spatial_dim = spatial_dim
        self.noise_scheduler = NoiseScheduler(num_timesteps)

        self.pos_emb = TorusMLP(ndim=spatial_dim)
        self.time_emb = FourierTimeEmbeddings(time_dim)
        self.drift_model = nn.Sequential(
            nn.Linear(hidden_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, spatial_dim)
        )

    def forward(self, x, t):
        batch_size: int = x.shape[0]
        if isinstance(t, Tensor):
            float_t = t.float()
        elif isinstance(t, int):
            float_t = torch.tensor([t], dtype=torch.float, device=x.device)
        else:
            raise AttributeError

        t_emb = self.time_emb(float_t).expand(batch_size, -1)
        x_emb = self.pos_emb(x)
        out = self.drift_model(torch.cat([x_emb, t_emb], dim=-1))
        # (n, d)
        return out

    def compute_score_matching_loss(self, x: Tensor):
        batch_size = x.shape[0]
        device = x.device

        # ------------ Sample noise level
        sampled_timesteps = self.noise_scheduler.uniform_sample_timestep(
            batch_size=x.shape[0], device=device
        )
        # (batch_size,)
        sigmas = self.noise_scheduler.sigmas[sampled_timesteps].view(*x.shape)
        # (batch_size, spatial_dim)
        sigmas_norm = self.noise_scheduler.sigmas_norm[sampled_timesteps].view(*x.shape)

        noise = torch.randn_like(x)
        with torch.no_grad():
            noisy_x = (x + sigmas * noise) % 1.0
            ground_truth_scores = d_log_p_wrapped_multivariate_normal(
                sigmas * noise, sigmas
            )

        predicted_scores = self(noisy_x, sampled_timesteps)
        loss = F.mse_loss(
            predicted_scores, ground_truth_scores / (torch.sqrt(sigmas_norm) + 1e-8)
        )
        return loss

    def trajectory_log_probs(self, trajectories: Tensor):
        """
        Args:
            trajectories: torch.float
                shape (batch_size, 1+num_timesteps, 2)

        Returns:
            (batch_size,)
        """
        num_timesteps: int = trajectories.shape[1]
        batch_size: int = trajectories.shape[0]
        device = trajectories.device

        x_t = trajectories[:, 1:]
        # (batch_size, num_timesteps, spatial_dim)
        x_t_min1 = trajectories[:, :-1]
        # (batch_size, num_timesteps, spatial_dim)
        ts = torch.arange(1, num_timesteps, device=device, dtype=torch.long)
        # (num_timesteps,)

        all_step_log_probs, kl_div = self.log_prob_step(
            x_t.reshape(-1, self.spatial_dim),
            x_t_min1.reshape(-1, self.spatial_dim),
            ts.repeat(batch_size),
        )
        all_step_log_probs = all_step_log_probs.view(batch_size, num_timesteps-1)
        kl_div = kl_div.view(batch_size, num_timesteps-1).sum(dim=-1)
        return all_step_log_probs, kl_div

    def log_prob_step(self, x_t_plus1: Tensor, x_t: Tensor, t: Tensor):
        """
        Args:
            x_t_plus1: torch.float
                (n, 2)
            x_t: torch.float
                (n, 2)
            t: torch.float
                (n,)

        Returns:
            (n,)
        """
        device = x_t.device
        batch_size = x_t.shape[0]

        sigma_norm_t_plus_1 = self.noise_scheduler.sigmas_norm[t+1][:, None]
        sigma_norm_t = self.noise_scheduler.sigmas_norm[t][:, None]
        # (n, 1)

        # --- Predictor
        # x_t <- x_t+1 + (sigma_t+1 ** 2 - sigma_t ** 2) * score(x_t+1, sigma_t+1)
        # z ~ N(0, I)
        # x_t <- x_t + sqrt(sigma_t+1 ** 2 - sigma_t ** 2) * z
        predicted_score = torch.sqrt(sigma_norm_t_plus_1) * self(x_t_plus1, t + 1)

        sigma_sq_diff = (
            self.noise_scheduler.sigmas[t+1] ** 2
            - self.noise_scheduler.sigmas[t] ** 2
        )[:, None]  # (n, 1)

        mean = x_t_plus1 + sigma_sq_diff * predicted_score
        std = torch.sqrt(sigma_sq_diff)
        step_log_probs = wrapped_gaussian_log_pdf(x_t.detach(), mean, std)
        # todo: use a pre-trained model
        with torch.no_grad():
            ref_log_probs = wrapped_gaussian_log_pdf(x_t, x_t_plus1, std)
        kl_div = kl_divergence(step_log_probs, ref_log_probs)
        return step_log_probs, kl_div

    def get_sample_and_kl_loss(
        self, batch_size: int, device: torch.device = "cpu", init_same_noise: bool = False
    ) -> (Tensor, Tensor, Tensor, Tensor, Tensor):
        """For online training"""
        num_timesteps = self.noise_scheduler.num_timesteps

        t = num_timesteps-1
        if init_same_noise:  # DanceGRPO claims this stabilizes things
            x_t = torch.zeros(1, self.spatial_dim, device=device).expand(batch_size, -1)
        else:
            x_t = torch.rand(batch_size, self.spatial_dim, device=device)

        kl_loss = torch.zeros(batch_size, device=device)
        log_probs = torch.zeros(batch_size, device=device)
        ref_log_probs = torch.zeros(batch_size, device=device)
        trajectories = torch.full(
            [batch_size, num_timesteps, self.spatial_dim],
            fill_value=-1.0, device=device
        )
        trajectories[:, t] = x_t

        step_log_probs = []  # todo
        ts_used = []  # todo
        for step in range(num_timesteps-1):
            (
                x_t_min1,           # (batch_size, 2)
                step_log_prob,      # (batch_size,)
                ref_step_log_prob,  # (batch_size,)
            ) = self.sample_and_log_prob_step(x_t, t)
            step_log_probs.append(step_log_prob)
            ts_used.append(t)

            log_probs = log_probs + step_log_prob
            ref_log_probs = ref_log_probs + ref_step_log_prob
            kl_loss = kl_loss + kl_divergence(step_log_prob, ref_step_log_prob)
            trajectories[:, t-1] = x_t_min1

            x_t = x_t_min1
            t = t-1

        # print(ts_used)

        x_0 = x_t
        # (batch_size, 2)
        return x_0, log_probs, ref_log_probs, kl_loss, trajectories, torch.flip(torch.stack(step_log_probs, dim=-1), dims=[-1])

    def sample_and_log_prob_step(
        self,
        x: Tensor,
        t: int,
        snr: float = 0.4,
        max_step_size: float = 1e6,
        greedy_decoding: bool = False,
    ) -> (Tensor, Tensor, Tensor):

        sigma_norm_t_plus_1 = self.noise_scheduler.sigmas_norm[t+1]
        sigma_norm_t = self.noise_scheduler.sigmas_norm[t]

        # --- Predictor
        # x_t <- x_t+1 + (sigma_t+1 ** 2 - sigma_t ** 2) * score(x_t+1, sigma_t+1)
        # z ~ N(0, I)
        # x_t <- x_t + sqrt(sigma_t+1 ** 2 - sigma_t ** 2) * z
        float_t = torch.tensor([t+1], dtype=torch.float, device=x.device)
        predicted_score = torch.sqrt(sigma_norm_t_plus_1) * self(x, float_t)

        sigma_sq_diff = (
            self.noise_scheduler.sigmas[t+1] ** 2
            - self.noise_scheduler.sigmas[t] ** 2
        )  # (,)
        noise = torch.randn_like(x)

        x_t_plus1 = x
        mean = x + sigma_sq_diff * predicted_score
        std = torch.sqrt(sigma_sq_diff)
        if greedy_decoding:
            x = mean
        else:
            x = mean + std * noise
        step_log_prob = wrapped_gaussian_log_pdf(x.detach(), mean, std)
        # todo: use a pre-trained model
        with torch.no_grad():
            ref_step_log_prob = wrapped_gaussian_log_pdf(x, x_t_plus1, std)

        if not self.training:
            # --- Corrector: Algo 4 in https://arxiv.org/pdf/2011.13456
            # z ~ N(0, I)
            # x_t <- x_t + eps_t * score(x_t, sigma_t) + sqrt(2 * eps_t) * z
            with torch.no_grad():
                # Corrector is meant to correct numerical errors from SDE sampling,
                # not to be used for training
                predicted_score = torch.sqrt(sigma_norm_t) * self(x, float_t)
                noise = torch.randn_like(x)

                noise_norm = (noise ** 2).sum(dim=-1).sqrt().mean()
                grad_norm = (predicted_score ** 2).sum(dim=-1).sqrt().mean()
                step_size = 2 * (snr * noise_norm / grad_norm).pow(2)
                step_size.nan_to_num_(nan=0., posinf=max_step_size, neginf=-max_step_size)

                mean = x + step_size * predicted_score
                std = torch.sqrt(2 * step_size)
                x = mean + std * noise
        return x % 1.0, step_log_prob, ref_step_log_prob

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: torch.device = "cpu",
        initial_state: Optional[Tensor] = None,
        greedy_decoding: bool = False,  # denoising steps predict expectation
    ) -> Tensor:
        _training = self.training
        self.training = False

        num_timesteps = self.noise_scheduler.num_timesteps
        t = num_timesteps-1
        if initial_state is None:
            x = torch.rand(batch_size, self.spatial_dim, device=device)
        else:
            x = initial_state

        for step in range(num_timesteps-1):
            x = self.sample_and_log_prob_step(x, t, greedy_decoding=greedy_decoding)[0]
            t = t-1

        self.training = _training
        return x


def compute_loss(
    method, log_probs, kl_loss, rewards, beta, old_log_probs, ratio_clip=1e-1
):
    # https://github.com/XueZeyue/DanceGRPO/blob/d97950b51def6e61fddda83b0dbcbc615b07997c/fastvideo/train_grpo_flux.py#L597
    ratio = torch.exp(log_probs - old_log_probs.detach()).clamp(
        min=1. - ratio_clip, max=1. + ratio_clip
    )
    unclipped_loss = -rewards * ratio
    clipped_loss = -rewards * torch.clamp(ratio, 1.0 - ratio_clip, 1. + ratio_clip)
    first_term = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

    kl_term = kl_loss.sum()
    loss = first_term + beta * kl_term
    artifacts = {
        "first term": first_term.detach(),
        "kl_loss": kl_term.detach(),
    }
    return loss, artifacts


@torch.no_grad()
def plot_current_policy(
    model, reward_fn, device, method: str, epoch: int
):
    samples = model.sample(100, device)
    if model.spatial_dim == 2:
        grid_x, grid_y = torch.meshgrid(torch.linspace(0, 1, steps=25), torch.linspace(0, 1, steps=25))
        grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1).to(device)
        target_dist = reward_fn(grid)
        plt.contourf(grid_x.cpu(), grid_y.cpu(), target_dist.view(25, 25).cpu())

        plt.plot(samples[:, 0], samples[:, 1], "rx", alpha=0.5)
        plt.savefig(output_dir / f"diffusion_rl_{method}_epoch{epoch}.png")
        plt.close()
    elif model.spatial_dim == 1:
        grid_x = torch.linspace(0, 1, steps=25).view(-1, 1)
        target_dist = reward_fn(grid_x)
        plt.plot(samples, torch.zeros_like(samples), "rx", alpha=0.5)
        plt.plot(grid_x, target_dist)
        plt.savefig(output_dir / f"diffusion_rl_{method}_epoch{epoch}.png")
        plt.close()


def plot_learning_curves(learning_curves, method):
    for k, v in learning_curves.items():
        learning_curves[k] = torch.stack(v).detach().cpu()

    xs = list(range(len(learning_curves["loss"])))
    fig, axs = plt.subplots(ncols=3)
    axs[0].plot(xs, learning_curves["loss"], label="loss")
    axs[1].plot(xs, learning_curves["reward"], label="reward")
    axs[2].plot(xs, learning_curves["grad_norm"], label="||grad||")
    axs[0].legend(frameon=False)
    axs[1].legend(frameon=False)
    axs[2].legend(frameon=False)
    plt.savefig(output_dir / f"diffusion_rl_curves_{method}.png")
    plt.close()

    plt.plot(xs, learning_curves["reward"])
    plt.ylabel("Reward")
    plt.xlabel("Iteration")
    plt.savefig(output_dir / "reward.png")
    plt.close()


@torch.no_grad()
def plot_noise_scheduler_pdfs(model):
    scheduler = model.noise_scheduler

    xs = torch.linspace(0, 1, steps=99)
    stds = scheduler.sigmas[None, None, :]

    m = 5
    translations = torch.arange(-m, m+1)
    mu = torch.zeros_like(xs) + 0.5
    mu = mu[:, None] + translations[None, :]
    # (num_xs, n_translations)
    log_pdf = -0.5 * (
        math.log(2. * math.pi)
        + 2 * torch.log(stds)
        + (xs[:, None, None] - mu[..., None]).pow(2) / (stds.pow(2) + 1e-6)
    )
    log_pdf = torch.logsumexp(log_pdf, dim=1)
    # (num_xs, num_sigmas)

    cmap = plt.get_cmap('Blues')
    idxs = torch.round(torch.linspace(0, stds.numel()-1, steps=10)).long()
    for i in idxs:
        pdf_i = log_pdf[:, i].exp()
        pdf_i = pdf_i - pdf_i.min()
        plt.plot(
            xs,
            pdf_i,
            label=f"$\sigma$={stds[0, 0, i]:0.3f}, step={i}/{stds.numel()}",
            # color=cmap((i + 1) / (idxs.max() + 1)),
        )
    plt.legend(frameon=False)
    plt.xlabel("x")
    plt.ylabel("Unnormalized PDF")
    plt.savefig("wrapped_normal_pdfs.png")
    plt.close()


def pretrain():
    # pretrain model with score matching
    if os.path.exists(pretrained_model_path):
        return

    lr = 1e-4

    model = DiffusionModel(**model_kwargs)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-5)

    # - Generate data from bimodal distribution. One mode will be the reward
    # distribution, and the other will later get zero reward.
    use_sparse_reward = False
    if spatial_dim == 2:
        energy_fn: nn.Module = get_energy_function(
            device, sparse=use_sparse_reward, discrete_energy=False
        )
        dist1 = energy_fn.gmm
        dist2 = Normal(
            torch.tensor([-0.5, -0.5]).view(1, 2),
            torch.tensor([0.1, 0.1]).view(1, 2)
        )
    elif spatial_dim == 1:
        dist1 = torch.distributions.Normal(torch.tensor([0.]), torch.tensor([0.05]))
        dist2 = Normal(torch.tensor([0.5]), torch.tensor(0.05))

    dataset_size = 100
    data = torch.cat(
        [
            dist1.sample((math.ceil(dataset_size / 2),)),
            dist2.sample((math.floor(dataset_size / 2),))
        ], dim=0
    ) % 1.0
    # (dataset_size, spatial_dim)

    # - Plot untrained model
    with torch.no_grad():
        samples = model.sample(100, device)
        if model.spatial_dim == 2:
            plt.plot(samples[:, 0], samples[:, 1], "bo", alpha=0.5, label="samples")
            plt.plot(data[:, 0], data[:, 1], "rx", alpha=0.2, label="data")
        elif model.spatial_dim == 1:
            plt.plot(samples[:, 0], torch.zeros(samples.shape[0]), "bo", alpha=0.5, label="samples")
            plt.plot(data[:, 0], .1 + torch.zeros(dataset_size), "rx", alpha=0.2, label="data")
        plt.legend(frameon=False)
        plt.savefig(parent_dir / f"untrained_model_samples.png")
        plt.close()

    # - Train with score matching, plot samples
    n_epochs = 1001
    batch_size = 32
    log_freq = 100
    num_batches_per_epoch = math.ceil(dataset_size / batch_size)
    losses = []
    loss_meter = AverageMeter()
    for epoch in range(n_epochs):
        shuffle_idxs = torch.randperm(dataset_size, device=device)
        shuffled_data = data[shuffle_idxs]
        for batch_idx in range(num_batches_per_epoch):
            batch = shuffled_data[batch_idx * batch_size : (batch_idx+1) * batch_size]
            # (batch_size, spatial_dim)
            loss = model.compute_score_matching_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            loss_meter.update(float(loss))
        if epoch % log_freq == 0:
            print(f"Epoch {epoch}: {loss_meter.avg:0.4f}")
            losses.append(loss_meter.avg)
            scheduler.step(loss_meter.avg)
            loss_meter.reset()

    # - Plot samples
    with torch.no_grad():
        plt.plot(log_freq * np.array(list(range(len(losses)))), losses)
        plt.xlabel("Epoch")
        plt.ylabel("SM loss")
        plt.savefig(parent_dir / f"pretrain_loss.png")
        plt.close()

        samples = model.sample(100, device)
        if model.spatial_dim == 2:
            plt.plot(samples[:, 0], samples[:, 1], "bo", alpha=0.5, label="samples")
            plt.plot(data[:, 0], data[:, 1], "rx", alpha=0.2, label="data")
        elif model.spatial_dim == 1:
            plt.plot(samples[:, 0], torch.zeros(samples.shape[0]), "bo", alpha=0.5, label="samples")
            plt.plot(data[:, 0], .1 + torch.zeros(dataset_size), "rx", alpha=0.2, label="data")
        plt.legend(frameon=False)
        plt.savefig(parent_dir / f"pretrained_model_samples.png")
        plt.close()

    # - Save model
    torch.save(model.state_dict(), pretrained_model_path)


def run_dpok(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_iterations: int,
    normalize_advantages: bool,
    batch_size: int,
    init_same_noise: bool,
    reward_fn: Callable,
    use_ema: bool,
    ema_decay: float,
    num_inner_steps: int,
    inner_batch_size: int,
    old_policy_update_freq: int,
    log_freq: int,
    beta: float,
    epsilon: float,
):
    old_policy = DiffusionModel(**model_kwargs)
    old_policy.load_state_dict(model.state_dict())
    old_policy.train()
    for p in old_policy.parameters():
        p.requires_grad = False
    old_policy.to(device)

    # -- Train
    meters = {"loss": AverageMeter(), "reward": AverageMeter(), "grad": AverageMeter()}
    learning_curves = {"reward": [], "loss": [], "grad_norm": []}
    for iter in range(num_iterations):
        # sample N points from current model, take num_inner_steps
        #  gradient steps on them with importance sampling
        with torch.no_grad():
            (
                samples,        # (batch_size, 2)
                old_log_probs,      # (batch_size,)
                ref_log_probs,  # (batch_size,)
                _,        # (batch_size,)
                trajectories,   # (batch_size, 1+num_timesteps, 2)
                step_old_log_probs,  # (batch_size, num_timesteps)
            ) = old_policy.get_sample_and_kl_loss(
                batch_size, device, init_same_noise
            )
            rewards = reward_fn(samples)  # (batch_size,)
            if normalize_advantages:
                advantages = (rewards - rewards.mean()) / (rewards.std().nan_to_num_(0.) + 1e-6)
            else:
                advantages = rewards - rewards.mean()

        # Sanity check
        if iter == 0:
            with torch.no_grad():
                step_current_log_probs = model.trajectory_log_probs(trajectories)[0]
                current_log_probs = step_current_log_probs.sum(dim=-1)
            assert torch.isclose(current_log_probs, old_log_probs, atol=1e-5).all()

        for step in range(num_inner_steps):
            idx1 = step * inner_batch_size
            idx2 = (step+1) * inner_batch_size
            step_trajectories = trajectories[idx1:idx2]
            step_advantages = advantages[idx1:idx2]
            step_old_log_probs = old_log_probs[idx1:idx2]

            current_step_log_probs, kl_loss = model.trajectory_log_probs(
                step_trajectories
            )
            # (batch_size, num_timesteps)
            current_log_probs = current_step_log_probs.sum(dim=-1)

            loss, artifacts = compute_loss(
                method,
                current_log_probs,
                kl_loss,
                step_advantages,
                beta,
                step_old_log_probs,
                epsilon,
            )
            loss = loss / model.num_timesteps
            # loss = current_step_log_probs.sum()  # sanity check grad flow

            loss.backward()

            # grad norm clipping value not super sensitive (0.1 and 1.0 both work)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            # # grad value clip values (1.0 works but very slow convergence,
            # # 0.1 unstable)
            # torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
            # grad_norm = compute_grad_norm(model)

            optimizer.step()
            optimizer.zero_grad()

            _loss = float(loss.detach().cpu())
            _rewards = rewards.detach().mean().cpu()
            meters["loss"].update(_loss)
            meters["reward"].update(_rewards)
            meters["grad"].update(grad_norm.detach())
            learning_curves["loss"].append(artifacts['first term'].detach())
            learning_curves["reward"].append(_rewards)
            learning_curves["grad_norm"].append(grad_norm)
        if iter % old_policy_update_freq == 0:
            update_old_policy(old_policy, model, use_ema=use_ema, decay=ema_decay)
        if iter % log_freq == 0:
            metric_str = (
                f"Iter [{iter}/{num_iterations}]: "
                f"loss {meters['loss'].avg:0.4f}, "
                f"reward {meters['reward'].avg:0.4f}, "
                f"first term: {artifacts['first term']:0.4f}, "
                f"kl loss: {artifacts['kl_loss']:0.4f}, "
                f"grad norm: {meters['grad'].avg:0.8f}"
            )
            print(metric_str)
            for meter in meters.values():
                meter.reset()
    plot_learning_curves(learning_curves, method)


def process_es_worker(
    xT: Tensor,
    seed_idxs: List[int],
    seeds: List[int],
    worker_model: nn.Module,
    rewards: List[float],
    barrier: mp.Barrier,
    noise_scale: float,
    reward_fn: Callable,
    num_iterations: int,
):
    worker_id = mp.current_process()._identity[0] - 1  # starts at 1
    for iter in range(num_iterations):
        for seed_idx, seed in zip(seed_idxs, seeds):
            # Noise model in-place
            for name, param in worker_model.named_parameters():
                gen = torch.Generator(device=param.device)
                gen.manual_seed(int(seed))
                param_noise = noise_scale * torch.randn(
                    param.shape,
                    generator=gen,
                    device=param.device,
                    dtype=param.dtype,
                )
                param.data.add_(param_noise)
                del param_noise

            # Sample from noised model
            samples = worker_model.sample(
                None, device, initial_state=xT, greedy_decoding=True
            )
            avg_reward = reward_fn(samples).mean()

            # Restore original weights in-place
            for name, param in worker_model.named_parameters():
                gen = torch.Generator(device=param.device)
                gen.manual_seed(int(seed))
                param_noise = noise_scale * torch.randn(
                    param.shape,
                    generator=gen,
                    device=param.device,
                    dtype=param.dtype,
                )
                param.data.add_(-param_noise)
                del param_noise

            # Update results
            rewards[seed_idx] = avg_reward

        barrier.wait()  # wait for all workers to finish the round
        barrier.wait()  # wait for main process to update seeds + base model


@torch.no_grad()
def run_evolutionary_search(
    base_model: nn.Module,
    num_iterations: int,
    reward_fn: Callable,
    log_freq: int,
):
    """https://arxiv.org/pdf/2509.24372"""
    population_size = 12  # default is 30
    noise_scale = 0.01
    learning_rate = noise_scale / 2
    batch_size_per_seed = 32

    use_gpu = torch.cuda.is_available()
    nprocs = torch.cuda.device_count() if use_gpu else min(12, os.cpu_count())
    print(f"Num processes: {nprocs}/{os.cpu_count()}")

    # Initialize seeds
    seeds = np.random.randint(
        0, 2**30, size=population_size, dtype=np.int64
    ).tolist()
    seed_idxs = list(range(population_size))
    if len(seeds) < nprocs:
        raise NotImplementedError

    # Copy model to each process
    model_list = []
    for _ in range(nprocs):
        worker_model = DiffusionModel(**model_kwargs).to(device)
        worker_model.load_state_dict(base_model.state_dict())
        model_list.append(worker_model)

    # Start workers
    manager = mp.Manager()
    workers = []
    rewards = manager.list([None] * population_size)  # thread-safe container for outputs
    barrier = mp.Barrier(nprocs + 1)  # +1 for main process
    xT = torch.rand(batch_size_per_seed, base_model.spatial_dim, device=device)
    for worker_idx in range(nprocs):
        p = mp.Process(
            target=process_es_worker,
            args=(
                xT,
                seed_idxs[worker_idx::nprocs],
                seeds[worker_idx::nprocs],
                model_list[worker_idx],
                rewards,
                barrier,
                noise_scale,
                reward_fn,
                num_iterations,
            ),
        )
        p.start()
        workers.append(p)

    rewards_meter = AverageMeter()
    max_rewards, min_rewards, mean_rewards = [], [], []
    for iter in range(num_iterations):
        # Wait for all workers to finish this round
        barrier.wait()

        # Normalize reward
        rewards_tensor = torch.tensor(rewards, device=device)
        r_mean = rewards_tensor.mean()
        r_std = rewards_tensor.std()
        r_max = rewards_tensor.max()
        r_min = rewards_tensor.min()
        rewards_normalized = (rewards_tensor - r_mean) / (r_std + 1e-8)
        max_rewards.append(r_max)
        min_rewards.append(r_min)
        mean_rewards.append(r_mean)
        rewards_meter.update(r_mean)
        if iter % log_freq == 0:
            print(f"Iter {iter} reward avg: {rewards_meter.avg}, {rewards}")
            rewards_meter.reset()

        # update base model
        for name, param in base_model.named_parameters():
            gen = torch.Generator(device=param.device)
            update = torch.zeros_like(param)
            for seed_idx in range(population_size):
                r_norm = rewards_normalized[seed_idx]
                seed = seeds[seed_idx]
                gen.manual_seed(int(seed))
                noise = torch.randn(
                    param.shape,
                    generator=gen,
                    device=param.device,
                    dtype=param.dtype
                )
                noise.mul_(float(r_norm))
                update.add_(noise)
                del noise
            update.div_(population_size)
            param.data.add_(learning_rate * update)

        # update worker models with new base model
        for model_idx in range(nprocs):
            worker_model = model_list[model_idx]
            for name, param in worker_model.named_parameters():
                param.data.copy_(base_model.get_parameter(name).data.clone())

        # Update seeds and xT
        _xT = torch.rand(batch_size_per_seed, base_model.spatial_dim, device=xT.device)
        xT.mul_(0.)
        xT.add_(_xT)
        seeds = np.random.randint(
            0, 2**30, size=population_size, dtype=np.int64
        ).tolist()

        # Allow all workers to load the new base model
        barrier.wait()

    # Finish processes
    for worker in workers:
        worker.join()

    xs = list(range(num_iterations))
    plt.plot(xs, mean_rewards, 'k-')
    plt.plot(xs, min_rewards, 'k--')
    plt.plot(xs, max_rewards, 'k--')
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.savefig("es_rewards.png")
    plt.close()


def get_nll_from_dist(x, dist):
    return -dist.log_prob(x)


def run(method: str, use_pretrained_model: bool = True):
    assert method in ["dpok", "es"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if method == "es":
        mp.set_start_method("spawn", force=True)

    # hyperparameters
    lr = 5e-5
    num_iterations = 1_001
    batch_size = 64
    use_sparse_reward = False
    reward_temperature = 1.0
    use_ema = False  # ema leads to no learning
    ema_decay = 0.99
    old_policy_update_freq = 1 if use_ema else 100
    epsilon = 0.1  # 0.2 instead of 0.1 leads to slow learning
    beta = 0.0
    n_log_prints = 10
    num_inner_steps = 4
    normalize_advantages = False
    log_freq = int(num_iterations / n_log_prints)
    inner_batch_size = int(batch_size / num_inner_steps)
    init_same_noise = False  # DanceGRPO claims this stabilizes training
    assert batch_size % num_inner_steps == 0
    assert batch_size > 1

    config = {
        "seed": seed,
        "lr": lr, "num_iterations": num_iterations, "batch_size": batch_size,
        "use_sparse_reward": use_sparse_reward,
        "reward_temperature": reward_temperature,
        "use_ema": use_ema, "old_policy_update_freq": old_policy_update_freq,
        "epsilon": epsilon, "beta": beta, "num_inner_steps": num_inner_steps,
        "num_timesteps": num_timesteps, "hidden_dim": hidden_dim,
        "spatial_dim": spatial_dim, "use_pretrained_model": use_pretrained_model,
        "normalize_advantages": normalize_advantages,
        "ema_decay": ema_decay,
    }
    print(config)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f)

    if spatial_dim == 2:
        energy_fn: nn.Module = get_energy_function(
            device, sparse=use_sparse_reward, discrete_energy=False
        )
    elif spatial_dim == 1:
        dist = torch.distributions.Normal(torch.tensor([0.5]), torch.tensor([0.05]))
        energy_fn: Callable = functools.partial(get_nll_from_dist, dist=dist)
    else:
        raise AttributeError
    reward_fn: Callable = functools.partial(
        wrapped_reward_fn,
        energy_fn=energy_fn,
        reward_temperature=reward_temperature,
    )

    # -- Model
    model = DiffusionModel(**model_kwargs)
    model.to(device)
    model.train()
    if use_pretrained_model and os.path.exists(pretrained_model_path):
        print("Loaded pretrained model")
        model.load_state_dict(torch.load(pretrained_model_path, weights_only=True))

    # -- Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -- Plot samples against reward
    plot_noise_scheduler_pdfs(model)
    plot_current_policy(model, reward_fn, device, method, 0)

    if method == "dpok":
        run_dpok(
            model, optimizer, num_iterations, normalize_advantages,
            batch_size, init_same_noise, reward_fn, use_ema, ema_decay,
            num_inner_steps, inner_batch_size, old_policy_update_freq,
            log_freq, beta, epsilon,
        )
    elif method == "es":
        run_evolutionary_search(model, num_iterations, reward_fn, log_freq)
    else:
        raise AttributeError

    # plot samples against reward
    plot_current_policy(model, reward_fn, device, method, num_iterations)


if __name__ == "__main__":
    # use_pretrained_model=True leads to worse performance
    use_pretrained_model = False
    method = "es"
    assert method in ["dpok", "es"]
    if use_pretrained_model:
        pretrain()
    run(method=method, use_pretrained_model=use_pretrained_model)

    # from cProfile import Profile
    # from pstats import SortKey, Stats
    #
    # with Profile() as profile:
    #      run(method="dpok")
    #      (
    #          Stats(profile)
    #          .strip_dirs()
    #          .sort_stats(SortKey.TIME)
    #          .print_stats()
    #      )
