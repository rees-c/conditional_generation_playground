"""DDPM style with standard gaussian (unwrapped)"""
seed = 42
import torch
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)
import numpy as np
np.random.seed(seed)
import random
random.seed(seed)

import os
import math
from typing import *
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt

from common_utils import AverageMeter
from rewards import get_energy_function
from grpo.utils import update_old_policy
from diffusion_grpo.model_utils import FourierTimeEmbeddings

Tensor: type = torch.Tensor

output_dir = Path(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
pretrained_model_path = "ckpt.pt"

# Model hyperparams
num_timesteps = 100
hidden_dim = 64
spatial_dim = 1
assert spatial_dim in [1, 2]


def gaussian_log_pdf(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """
    Args:
        x: (batch_size, 2)
        mean: (batch_size, 2)
        std: (,)

    Returns:
        (batch_size,)
    """
    ndim = x.shape[-1]
    std = std.clamp(min=1e-6)
    return -0.5 * (
        ndim * math.log(2. * math.pi)
        + 2 * ndim * torch.log(std)
        + (x - mean).pow(2).sum(dim=-1) / (std.pow(2))
    )


def kl_divergence(log_p: Tensor, log_q: Tensor) -> Tensor:
    # todo: use the lower-variance Rao-Blackwellized estimator instead
    #  (https://arxiv.org/pdf/2504.10637)
    #  KL divergence between Gaussians can be computed exactly
    return log_p.exp() * (log_p - log_q)


class BetaScheduler(nn.Module):
    def __init__(
        self,
        num_timesteps: int,
        scheduler_mode: str = "cosine",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        super(BetaScheduler, self).__init__()
        self.num_timesteps = num_timesteps
        if scheduler_mode == 'cosine':
            # DiffCSP++ uses s=0.008
            betas = self.cosine_beta_schedule(num_timesteps, s=0.008)
        elif scheduler_mode == 'linear':
            betas = self.linear_beta_schedule(num_timesteps, beta_start, beta_end)
        else:
            raise AttributeError

        betas = torch.cat([torch.zeros([1]), betas], dim=0)
        alphas = 1. - betas
        alpha_bar = torch.cumprod(alphas, axis=0)  # \bar{alpha}

        sigmas = torch.zeros_like(betas)

        # Eqn 4 in DiffCSP++. Standard dev of the noise.
        sigmas[1:] = betas[1:] * (1. - alpha_bar[:-1]) / (1. - alpha_bar[1:])
        sigmas = torch.sqrt(sigmas)

        self.register_buffer('betas', betas)
        # (1 + timesteps,)
        self.register_buffer('alphas', alphas)
        # (1 + timesteps,)
        self.register_buffer('alpha_bar', alpha_bar)
        # (1 + timesteps,)
        self.register_buffer('sigmas', sigmas)
        # (1 + timesteps,)

    @staticmethod
    def cosine_beta_schedule(timesteps: int, s=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)    # shape (timesteps,)

    @staticmethod
    def linear_beta_schedule(timesteps: int, beta_start: float, beta_end: float):
        return torch.linspace(beta_start, beta_end, timesteps)

    def uniform_sample_timestep(self, batch_size: int, device: torch.device):
        return torch.randint(
            low=1,
            high=self.num_timesteps+1,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )


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
        self.noise_scheduler = BetaScheduler(num_timesteps)

        self.pos_emb = nn.Linear(spatial_dim, hidden_dim)
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
        pred_noise = self.drift_model(torch.cat([x_emb, t_emb], dim=-1))
        # (n, d)
        return pred_noise

    def compute_score_matching_loss(self, x0: Tensor):
        """
        x0: shape (batch_size, spatial_dim)
        """
        batch_size = x0.shape[0]
        device = x0.device

        times: Tensor = self.noise_scheduler.uniform_sample_timestep(batch_size, device)
        alpha_bar = self.noise_scheduler.alpha_bar[times]
        # (batch_size,)
        # \sqrt{\bar{\alpha}}
        c0 = torch.sqrt(alpha_bar)
        # \sqrt{1-\bar{\alpha}}
        c1 = torch.sqrt(1. - alpha_bar)

        # q(kt|k0) = N(sqrt(alpha_bar)k0, (1-alpha_bar)*I)
        noise = torch.randn_like(x0)
        xt = c0[:, None] * x0 + c1[:, None] * noise

        pred_noise = self(xt, times)
        loss = F.mse_loss(pred_noise, noise)
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
            t: torch.long
                (n,)

        Returns:
            (n,)
        """
        alpha = self.noise_scheduler.alphas[t]
        alpha_bar = self.noise_scheduler.alpha_bar[t]
        sigma = self.noise_scheduler.sigmas[t]

        # 1/sqrt(alpha) = 1/sqrt(1-beta)
        c0 = 1.0 / torch.sqrt(alpha)
        # beta / sqrt(1-\bar{alpha})
        c1 = (1. - alpha) / torch.sqrt(1 - alpha_bar)

        # --- Predictor
        # x_t <- x_t+1 + (sigma_t+1 ** 2 - sigma_t ** 2) * score(x_t+1, sigma_t+1)
        # z ~ N(0, I)
        # x_t <- x_t + sqrt(sigma_t+1 ** 2 - sigma_t ** 2) * z
        pred_noise = self(x_t_plus1, t + 1)
        # (n, d)

        """
        noise = torch.randn_like(x_t_plus1)
        x_t = c0 * (x_t_plus1 - c1 * pred_noise) + sigma * noise
        """
        mean = c0[:, None] * (x_t_plus1 - c1[:, None] * pred_noise)
        std = sigma

        step_log_probs = gaussian_log_pdf(x_t.detach(), mean, std)
        with torch.no_grad():
            # todo: use a pre-trained model
            ref_log_probs = gaussian_log_pdf(x_t, x_t_plus1, std)
        kl_div = kl_divergence(step_log_probs, ref_log_probs)
        return step_log_probs, kl_div

    def get_sample_and_kl_loss(
        self, batch_size: int, device: torch.device = "cpu", init_same_noise: bool = False
    ) -> (Tensor, Tensor, Tensor, Tensor, Tensor):
        """For online training"""
        num_timesteps = self.noise_scheduler.num_timesteps

        t = num_timesteps-1
        if init_same_noise:  # DanceGRPO claims this stabilizes things
            x_t = torch.rand(1, self.spatial_dim, device=device).expand(batch_size, -1)
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

        # print(x_t)
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

        # print(torch.stack(step_log_probs, dim=-1))
        # print(torch.tensor(ts_used))
        # print(trajectories.view(-1))

        x_0 = x_t
        # (batch_size, 2)
        return x_0, log_probs, ref_log_probs, kl_loss, trajectories, torch.flip(torch.stack(step_log_probs, dim=-1), dims=[-1])

    def sample_and_log_prob_step(self, x: Tensor, t: int) -> (Tensor, Tensor, Tensor):

        alpha = self.noise_scheduler.alphas[t]
        alpha_bar = self.noise_scheduler.alpha_bar[t]
        sigma = self.noise_scheduler.sigmas[t]

        # 1/sqrt(alpha) = 1/sqrt(1-beta)
        c0 = 1.0 / torch.sqrt(alpha)
        # beta / sqrt(1-\bar{alpha})
        c1 = (1. - alpha) / torch.sqrt(1 - alpha_bar)  # todo: this is nan at t=0

        # --- Predictor
        # x_t <- x_t+1 + (sigma_t+1 ** 2 - sigma_t ** 2) * score(x_t+1, sigma_t+1)
        # z ~ N(0, I)
        # x_t <- x_t + sqrt(sigma_t+1 ** 2 - sigma_t ** 2) * z
        pred_noise = self(x, t+1)
        # (n, d)

        noise = torch.randn_like(x)
        x_t_plus1 = x

        mean = c0 * (x_t_plus1 - c1 * pred_noise)
        std = sigma
        x = mean + std * noise

        step_log_probs = gaussian_log_pdf(x.detach(), mean, std)
        with torch.no_grad():
            # todo: use a pre-trained model
            ref_log_probs = gaussian_log_pdf(x, x_t_plus1, std)

        return x, step_log_probs, ref_log_probs

    @torch.no_grad()
    def sample(self, batch_size: int, device: torch.device = "cpu") -> Tensor:
        _training = self.training
        self.training = False

        num_timesteps = self.noise_scheduler.num_timesteps
        x = torch.rand(batch_size, self.spatial_dim, device=device)
        t = num_timesteps-1
        for step in range(num_timesteps-1):
            x = self.sample_and_log_prob_step(x, t)[0]
            t = t-1

        self.training = _training
        return x


def compute_loss(
    method, log_probs, kl_loss, rewards, beta, old_log_probs, ratio_clip=1e-1
):
    # https://github.com/XueZeyue/DanceGRPO/blob/d97950b51def6e61fddda83b0dbcbc615b07997c/fastvideo/train_grpo_flux.py#L597
    ratio = torch.exp(log_probs - old_log_probs.detach())
    ratio = torch.clamp(ratio, 1.0 - ratio_clip, 1. + ratio_clip)
    first_term = torch.mean(-rewards * ratio)

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
        grid_x = torch.linspace(-5, 5, steps=25)
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


def pretrain():
    # pretrain model with score matching
    if os.path.exists(pretrained_model_path):
        return

    lr = 1e-3

    model = DiffusionModel(
        num_timesteps=num_timesteps, hidden_dim=hidden_dim, spatial_dim=spatial_dim
    )
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
        dist1 = torch.distributions.Normal(torch.tensor([0.]), torch.tensor([1.]))
        dist2 = Normal(torch.tensor([5.5]), torch.tensor(0.5))

    dataset_size = 100
    data = torch.cat(
        [
            dist1.sample((math.ceil(dataset_size / 2),)),
            dist2.sample((math.floor(dataset_size / 2),))
        ], dim=0
    )
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
        plt.savefig(output_dir / f"untrained_model_samples.png")
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
        plt.savefig(output_dir / f"pretrain_loss.png")
        plt.close()

        samples = model.sample(100, device)
        if model.spatial_dim == 2:
            plt.plot(samples[:, 0], samples[:, 1], "bo", alpha=0.5, label="samples")
            plt.plot(data[:, 0], data[:, 1], "rx", alpha=0.2, label="data")
        elif model.spatial_dim == 1:
            plt.plot(samples[:, 0], torch.zeros(samples.shape[0]), "bo", alpha=0.5, label="samples")
            plt.plot(data[:, 0], .1 + torch.zeros(dataset_size), "rx", alpha=0.2, label="data")
        plt.legend(frameon=False)
        plt.savefig(output_dir / f"pretrained_model_samples.png")
        plt.close()

    # - Save model
    torch.save(model.state_dict(), pretrained_model_path)


def run(method: str, use_pretrained_model: bool = True):
    assert method in ["dpok", "grpo"]

    # hyperparameters
    lr = 5e-5
    num_iterations = 5_001
    batch_size = 64
    use_sparse_reward = False
    reward_temperature = 1.0
    use_ema = False
    old_policy_update_freq = 10 if use_ema else 100
    epsilon = 0.1
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
        "lr": lr, "num_iterations": num_iterations, "batch_size": batch_size,
        "use_sparse_reward": use_sparse_reward,
        "reward_temperature": reward_temperature,
        "use_ema": use_ema, "old_policy_update_freq": old_policy_update_freq,
        "epsilon": epsilon, "beta": beta, "num_inner_steps": num_inner_steps,
        "num_timesteps": num_timesteps, "hidden_dim": hidden_dim,
        "spatial_dim": spatial_dim, "use_pretrained_model": use_pretrained_model,
        "normalize_advantages": normalize_advantages,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f)

    if spatial_dim == 2:
        energy_fn: nn.Module = get_energy_function(
            device, sparse=use_sparse_reward, discrete_energy=False
        )
    elif spatial_dim == 1:
        dist = torch.distributions.Normal(0, 1)
        energy_fn: Callable = lambda x: -dist.log_prob(x)

    reward_fn: Callable = (
        lambda x: 10.0 * torch.exp(-energy_fn(x.detach()-0.5) / reward_temperature)
    )

    # -- Model
    model = DiffusionModel(
        num_timesteps=num_timesteps, hidden_dim=hidden_dim, spatial_dim=spatial_dim
    )
    model.to(device)
    model.train()
    if use_pretrained_model and os.path.exists(pretrained_model_path):
        print("Loaded pretrained model")
        model.load_state_dict(torch.load(pretrained_model_path, weights_only=True))

    old_policy = DiffusionModel(
        num_timesteps=num_timesteps, hidden_dim=hidden_dim, spatial_dim=spatial_dim
    )
    old_policy.load_state_dict(model.state_dict())
    old_policy.train()
    for p in old_policy.parameters():
        p.requires_grad = False
    old_policy.to(device)

    # -- Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -- Plot samples against reward
    plot_current_policy(model, reward_fn, device, method, 0)

    # -- Train
    meters = {"loss": AverageMeter(), "reward": AverageMeter()}
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
            # todo: try ema with higher clipping value or learning rate?
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()

            _loss = float(loss.detach().cpu())
            _rewards = rewards.detach().mean().cpu()
            meters["loss"].update(_loss)
            meters["reward"].update(_rewards)
            learning_curves["loss"].append(artifacts['first term'].detach())
            learning_curves["reward"].append(_rewards)
            learning_curves["grad_norm"].append(grad_norm)
        if iter % old_policy_update_freq == 0:
            update_old_policy(old_policy, model, use_ema=use_ema)
        if iter % log_freq == 0:
            metric_str = (
                f"Iter [{iter}/{num_iterations}]: "
                f"loss {meters['loss'].avg:0.4f}, "
                f"reward {meters['reward'].avg:0.4f}, "
                f"first term: {artifacts['first term']:0.4f}, "
                f"kl loss: {artifacts['kl_loss']:0.4f}, "
                f"grad norm: {grad_norm:0.3f}"
            )
            print(metric_str)
            for meter in meters.values():
                meter.reset()

    # plot samples against reward
    plot_current_policy(model, reward_fn, device, method, num_iterations)
    plot_learning_curves(learning_curves, method)


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    pretrain()
    run(method="dpok")

