import math

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from adjoint_sampling.utils import (
    FourierTimeEmbeddings,
    GeometricNoiseScheduler,
)
from aliases import *


class DiffusionModel(nn.Module):
    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 1.0):
        super().__init__()
        time_emb_dim = 32
        self.hidden_dim = 64
        self.time_embedder = FourierTimeEmbeddings(time_emb_dim)
        self.layers = nn.Sequential(
            nn.Linear(2 + time_emb_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 2)
        )
        self.register_buffer("base_dist_loc", torch.zeros(2))
        self.noise_scheduler = GeometricNoiseScheduler(
            sigma_min=sigma_min, sigma_max=sigma_max
        )
        self.num_timesteps = 1000

    def grad_log_p_base(self, x: Tensor, t: Tensor, mu: Union[Tensor, float] = 0.0):
        std = self.noise_scheduler.get_sigma(t).view(-1, 1)
        return -(x - mu) / std.pow(2)

    def forward(self, x, time_emb):
        return self.layers(torch.cat([x, time_emb], dim=-1))

    def compute_loss(self, x, energy_grads=None, max_target_score_norm=None):
        """
        x: shape (batch_size, 2)
        energy_grads: shape (batch_size, 2)
        """
        device = x.device
        batch_size = x.shape[0]

        # - Sample noise levels
        t: Tensor = self.noise_scheduler.uniform_sample_timestep(batch_size, device)
        # (batch_size,)
        time_embs = self.time_embedder(t.float())
        # (batch_size, time_emb)
        sigmas = self.noise_scheduler.get_sigma(t)[:, None]
        # (batch_size, 1)
        nu_t_given_1 = self.noise_scheduler.get_nu_t_given_1(t)[:, None]
        # (batch_size, 1)
        alpha_t_given_1 = self.noise_scheduler.get_alpha_t_given_1(t)[:, None]
        # (batch_size, 1)

        with torch.no_grad():
            # Sample from p(x_t | x_1) and get grad_g
            noisy_x = alpha_t_given_1 * (x + nu_t_given_1 * torch.randn_like(x))
            if energy_grads is None:
                # Compute score of Gaussian
                grad_g = (x - noisy_x) / sigmas.pow(2)
            else:
                # Compute RAM loss grad_g
                grad_g = energy_grads + self.grad_log_p_base(x, t, mu=0.0)

            if max_target_score_norm is not None:
                # Clip scores for stability (e.g. prevent huge scores where
                # energy scales are large)
                score_norms = torch.sqrt(grad_g.pow(2).sum(dim=-1))
                clip_coefficient = torch.clamp(
                    max_target_score_norm / (score_norms + 1e-6), max=1.0
                )
                grad_g = grad_g * clip_coefficient[:, None]

        # - Predict scores
        predicted_scores = self(noisy_x, time_embs) / sigmas

        loss = torch.mean((predicted_scores + grad_g).pow(2).sum(dim=-1))
        return loss

    @torch.no_grad()
    def sample(
        self,
        num_samples: int = 1,
        device: torch.device = "cpu",
        sampling_type: str = "euler_maruyama"
    ):
        assert sampling_type in ["predictor", "euler_maruyama"]
        base_dist = Normal(self.base_dist_loc, self.noise_scheduler.base_dist_std)

        x0 = base_dist.sample((num_samples,))
        # (num_samples, 2)

        x_current = x0
        ts = torch.linspace(0, 1, steps=self.num_timesteps+1, device=device)
        dt = ts[1] - ts[0]
        for i in range(self.num_timesteps):
            t = ts[i]
            t_next = ts[i+1]
            time_emb = self.time_embedder(t.view(-1)).expand(num_samples, -1)
            # (num_samples, time_dim)

            sigma_current = self.noise_scheduler.get_sigma(t)  # (,)
            sigma_next = self.noise_scheduler.get_sigma(t_next)  # (,)

            if sampling_type == "predictor":
                x_current = self.predictor_step(x_current, time_emb, sigma_next, sigma_current)
            elif sampling_type == "euler_maruyama":
                x_current = self.euler_maruyama_step(t, x_current, time_emb, dt)

        return x_current

    def predictor_step(
        self,
        x_current: Tensor,
        time_emb: Tensor,
        sigma_next: Tensor,
        sigma_current: Tensor,
    ):
        # --- Predictor
        # x_t <- x_t+1 + (sigma_t+1 ** 2 - sigma_t ** 2) * score(x_t+1, sigma_t+1)
        # z ~ N(0, I)
        # x_t <- x_t + sqrt(sigma_t+1 ** 2 - sigma_t ** 2) * z

        # Get score(x_t+1, sigma_t+1)
        predicted_score = self(x_current, time_emb) * sigma_next
        sigma_sq_diff = sigma_next ** 2 - sigma_current ** 2
        x_current = x_current + sigma_sq_diff * predicted_score
        noise = torch.randn_like(x_current)
        x_current = x_current + torch.sqrt(sigma_sq_diff) * noise
        assert (~x_current.isnan()).any()

        # # --- Corrector: Algo 4 in https://arxiv.org/pdf/2011.13456
        # # z ~ N(0, I)
        # # x_t <- x_t + eps_t * score(x_t, sigma_t) + sqrt(2 * eps_t) * z
        #
        # # Get score(x_t, sigma_t)
        # predicted_score = sigma_current * self(x_current, time_emb)
        # noise = torch.randn_like(x_current)
        # noise_norm = (noise ** 2).sum(dim=-1).sqrt().mean()
        # grad_norm = (predicted_score ** 2).sum(dim=-1).sqrt().mean().clamp(min=1e-7)
        # step_size = 2 * (snr * noise_norm / grad_norm).pow(2)
        # step_size = step_size.clamp(max=1e6)
        # x_current = x_current + step_size * predicted_score + torch.sqrt(2 * step_size) * noise
        # assert (~x_current.isnan()).any()
        return x_current

    def euler_maruyama_step(self, t, x_current, time_emb, dt):
        sigma = self.noise_scheduler.get_sigma(t)
        drift = dt * sigma * self(x_current, time_emb)
        noise = (
            torch.sqrt(dt)
            * sigma
            * torch.randn_like(x_current)
        )
        return x_current + drift + noise
