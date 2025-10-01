import torch
import torch.nn as nn

from aliases import *


def update_old_policy(
    old_policy: nn.Module, new_policy: nn.Module, use_ema: bool = True, decay: float = 0.99
) -> None:
    if use_ema:
        with torch.no_grad():
            for p_old, p_new in zip(old_policy.parameters(), new_policy.parameters()):
                p_old.data.mul_(decay).add_((1.-decay) * p_new.data)
    else:
        old_policy.load_state_dict(new_policy.state_dict())


def compute_grpo_loss(
    samples,
    n_tokens_per_sample: Tensor,
    advantages: Tensor,
    policy: nn.Module,
    old_policy: nn.Module,
    reference_policy: nn.Module,
    epsilon: float = 0.1,
    beta: float = 0.1,
) -> Tensor:
    """
    Args:
        samples:
            batch of samples
        n_tokens_per_sample: torch.long
            shape (batch_size,)
        advantages: torch.float
            shape (batch_size,)
        policy:
        old_policy:
        reference_policy:
        epsilon: float
            Ex. If epsilon=0.2, then only update policy if its token probability
            differs from old_policy by at most 0.2. Ensures that policy doesn't
            change too much from old_policy.
        beta: float
            regularization strength for KL divergence between policy and
            reference_policy.
    """
    assert epsilon >= 0.0 and beta >= 0.0

    policy_probs = policy(samples)
    with torch.no_grad():
        old_policy_probs = old_policy(samples)
        reference_policy_probs = reference_policy(samples)

    token_probs_ratio = policy_probs / old_policy_probs
    # (batch_size, n_tokens)
    reward_loss = -1.0 * (
        token_probs_ratio.clamp(min=1.-epsilon, max=1.+epsilon)
        * advantages[:, None]
    )
    # (batch_size, n_tokens)

    ref_token_probs_ratio = reference_policy_probs / policy_probs
    kl_loss = (
        ref_token_probs_ratio
        - torch.log(ref_token_probs_ratio)
        - 1.0
    )
    return torch.sum(reward_loss + beta * kl_loss) / n_tokens_per_sample.sum()
