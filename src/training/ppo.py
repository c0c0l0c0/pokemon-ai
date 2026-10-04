"""
PPO (Proximal Policy Optimization) for SinglesPolicy.

Each seat's decisions in a battle form a trajectory. Advantages come from GAE: how much
better each action turned out than the value head expected. The update makes actions
with positive advantage more likely, but clips the change so one batch can't move the
policy too far.
"""

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from src.models.singles import SinglesPolicy


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatch_size: int = 512
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    # Stops the epochs early if the policy moved more than this (0 to never stop)
    target_kl: float = 0.05


@dataclass
class Trajectory:
    """
    The decisions of one seat in one battle. rewards[t] is everything the seat got
    between decision t and the next one (or the end of the battle).
    """

    observations: list[dict[str, np.ndarray]] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.actions)

    def add(
        self,
        observation: dict[str, np.ndarray],
        mask: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
    ):
        self.observations.append(observation)
        self.masks.append(mask)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(0.0)

    def add_reward(self, reward: float):
        # Rewards before the first decision have no action to be credited to
        if self.rewards:
            self.rewards[-1] += reward


@dataclass
class Batch:
    observations: dict[str, np.ndarray]
    masks: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray

    def __len__(self) -> int:
        return len(self.actions)


def compute_gae(
    rewards: np.ndarray, values: np.ndarray, gamma: float, lam: float
) -> np.ndarray:
    """
    Generalized advantage estimation for a finished trajectory (no value after the end).
    """
    advantages = np.zeros(len(rewards), dtype=np.float32)
    next_value, next_advantage = 0.0, 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        next_advantage = delta + gamma * lam * next_advantage
        advantages[t] = next_advantage
        next_value = values[t]
    return advantages


def make_batch(trajectories: list[Trajectory], gamma: float, lam: float) -> Batch:
    trajectories = [trajectory for trajectory in trajectories if len(trajectory)]
    advantages = np.concatenate(
        [
            compute_gae(np.array(t.rewards), np.array(t.values), gamma, lam)
            for t in trajectories
        ]
    )
    values = np.array([v for t in trajectories for v in t.values], dtype=np.float32)
    observations = [obs for t in trajectories for obs in t.observations]

    return Batch(
        observations={
            key: np.stack([obs[key] for obs in observations]) for key in observations[0]
        },
        masks=np.stack([mask for t in trajectories for mask in t.masks]),
        actions=np.array([a for t in trajectories for a in t.actions], dtype=np.int64),
        log_probs=np.array(
            [p for t in trajectories for p in t.log_probs], dtype=np.float32
        ),
        values=values,
        advantages=advantages,
        returns=advantages + values,
    )


def ppo_update(
    policy: SinglesPolicy,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    cfg: PPOConfig,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """
    A few epochs of minibatch updates on the batch. Returns the mean losses and
    diagnostics.
    """
    observations = {
        key: torch.as_tensor(value, device=device)
        for key, value in batch.observations.items()
    }
    masks = torch.as_tensor(batch.masks, device=device)
    actions = torch.as_tensor(batch.actions, device=device)
    old_log_probs = torch.as_tensor(batch.log_probs, device=device)
    returns = torch.as_tensor(batch.returns, device=device)
    advantages = torch.as_tensor(
        (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8),
        device=device,
    )

    stats: dict[str, list[float]] = defaultdict(list)
    for _ in range(cfg.epochs):
        epoch_kl = []
        for idx in torch.randperm(len(batch), device=device).split(cfg.minibatch_size):
            logits, values = policy(
                {key: value[idx] for key, value in observations.items()}, masks[idx]
            )
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions[idx])
            log_ratio = log_probs - old_log_probs[idx]
            ratio = log_ratio.exp()

            clipped_ratio = ratio.clamp(1 - cfg.clip, 1 + cfg.clip)
            policy_loss = -torch.min(
                ratio * advantages[idx], clipped_ratio * advantages[idx]
            ).mean()
            value_loss = 0.5 * (values - returns[idx]).pow(2).mean()
            entropy = dist.entropy().mean()
            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                # Low variance KL estimate, see http://joschu.net/blog/kl-approx.html
                approx_kl = ((ratio - 1) - log_ratio).mean().item()
                clip_fraction = ((ratio - 1).abs() > cfg.clip).float().mean().item()
            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy"].append(entropy.item())
            stats["approx_kl"].append(approx_kl)
            stats["clip_fraction"].append(clip_fraction)
            epoch_kl.append(approx_kl)

        if cfg.target_kl and np.mean(epoch_kl) > cfg.target_kl:
            break

    result = {key: float(np.mean(values)) for key, values in stats.items()}
    # How much of the return variance the value head explains (1 is perfect)
    result["explained_variance"] = float(
        1 - np.var(batch.returns - batch.values) / (np.var(batch.returns) + 1e-8)
    )
    return result
