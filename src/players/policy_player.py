"""
A poke-env Player driven by a SinglesPolicy, to evaluate it against other players or
to play against it on a local Showdown server.
"""

import numpy as np
import torch
from poke_env.battle import AbstractBattle, Battle
from poke_env.environment import SinglesEnv
from poke_env.player import BattleOrder, Player
from torch.distributions import Categorical

from src.models.singles import SinglesPolicy, to_tensors
from src.observations.singles import SinglesObservation


class PolicyPlayer(Player):
    def __init__(
        self,
        policy: SinglesPolicy,
        observation: SinglesObservation | None = None,
        greedy: bool = False,
        **kwargs,
    ):
        """
        Samples each move from the policy, or takes the most likely one if greedy.
        The other kwargs go to poke-env's Player (battle_format, team...).
        """
        super().__init__(**kwargs)
        self.policy = policy
        self.observation = observation or SinglesObservation()
        self.greedy = greedy

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        assert isinstance(battle, Battle)
        device = next(self.policy.parameters()).device
        obs = to_tensors([self.observation.embed(battle)], device)
        mask = torch.as_tensor(
            np.array([SinglesEnv.get_action_mask(battle)]), device=device
        )
        with torch.no_grad():
            logits, _ = self.policy(obs, mask)

        if self.greedy:
            action = logits.argmax(dim=-1)
        else:
            action = Categorical(logits=logits).sample()
        return SinglesEnv.action_to_order(np.int64(action.item()), battle, strict=False)
