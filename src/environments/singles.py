from typing import Any

import numpy as np
from gymnasium.spaces import Space
from poke_env.battle import AbstractBattle, Battle
from poke_env.environment import SingleAgentWrapper
from poke_env.environment.singles_env import SinglesEnv
from poke_env.player import Player, RandomPlayer

from src.observations.singles import SinglesObservation
from src.observations.vocab import Vocab


class SinglesBattleEnv(SinglesEnv):
    """
    Singles battle env. Action space and action mask are inherited from SinglesEnv:

        Discrete(6 + 4 * (n_gimmicks + 1))  -> 26 in gen9
        0-5   switch to team slot i
        6-9   move i
        10-13 move i + mega
        14-17 move i + z-move
        18-21 move i + dynamax
        22-25 move i + terastallize

    Observations are Dict({"observation": SinglesObservation.space, "action_mask": Box(n_actions)}).
    The action_mask key is added automatically by PokeEnv when observation_spaces is set.
    """

    def __init__(
        self, vocab: Vocab | None = None, shaping_weight: float = 1.0, **kwargs
    ):
        super().__init__(**kwargs)
        # Weight of the HP / fainted / status part of the reward, can be annealed to 0
        self.shaping_weight = shaping_weight
        self.observation_builder = SinglesObservation(vocab)
        # PokeEnv.__setattr__ wraps each raw space into Dict({observation, action_mask}),
        # so the declared (wrapped) type doesn't match what we assign here
        observation_spaces: dict[str, Space[Any]] = {
            agent: self.observation_builder.space for agent in self.possible_agents
        }
        self.observation_spaces = observation_spaces

    @classmethod
    def create_single_agent_env(
        cls,
        battle_format: str = "gen9randombattle",
        opponent: Player | None = None,
        **kwargs,
    ) -> SingleAgentWrapper:
        kwargs.setdefault("log_level", 40)
        kwargs.setdefault("open_timeout", None)
        kwargs.setdefault("strict", False)
        env = cls(battle_format=battle_format, **kwargs)
        if opponent is None:
            opponent = RandomPlayer(battle_format=battle_format, start_listening=False)
        return SingleAgentWrapper(env, opponent)

    def calc_reward(self, battle: AbstractBattle) -> float:
        """
        +1 for a win, -1 for a loss, plus the change in a hand-made state value (HP,
        fainted, status). The weights keep the old ratios, where a win was worth 30.
        """
        shaping = (
            self.reward_computing_helper(
                battle,
                fainted_value=2.0,
                hp_value=1.0,
                status_value=0.5,
                victory_value=0.0,
            )
            / 30
        )
        outcome = 1.0 if battle.won else -1.0 if battle.lost else 0.0
        return self.shaping_weight * shaping + outcome

    def embed_battle(self, battle: AbstractBattle) -> dict[str, np.ndarray]:
        assert isinstance(battle, Battle)
        return self.observation_builder.embed(battle)
