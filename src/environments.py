from gymnasium.spaces import Box
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.environment import SingleAgentWrapper
from poke_env.environment.singles_env import SinglesEnv
from poke_env.player import SimpleHeuristicsPlayer

from .types import BattleFormat
from .observations import PokemonObs


class BattleEnv(SinglesEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_spaces = ...

    @classmethod
    def create_env(cls, format: str) -> SingleAgentWrapper:
        env = cls(battle_format=format, log_level=40, open_timeout=None)
        opponent = SimpleHeuristicsPlayer(start_listening=False)
        return SingleAgentWrapper(env, opponent)

    def calc_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle,
            fainted_value=2.0,
            hp_value=1.0,
            status_value=0.5,
            victory_value=30.0,
        )

    def embed_battle(self, battle: AbstractBattle):
