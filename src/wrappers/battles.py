import numpy as np
from gymnasium.spaces import Box
from poke_env.battle import AbstractBattle
from poke_env.environment.singles_env import SinglesEnv

from src.types import BattleFormat
from src.wrappers.teams import RandomTeam


class SingleBattle(SinglesEnv):
    def __init__(self, format: str = "", **kwargs):
        self.format = BattleFormat(format)
        if self.format.doubles:
            raise ValueError(f"{self.format.name} is not a singles format")

        super().__init__(
            battle_format=self.format.name,
            team=RandomTeam(self.format.name),
            **kwargs,
        )

        low = np.array([-1] * 4 + [0] * 4 + [0, 0], dtype=np.float32)
        high = np.array([3] * 4 + [4] * 4 + [1, 1], dtype=np.float32)
        self.observation_spaces = {  # type: ignore[assignment]
            agent: Box(low, high, dtype=np.float32) for agent in self.possible_agents
        }

    def calc_reward(self, battle: AbstractBattle) -> float:
        return self.reward_computing_helper(
            battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        # Placeholder features: move base power/multipliers, type effectiveness,
        # and fraction of each side's team fainted. Replace with your own.
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves[:4]):
            moves_base_power[i] = move.base_power / 100
            if battle.opponent_active_pokemon is not None:
                moves_dmg_multiplier[i] = (
                    battle.opponent_active_pokemon.damage_multiplier(move)
                )

        fainted_team = sum(p.fainted for p in battle.team.values()) / 6
        fainted_opp = sum(p.fainted for p in battle.opponent_team.values()) / 6

        return np.concatenate(
            [moves_base_power, moves_dmg_multiplier, [fainted_team, fainted_opp]]
        ).astype(np.float32)
