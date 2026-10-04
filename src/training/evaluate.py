"""
Evaluation against poke-env's scripted bots. Self-play win rates stay around 50% by
design, so fixed opponents are what show real progress.
"""

import asyncio

from src.models.singles import SinglesPolicy
from src.observations.singles import SinglesObservation
from src.players.policy_player import PolicyPlayer
from src.training.opponents import SCRIPTED_PLAYERS
from src.wrappers.teams import team_for_format


class Evaluator:
    def __init__(
        self,
        policy: SinglesPolicy,
        observation: SinglesObservation,
        battle_format: str,
        *,
        greedy: bool = False,
        concurrent_battles: int = 10,
    ):
        # Created once and reused, so each evaluation doesn't open new connections
        self.player = PolicyPlayer(
            policy,
            observation,
            greedy=greedy,
            battle_format=battle_format,
            team=team_for_format(battle_format),
            max_concurrent_battles=concurrent_battles,
            log_level=40,
        )
        self.opponents = {
            name: cls(
                battle_format=battle_format,
                team=team_for_format(battle_format),
                max_concurrent_battles=concurrent_battles,
                log_level=40,
            )
            for name, cls in SCRIPTED_PLAYERS.items()
        }

    def run(self, n_battles: int) -> dict[str, float]:
        """
        Win rate against each scripted bot over n_battles.
        """
        win_rates = {}
        for name, opponent in self.opponents.items():
            self.player.reset_battles()
            opponent.reset_battles()
            asyncio.run(self.player.battle_against(opponent, n_battles=n_battles))
            finished = max(self.player.n_finished_battles, 1)
            win_rates[f"eval_{name}"] = self.player.n_won_battles / finished
        return win_rates
