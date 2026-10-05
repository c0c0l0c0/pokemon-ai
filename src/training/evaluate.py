"""
Evaluation against poke-env's scripted bots. Self-play win rates stay around 50% by
design, so fixed opponents are what show real progress.
"""

import asyncio

from poke_env.ps_client import AccountConfiguration

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
        # Created once and reused, so each evaluation doesn't open new connections.
        # Random name suffixes, so they don't clash with other processes' players
        # (poke-env numbers names from 1 in every process).
        self.player = PolicyPlayer(
            policy,
            observation,
            greedy=greedy,
            account_configuration=AccountConfiguration.generate(
                "PolicyPlayer", rand=True
            ),
            battle_format=battle_format,
            team=team_for_format(battle_format),
            max_concurrent_battles=concurrent_battles,
            log_level=40,
        )
        self.opponents = {
            name: cls(
                account_configuration=AccountConfiguration.generate(
                    cls.__name__, rand=True
                ),
                battle_format=battle_format,
                team=team_for_format(battle_format),
                max_concurrent_battles=concurrent_battles,
                log_level=40,
            )
            for name, cls in SCRIPTED_PLAYERS.items()
        }

    def run(self, n_battles: int) -> dict[str, float]:
        """
        Win rate against each scripted bot over n_battles, by bot name.
        """
        win_rates = {}
        for name, opponent in self.opponents.items():
            self.player.reset_battles()
            opponent.reset_battles()
            asyncio.run(self.player.battle_against(opponent, n_battles=n_battles))
            finished = max(self.player.n_finished_battles, 1)
            win_rates[name] = self.player.n_won_battles / finished
        return win_rates
