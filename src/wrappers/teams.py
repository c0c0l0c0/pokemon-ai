from poke_env.teambuilder import Teambuilder

from src.types import BattleFormat
from src.utils.load_pokemon import (
    choose_random_pokemon_from_format,
)


class RandomTeam(Teambuilder):
    def __init__(self, format: str = ""):
        self.format = BattleFormat(format)
        self.team = [
            choose_random_pokemon_from_format(self.format.name) for _ in range(6)
        ]

    def __repr__(self):
        return "\n\n".join(self.team)

    def yield_team(self) -> str:
        return "\n\n".join(self.team)
