from random import choice

from poke_env.teambuilder import Teambuilder

from src.constants import BATTLE_FORMATS
from src.utils.load_pokemon import (
    choose_random_pokemon_from_format,
)


class RandomTeam(Teambuilder):
    def __init__(self, format: str = ""):
        self.format = format if format != "" else choice(BATTLE_FORMATS)
        self.team = [choose_random_pokemon_from_format(self.format) for _ in range(6)]

    def __repr__(self):
        return "\n\n".join(self.team)

    def yield_team(self) -> str:
        return "\n\n".join(self.team)
