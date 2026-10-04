from random import choice

from poke_env.teambuilder import Teambuilder

from src.constants import BATTLE_FORMATS
from src.utils.load_pokemon import choose_random_team_from_format


class RandomTeam(Teambuilder):
    """
    Random builds from data/showdown_sets, with a new team (no repeated species) for
    every battle.
    """

    def __init__(self, format: str = ""):
        self.format = format if format != "" else choice(BATTLE_FORMATS)

    def __repr__(self):
        return f"RandomTeam({self.format!r})"

    def yield_team(self) -> str:
        # Showdown expects teams in packed format
        team = choose_random_team_from_format(self.format)
        return self.join_team(self.parse_showdown_team(team))


def team_for_format(format: str) -> Teambuilder | None:
    """
    Random battle formats don't take a team, the server builds one for each battle.
    """
    return None if "random" in format else RandomTeam(format)
