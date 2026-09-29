from random import choice

from .constants import BATTLE_FORMATS


class BattleFormat:
    def __init__(self, format: str = ""):
        self.name = format if format != "" else choice(BATTLE_FORMATS)
        self.doubles: bool = "doubles" in self.name

    def __repr__(self):
        return f"{self.name}"
