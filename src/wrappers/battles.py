from logging import getLogger

from poke_env.battle import Battle

from src.types import BattleFormat


class SingleBattle(Battle):
    def __init__(self, format: BattleFormat, tag: str = ""):
        if tag == "":
            tag = f"{format}-battle"
        super().__init__(
            battle_tag=tag, username="agent", logger=getLogger("battle"), gen=format.gen
        )

        self.battle_format = format
