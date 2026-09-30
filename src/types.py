import re
from enum import Enum
from random import choice
from typing import ClassVar

from poke_env.data import GenData
from poke_env.data.normalize import to_id_str

from .constants import BATTLE_FORMATS


class Type(Enum):
    NONE = 0
    NORMAL = 1
    GRASS = 2
    FIRE = 3
    WATER = 4
    FIGHTING = 5
    PSYCHIC = 6
    DARK = 7
    GHOST = 8
    ROCK = 9
    FLYING = 10
    GROUND = 11
    ELECTRIC = 12
    POISON = 13
    ICE = 14
    BUG = 15
    DRAG0N = 16
    STEEL = 17
    FAIRY = 18

    @classmethod
    def from_pokemon(cls, pokemon_name: str, gen: int = 9) -> tuple["Type", "Type"]:
        """
        Returns the pokemon's two types. Single-type pokemon get Type.NONE as the second.
        """
        types = GenData.from_gen(gen).pokedex[to_id_str(pokemon_name)]["types"]
        first = cls[types[0].upper()]
        second = cls[types[1].upper()] if len(types) > 1 else cls.NONE
        return first, second


class PokemonStats:
    _SHOWDOWN_STAT_NAMES: ClassVar[dict[str, str]] = {
        "HP": "hp",
        "Atk": "attack",
        "Def": "defense",
        "SpA": "special_attack",
        "SpD": "special_defense",
        "Spe": "speed",
    }

    def __init__(
        self,
        hp: int = 0,
        attack: int = 0,
        defense: int = 0,
        special_attack: int = 0,
        special_defense: int = 0,
        speed: int = 0,
    ):
        self.hp = hp
        self.attack = attack
        self.defense = defense
        self.special_attack = special_attack
        self.special_defense = special_defense
        self.speed = speed

    @classmethod
    def from_string(cls, stats_string: str, default: int = 0) -> "PokemonStats":
        """
        Parses a showdown EVs/IVs line, e.g. "EVs: 252 Atk / 4 Def / 252 Spe".
        """
        stats = {attribute: default for attribute in cls._SHOWDOWN_STAT_NAMES.values()}

        values = stats_string.split(":")[-1]
        for part in values.split("/"):
            value, stat = part.split()
            stats[cls._SHOWDOWN_STAT_NAMES[stat]] = int(value)

        return cls(**stats)

    @classmethod
    def from_pokemon(cls, pokemon_name: str, gen: int = 9) -> "PokemonStats":
        """
        Returns the pokemon's base stats.
        """
        base_stats = GenData.from_gen(gen).pokedex[to_id_str(pokemon_name)]["baseStats"]
        return cls(
            **{
                attribute: base_stats[stat.lower()]
                for stat, attribute in cls._SHOWDOWN_STAT_NAMES.items()
            }
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "hp": self.hp,
            "attack": self.attack,
            "defense": self.defense,
            "special_attack": self.special_attack,
            "special_defense": self.special_defense,
            "speed": self.speed,
        }

    def to_list(self) -> list[int]:
        return [
            self.hp,
            self.attack,
            self.defense,
            self.special_attack,
            self.special_defense,
            self.speed,
        ]


class BattleFormat:
    def __init__(self, format: str = ""):
        self.name: str = format if format != "" else choice(BATTLE_FORMATS)
        self.doubles: bool = "doubles" in self.name or "vgc" in self.name

        match = re.search(r"\d", self.name)
        self.gen = int(match.group()) if match else 0

    def __repr__(self):
        return f"{self.name}"
