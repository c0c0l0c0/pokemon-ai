from typing import Any

from src.types import PokemonStats, Type


def parse_showdown_string(showdown_string: str) -> dict[str, Any]:
    """
    Given a showdown string, returns the necessary data to create a PokemonObs
    """
    lines = showdown_string.split("\n")

    name: str = ""
    types: tuple[Type, Type] = (Type.NONE, Type.NONE)
    tera_type: Type = Type.NONE
    moves: list[str] = []
    ability: str = ""
    nature: str = ""
    item: str = ""
    stats: PokemonStats = PokemonStats()
    evs: PokemonStats = PokemonStats()

    for i, line in enumerate(lines):
        if i == 0:
            name = line
            types = Type.from_pokemon(name)
            stats = PokemonStats.from_pokemon(name)
            if "@" in line:
                item = line.split("@")[1][1:]

        if "Ability" in line:
            ability = line[9:]

        if "Nature" in line:
            nature = line.split(" ")[0]

        if "EVs" in line:
            evs = PokemonStats.from_string(line)

        if "Tera Type" in line:
            tera_type = Type[line[11:].upper()]
        if line.startswith("-"):
            moves.append(line[2:])

    return {
        "name": name,
        "types": types,
        "tera_type": tera_type,
        "moves": moves,
        "ability": ability,
        "nature": nature,
        "item": item,
        "stats": stats,
        "evs": evs,
    }
