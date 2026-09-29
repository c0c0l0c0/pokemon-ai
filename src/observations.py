from src.utils.showdown_parsing import parse_showdown_string

from .types import PokemonStats, Type


class PokemonObs:
    def __init__(self, showdown_string: str):
        pokemon_data = parse_showdown_string(showdown_string)

        self.name: str = pokemon_data["name"]
        self.types: tuple[Type, Type] = pokemon_data["types"]
        self.tera_type: Type = pokemon_data["tera_type"]
        self.moves: list[str] = pokemon_data["moves"]
        self.ability: str = pokemon_data["ability"]
        self.nature: str = pokemon_data["nature"]
        self.item: str = pokemon_data["item"]
        self.stats: PokemonStats = pokemon_data["stats"]
        self.evs: PokemonStats = pokemon_data["evs"]
