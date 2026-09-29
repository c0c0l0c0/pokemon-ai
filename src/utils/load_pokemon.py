from random import choice

from src.constants import BATTLE_FORMATS


def choose_random_pokemon_from_format(format: str) -> str:
    """
    Chooses a random pokemon given the battle format.

    Returns the pokemon's build for showdown.
    """
    with open(f"data/showdown_sets/{format}.txt") as f:
        pokemon_string = f.read()

    pokemon_list = pokemon_string.split("\n\n")

    return choice(pokemon_list)


def choose_random_pokemon() -> str:
    """
    Chooses a random pokemon from a random battle format.

    Returns the pokemon's build for showdown.
    """
    format = choice(BATTLE_FORMATS)

    return choose_random_pokemon_from_format(format)
