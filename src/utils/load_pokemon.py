from functools import cache
from random import choice, sample

from poke_env.data import GenData, to_id_str
from poke_env.teambuilder import Teambuilder

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


def choose_random_team_from_format(format: str, size: int = 6) -> str:
    """
    Chooses `size` random pokemon of different species (species clause) given the
    battle format.

    Returns the team's builds for showdown.
    """
    sets_by_species = _sets_by_species(format)
    if len(sets_by_species) < size:
        raise ValueError(f"{format} only has sets for {len(sets_by_species)} species")

    species = sample(list(sets_by_species), size)
    return "\n\n".join(choice(sets_by_species[s]) for s in species)


@cache
def _sets_by_species(format: str) -> dict[int | str, list[str]]:
    """
    Groups the format's builds by national dex number, since the species clause counts
    formes (e.g. Rotom-Wash and Rotom-Heat) as the same species.
    """
    pokedex = GenData.from_format(format).pokedex
    with open(f"data/showdown_sets/{format}.txt") as f:
        builds = [build.strip() for build in f.read().split("\n\n") if build.strip()]

    sets_by_species: dict[int | str, list[str]] = {}
    for build in builds:
        parsed = Teambuilder.parse_showdown_team(build)
        if not parsed:
            continue
        species = to_id_str(parsed[0].species or parsed[0].nickname or "")
        number = pokedex.get(species, {}).get("num", species)
        sets_by_species.setdefault(number, []).append(build)
    return sets_by_species
