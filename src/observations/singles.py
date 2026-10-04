"""
Observation for singles battles.

One row per Pokémon (6 mine, then 6 opponent) with 4 move slots each, plus a global
field vector. Species, items, abilities and moves are int ids for nn.Embedding,
everything else is numeric in [-1, 1].

Row order matches SinglesEnv's action indices:
- rows 0-5 are list(battle.team.values()), the Pokémon of switch actions 0-5
- move slots of my rows are list(pokemon.moves.values())[:4], move actions 6-9
"""

from collections.abc import Iterable
from enum import Enum
from math import log2

import numpy as np
from gymnasium import spaces
from poke_env.battle import (
    STACKABLE_CONDITIONS,
    Battle,
    Effect,
    Field,
    Move,
    MoveCategory,
    Pokemon,
    PokemonType,
    SideCondition,
    Status,
    Weather,
)
from poke_env.data import GenData
from poke_env.stats import compute_raw_stats

from src.observations.vocab import PAD, UNKNOWN, Vocab

N_TEAM = 6
N_POKEMON = 2 * N_TEAM
N_MOVES = 4

STATS = ("hp", "atk", "def", "spa", "spd", "spe")
BOOSTS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")

# Opponent stats are unknown, so they're estimated with a random battles style spread
ESTIMATED_EVS = [84] * 6
ESTIMATED_IVS = [31] * 6
ESTIMATED_NATURE = "serious"

# Columns of pokemon_ids
SPECIES, ITEM, ABILITY, LAST_MOVE = range(4)

# First columns of pokemon_feats, used by the model to find the active Pokémon
IS_MINE, REVEALED, ACTIVE, FAINTED, HP = range(5)

POKEMON_FEATS = (
    5  # is mine, revealed, active, fainted, hp fraction
    + len(Status)
    + 1  # status counter
    + len(BOOSTS)
    + 2 * len(STATS)  # base stats, stats
    + 1  # stats estimated
    + 2  # level, weight
    + 2 * len(PokemonType)  # current types, tera type
    + 1  # terastallized
    + 4  # first turn, must recharge, preparing, protect counter
    + len(Effect)
    + 2  # type matchup vs the opposing active: offense, defense
)
MOVE_FEATS = (
    2  # known, available now
    + 6  # base power, accuracy, priority, pp, crit ratio, expected hits
    + len(MoveCategory)
    + len(PokemonType)
    + 15  # move effects
    + 3  # vs the opposing active: effectiveness, immune, STAB
)
FIELD_FEATS = (
    len(Weather)
    + 1  # weather turns
    + 2 * len(Field)  # presence, turns
    + 2 * 2 * len(SideCondition)  # mine and theirs: presence, turns
    + 6  # turn, remaining (mine, theirs), force switch, trapped, maybe trapped
    + 12  # gimmicks: can / used / opponent used
    + 2  # dynamax turns left (mine, theirs)
)


def _index(enum: type[Enum]) -> dict:
    return {member: i for i, member in enumerate(enum)}


STATUS_INDEX = _index(Status)
TYPE_INDEX = _index(PokemonType)
EFFECT_INDEX = _index(Effect)
CATEGORY_INDEX = _index(MoveCategory)
WEATHER_INDEX = _index(Weather)
FIELD_INDEX = _index(Field)
SIDE_CONDITION_INDEX = _index(SideCondition)


class SinglesObservation:
    """
    Builds the observation dict of a singles Battle:

        pokemon_ids    int64   (12, 4)       species, item, ability, last move
        pokemon_feats  float32 (12, P)       hp, status, boosts, stats, types, effects...
        move_ids       int64   (12, 4)       move per slot
        move_feats     float32 (12, 4, M)    power, accuracy, type, effectiveness...
        field_feats    float32 (F,)          weather, terrain, hazards, screens, gimmicks...

    Unrevealed values get the UNKNOWN id, and empty slots the PAD id.
    """

    def __init__(self, vocab: Vocab | None = None):
        self.vocab = vocab or Vocab.load()

    @property
    def space(self) -> spaces.Dict:
        max_ids = [
            self.vocab.n_species - 1,
            self.vocab.n_items - 1,
            self.vocab.n_abilities - 1,
            self.vocab.n_moves - 1,
        ]
        return spaces.Dict(
            {
                "pokemon_ids": spaces.Box(
                    low=0,
                    high=np.tile(max_ids, (N_POKEMON, 1)),
                    shape=(N_POKEMON, 4),
                    dtype=np.int64,
                ),
                "pokemon_feats": spaces.Box(
                    -1.0, 1.0, shape=(N_POKEMON, POKEMON_FEATS), dtype=np.float32
                ),
                "move_ids": spaces.Box(
                    0,
                    self.vocab.n_moves - 1,
                    shape=(N_POKEMON, N_MOVES),
                    dtype=np.int64,
                ),
                "move_feats": spaces.Box(
                    -1.0, 1.0, shape=(N_POKEMON, N_MOVES, MOVE_FEATS), dtype=np.float32
                ),
                "field_feats": spaces.Box(
                    -1.0, 1.0, shape=(FIELD_FEATS,), dtype=np.float32
                ),
            }
        )

    def embed(self, battle: Battle) -> dict[str, np.ndarray]:
        pokemon_ids = np.full((N_POKEMON, 4), PAD, dtype=np.int64)
        pokemon_feats = np.zeros((N_POKEMON, POKEMON_FEATS), dtype=np.float32)
        move_ids = np.full((N_POKEMON, N_MOVES), PAD, dtype=np.int64)
        move_feats = np.zeros((N_POKEMON, N_MOVES, MOVE_FEATS), dtype=np.float32)

        my_active = battle.active_pokemon
        opponent_active = battle.opponent_active_pokemon
        available = {move.id for move in battle.available_moves}
        tera_types = _my_tera_types(battle)

        for row, (identifier, mon) in enumerate(list(battle.team.items())[:N_TEAM]):
            pokemon_ids[row] = self._pokemon_ids(mon)
            pokemon_feats[row] = _pokemon_features(
                mon,
                mine=True,
                opponent=opponent_active,
                tera_type=tera_types.get(identifier, mon.tera_type),
            )
            for slot, move in enumerate(list(mon.moves.values())[:N_MOVES]):
                move_ids[row, slot] = self.vocab.move_id(move.id)
                move_feats[row, slot] = _move_features(
                    move,
                    user=mon,
                    target=opponent_active,
                    available=mon is my_active and move.id in available,
                )

        opponents = _opponent_pokemon(battle)
        for i in range(N_TEAM):
            row = N_TEAM + i
            # Opponent slots always hold a Pokémon, unrevealed ones are UNKNOWN
            move_ids[row] = UNKNOWN
            if i >= len(opponents):
                pokemon_ids[row] = [UNKNOWN, UNKNOWN, UNKNOWN, PAD]
                pokemon_feats[row, HP] = 1.0
                continue

            mon = opponents[i]
            pokemon_ids[row] = self._pokemon_ids(mon)
            pokemon_feats[row] = _pokemon_features(
                mon, mine=False, opponent=my_active, tera_type=mon.tera_type
            )
            for slot, move in enumerate(list(mon.moves.values())[:N_MOVES]):
                move_ids[row, slot] = self.vocab.move_id(move.id)
                move_feats[row, slot] = _move_features(
                    move, user=mon, target=my_active, available=False
                )

        return {
            "pokemon_ids": pokemon_ids,
            "pokemon_feats": pokemon_feats,
            "move_ids": move_ids,
            "move_feats": move_feats,
            "field_feats": _field_features(battle, opponents),
        }

    def _pokemon_ids(self, mon: Pokemon) -> list[int]:
        last_move = mon.last_move
        return [
            self.vocab.species_id(mon.species),
            self.vocab.item_id(mon.item),
            self.vocab.ability_id(mon.ability),
            self.vocab.move_id(last_move.id) if last_move is not None else PAD,
        ]


def _pokemon_features(
    mon: Pokemon,
    *,
    mine: bool,
    opponent: Pokemon | None,
    tera_type: PokemonType | None,
) -> np.ndarray:
    stats, estimated = _stats(mon, mine)
    # Opponents seen only in team preview haven't taken damage yet
    hp = mon.current_hp_fraction if mine or mon.revealed else 1.0

    features = [
        float(mine),
        float(mon.revealed),
        float(bool(mon.active)),
        float(mon.fainted),
        hp,
        *_one_hot(STATUS_INDEX, [mon.status]),
        min(mon.status_counter, 8) / 8,
        *(mon.boosts[boost] / 6 for boost in BOOSTS),
        *(mon.base_stats[stat] / 255 for stat in STATS),
        *(min(stat / 1000, 1.0) for stat in stats),
        float(estimated),
        mon.level / 100,
        min(mon.weight / 1000, 1.0),
        *_one_hot(TYPE_INDEX, mon.types),
        *_one_hot(TYPE_INDEX, [tera_type]),
        float(mon.is_terastallized),
        float(mon.first_turn),
        float(mon.must_recharge),
        float(mon.preparing),
        min(mon.protect_counter, 3) / 3,
        *_one_hot(EFFECT_INDEX, mon.effects),
        *_type_matchup(mon, opponent),
    ]
    return np.asarray(features, dtype=np.float32)


def _move_features(
    move: Move, *, user: Pokemon, target: Pokemon | None, available: bool
) -> np.ndarray:
    multiplier = target.damage_multiplier(move) if target is not None else 1.0
    max_pp = move.max_pp

    features = [
        1.0,  # known
        float(available),
        min(move.base_power / 250, 1.0),
        move.accuracy,
        move.priority / 7,
        move.current_pp / max_pp if max_pp else 0.0,
        min(move.crit_ratio, 6) / 6,
        min(move.expected_hits, 10) / 10,
        *_one_hot(CATEGORY_INDEX, [move.category]),
        *_one_hot(TYPE_INDEX, [move.type]),
        move.drain,
        move.recoil,
        move.heal,
        float(bool(move.self_switch)),
        float(move.force_switch),
        float(move.is_protect_move),
        float(move.breaks_protect),
        float(bool(move.secondary)),
        float(move.status is not None),
        float(move.volatile_status is not None),
        float(move.side_condition is not None),
        float(move.weather is not None),
        float(move.terrain is not None),
        _boost_sum(move.self_boost),
        _boost_sum(move.boosts),
        _log_multiplier(multiplier) if target is not None else 0.0,
        float(multiplier == 0),
        float(move.type in user.types),
    ]
    return np.asarray(features, dtype=np.float32)


def _field_features(battle: Battle, opponents: list[Pokemon]) -> np.ndarray:
    my_fainted = sum(mon.fainted for mon in battle.team.values())
    opponent_fainted = sum(mon.fainted for mon in opponents)

    features = [
        *_one_hot(WEATHER_INDEX, battle.weather),
        max(
            (_turns_since(battle, start) for start in battle.weather.values()),
            default=0.0,
        ),
        *_one_hot(FIELD_INDEX, battle.fields),
        *_turns_active(battle, FIELD_INDEX, battle.fields),
        *_side_condition_features(battle, battle.side_conditions),
        *_side_condition_features(battle, battle.opponent_side_conditions),
        min(battle.turn / 50, 1.0),
        (N_TEAM - my_fainted) / N_TEAM,
        (N_TEAM - opponent_fainted) / N_TEAM,
        float(battle.force_switch),
        float(battle.trapped),
        float(battle.maybe_trapped),
        float(battle.can_tera),
        float(battle.can_mega_evolve),
        float(battle.can_z_move),
        float(battle.can_dynamax),
        float(battle.used_tera),
        float(battle.used_mega_evolve),
        float(battle.used_z_move),
        float(battle.used_dynamax),
        float(battle.opponent_used_tera),
        float(battle.opponent_used_mega_evolve),
        float(battle.opponent_used_z_move),
        float(battle.opponent_used_dynamax),
        (battle.dynamax_turns_left or 0) / 3,
        (battle.opponent_dynamax_turns_left or 0) / 3,
    ]
    return np.asarray(features, dtype=np.float32)


def _opponent_pokemon(battle: Battle) -> list[Pokemon]:
    """
    Revealed opponents first, then the ones only seen in team preview.
    """
    opponents = list(battle.opponent_team.values())
    seen = {mon.base_species for mon in opponents}
    for mon in battle.teampreview_opponent_team:
        if mon.base_species not in seen:
            opponents.append(mon)
            seen.add(mon.base_species)
    return opponents[:N_TEAM]


def _my_tera_types(battle: Battle) -> dict[str, PokemonType]:
    """
    poke-env doesn't read the tera types of my team from the request, so read them here.
    """
    side = battle.last_request.get("side", {}).get("pokemon", [])
    return {
        mon["ident"]: PokemonType.from_name(mon["teraType"])
        for mon in side
        if mon.get("teraType")
    }


def _stats(mon: Pokemon, mine: bool) -> tuple[list[int], bool]:
    """
    Real stats for my team, estimated ones for the opponent's.
    """
    if mine:
        stats = dict(mon.stats)
        if stats.get("hp") is None:
            stats["hp"] = mon.max_hp
        if all(stats.get(stat) is not None for stat in STATS):
            return [stats[stat] or 0 for stat in STATS], False

    try:
        estimated = compute_raw_stats(
            mon.species,
            ESTIMATED_EVS,
            ESTIMATED_IVS,
            mon.level,
            ESTIMATED_NATURE,
            GenData.from_gen(mon.gen),
        )
    except KeyError:
        estimated = [0] * len(STATS)
    return estimated, True


def _type_matchup(mon: Pokemon, opponent: Pokemon | None) -> tuple[float, float]:
    """
    Best multiplier mon's types deal to the opponent, and worst it takes from theirs.
    """
    if opponent is None:
        return 0.0, 0.0
    offense = max(opponent.damage_multiplier(t) for t in mon.types)
    defense = max(mon.damage_multiplier(t) for t in opponent.types)
    return _log_multiplier(offense), _log_multiplier(defense)


def _side_condition_features(
    battle: Battle, conditions: dict[SideCondition, int]
) -> list[float]:
    """
    Presence (layers / max layers for Spikes and Toxic Spikes) and turns since set.
    """
    presence = [0.0] * len(SideCondition)
    turns = [0.0] * len(SideCondition)
    for condition, value in conditions.items():
        i = SIDE_CONDITION_INDEX[condition]
        if condition in STACKABLE_CONDITIONS:
            presence[i] = value / STACKABLE_CONDITIONS[condition]
        else:
            presence[i] = 1.0
            turns[i] = _turns_since(battle, value)
    return presence + turns


def _one_hot(index: dict, values: Iterable) -> list[float]:
    """
    One-hot (or multi-hot) vector over an enum, ignoring None values.
    """
    vector = [0.0] * len(index)
    for value in values:
        if value is not None:
            vector[index[value]] = 1.0
    return vector


def _turns_active(battle: Battle, index: dict, starts: dict) -> list[float]:
    vector = [0.0] * len(index)
    for member, start in starts.items():
        vector[index[member]] = _turns_since(battle, start)
    return vector


def _turns_since(battle: Battle, start: int, horizon: int = 8) -> float:
    return min(max(battle.turn - start, 0), horizon) / horizon


def _log_multiplier(multiplier: float) -> float:
    """
    0.25x -> -1, 1x -> 0, 4x -> 1. Immunities are also -1.
    """
    if multiplier <= 0:
        return -1.0
    return float(np.clip(log2(multiplier) / 2, -1.0, 1.0))


def _boost_sum(boosts: dict[str, int] | None) -> float:
    if not boosts:
        return 0.0
    return float(np.clip(sum(boosts.values()) / 6, -1.0, 1.0))
