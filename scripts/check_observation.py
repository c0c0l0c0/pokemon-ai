"""
Offline check of the singles observation and model, no Showdown server needed.

Builds a Battle from a fake request (my team) and protocol messages (the opponent's),
checks the observation against its space, and runs the model on it.

Run from the project root:
    .venv/bin/python -m scripts.check_observation
"""

import logging

import numpy as np
import torch
from gymnasium import spaces
from poke_env.battle import Battle, Pokemon
from poke_env.data import GenData
from poke_env.environment import SinglesEnv
from poke_env.stats import compute_raw_stats
from poke_env.teambuilder import Teambuilder, TeambuilderPokemon

from src.environments.singles import SinglesBattleEnv
from src.models.singles import SinglesPolicy, to_tensors
from src.observations.singles import (
    ACTIVE,
    HP,
    IS_MINE,
    LAST_MOVE,
    N_TEAM,
    REVEALED,
    SPECIES,
    SinglesObservation,
)
from src.observations.vocab import PAD, UNKNOWN, Vocab
from src.utils.load_pokemon import choose_random_pokemon_from_format

FORMAT = "gen9ou"
GEN = 9


def random_team(format: str) -> list[TeambuilderPokemon]:
    team: dict[str, TeambuilderPokemon] = {}
    while len(team) < N_TEAM:
        sets = choose_random_pokemon_from_format(format)
        mon = Teambuilder.parse_showdown_team(sets)[0]
        team.setdefault(name(mon), mon)
    return list(team.values())


def name(mon: TeambuilderPokemon) -> str:
    # species is only set when the set has a nickname
    return mon.species or mon.nickname or ""


def request_for(team: list[TeambuilderPokemon]) -> dict:
    """
    A Showdown request for my side, with the first Pokémon active.
    """
    side = []
    for i, tb in enumerate(team):
        mon = Pokemon(gen=GEN, teambuilder=tb)
        hp, atk, def_, spa, spd, spe = compute_raw_stats(
            mon.species,
            tb.evs or [0] * 6,
            tb.ivs or [31] * 6,
            100,
            (tb.nature or "serious").lower(),
            GenData.from_gen(GEN),
        )
        side.append(
            {
                "ident": f"p1: {name(tb)}",
                "details": name(tb),
                "condition": f"{hp}/{hp}",
                "active": i == 0,
                "stats": {"atk": atk, "def": def_, "spa": spa, "spd": spd, "spe": spe},
                "moves": list(mon.moves),
                "baseAbility": mon.ability,
                "ability": mon.ability,
                "item": mon.item or "",
                "teraType": tb.tera_type,
            }
        )
    active_moves = [
        {
            "move": move,
            "id": move,
            "pp": 16,
            "maxpp": 16,
            "target": "normal",
            "disabled": False,
        }
        for move in side[0]["moves"]
    ]
    return {
        "active": [{"moves": active_moves, "canTerastallize": side[0]["teraType"]}],
        "side": {"name": "me", "id": "p1", "pokemon": side},
        "rqid": 1,
    }


def build_battle(team_preview: bool) -> Battle:
    battle = Battle(f"battle-{FORMAT}-1", "me", logging.getLogger("check"), gen=GEN)
    battle.player_role = "p1"

    my_team = random_team(FORMAT)
    opponent_team = random_team(FORMAT)
    me, opponent = name(my_team[0]), name(opponent_team[0])

    if team_preview:
        for mon in opponent_team:
            battle.parse_message(["", "poke", "p2", f"{name(mon)}, L100", ""])
    battle.parse_request(request_for(my_team))

    opponent_move = opponent_team[0].moves[0]
    for message in [
        ["", "switch", f"p1a: {me}", me, "100/100"],
        ["", "switch", f"p2a: {opponent}", f"{opponent}, L100", "100/100"],
        ["", "move", f"p2a: {opponent}", opponent_move, f"p1a: {me}"],
        ["", "-damage", f"p2a: {opponent}", "63/100"],
        ["", "-boost", f"p2a: {opponent}", "atk", "2"],
        ["", "-status", f"p1a: {me}", "brn"],
        ["", "-item", f"p2a: {opponent}", "Leftovers"],
        ["", "-start", f"p2a: {opponent}", "Substitute"],
        ["", "-weather", "SunnyDay"],
        ["", "-fieldstart", "move: Grassy Terrain"],
        ["", "-sidestart", "p1: me", "move: Stealth Rock"],
        ["", "-sidestart", "p2: opponent", "Spikes"],
        ["", "-sidestart", "p2: opponent", "Spikes"],
        ["", "turn", "3"],
    ]:
        battle.parse_message(message)
    return battle


def check_observation(
    observation: SinglesObservation, battle: Battle, team_preview: bool
):
    obs = observation.embed(battle)
    vocab = observation.vocab

    assert observation.space.contains(obs), {
        key: (value.shape, value.dtype, value.min(), value.max())
        for key, value in obs.items()
    }
    assert all(np.isfinite(value).all() for value in obs.values())

    my_species = [vocab.species_id(mon.species) for mon in battle.team.values()]
    assert obs["pokemon_ids"][:N_TEAM, SPECIES].tolist() == my_species
    assert obs["pokemon_feats"][:N_TEAM, IS_MINE].all()
    assert (
        obs["pokemon_feats"][0, ACTIVE] == 1
        and obs["pokemon_feats"][1:N_TEAM, ACTIVE].sum() == 0
    )

    # Opponent row 0 is the active one: revealed, damaged, one known move
    opponent = obs["pokemon_feats"][N_TEAM]
    assert (
        opponent[REVEALED] == 1
        and opponent[ACTIVE] == 1
        and abs(opponent[HP] - 0.63) < 1e-6
    )
    assert obs["move_ids"][N_TEAM, 0] not in (PAD, UNKNOWN)
    assert (obs["move_ids"][N_TEAM, 1:] == UNKNOWN).all()
    assert obs["pokemon_ids"][N_TEAM, LAST_MOVE] == obs["move_ids"][N_TEAM, 0]

    # The other 5 opponents: species from team preview if there was one, else unknown
    hidden = obs["pokemon_ids"][N_TEAM + 1 :, SPECIES]
    if team_preview:
        assert (hidden > UNKNOWN).all()
    else:
        assert (hidden == UNKNOWN).all()
    assert (obs["pokemon_feats"][N_TEAM + 1 :, REVEALED] == 0).all()
    assert (obs["pokemon_feats"][N_TEAM + 1 :, HP] == 1).all()
    assert (obs["move_ids"][N_TEAM + 1 :] == UNKNOWN).all()
    return obs


def main():
    torch.manual_seed(0)
    vocab = Vocab.load()
    observation = SinglesObservation(vocab)
    print(
        f"vocab sizes: species={vocab.n_species} moves={vocab.n_moves} "
        f"items={vocab.n_items} abilities={vocab.n_abilities}"
    )

    battles, observations = [], []
    for team_preview in (True, False):
        battle = build_battle(team_preview)
        battles.append(battle)
        observations.append(check_observation(observation, battle, team_preview))
    for key, value in observations[0].items():
        print(f"{key}: {value.shape} {value.dtype}")

    env = SinglesBattleEnv(battle_format=FORMAT, vocab=vocab, start_listening=False)
    agent = env.possible_agents[0]
    observation_space = env.observation_spaces[agent]
    action_space = env.action_spaces[agent]
    assert isinstance(observation_space, spaces.Dict)
    assert isinstance(action_space, spaces.Discrete)
    assert observation_space["observation"] == observation.space
    assert observation_space["observation"].contains(env.embed_battle(battles[0]))
    n_actions = int(action_space.n)
    print(f"env spaces ok, {n_actions} actions")

    masks = torch.as_tensor(np.array([SinglesEnv.get_action_mask(b) for b in battles]))
    policy = SinglesPolicy(vocab, n_actions)
    logits, values = policy(to_tensors(observations), masks)
    assert logits.shape == (2, n_actions) and values.shape == (2,)
    assert (logits[masks == 0] == -1e9).all() and torch.isfinite(
        logits[masks == 1]
    ).all()

    for battle, battle_logits in zip(battles, logits):
        action = torch.distributions.Categorical(logits=battle_logits).sample()
        order = SinglesEnv.action_to_order(np.int64(action.item()), battle)
        print(
            f"legal actions {np.flatnonzero(SinglesEnv.get_action_mask(battle)).tolist()}, sampled {action.item()} -> {order}"
        )

    n_params = sum(p.numel() for p in policy.parameters())
    print(
        f"model ok: logits {tuple(logits.shape)}, values {tuple(values.shape)}, {n_params:,} params"
    )


if __name__ == "__main__":
    main()
