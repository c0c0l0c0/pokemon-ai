"""
Playground to learn the poke_env library, printing everything along the way.

Run from the project root:
    .venv/bin/python -m scripts.explore_poke_env

Parts 1-4 work offline. Parts 5-7 play real battles, so they need a local
Pokemon Showdown server on localhost:8000:
    git clone https://github.com/smogon/pokemon-showdown.git
    cd pokemon-showdown && npm install
    cp config/config-example.js config/config.js
    node pokemon-showdown start --no-security
"""

import asyncio
import socket
from random import choice

import numpy as np
from gymnasium.spaces import Box
from poke_env import RandomPlayer, SimpleHeuristicsPlayer
from poke_env.battle import AbstractBattle, Battle, Move, Pokemon, PokemonType
from poke_env.data import GenData, to_id_str
from poke_env.environment import SingleAgentWrapper, SinglesEnv
from poke_env.player import Player
from poke_env.stats import compute_raw_stats
from poke_env.teambuilder import Teambuilder

from src.utils.load_pokemon import choose_random_pokemon_from_format

FORMAT = "gen9ou"
GEN = 9


def section(title: str):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def server_is_up(host: str = "localhost", port: int = 8000) -> bool:
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0


# ---------------------------------------------------------------------------
# 1. Static game data (GenData)
# ---------------------------------------------------------------------------
def explore_data():
    section("1. GenData: the static Pokedex / moves / type chart")

    data = GenData.from_gen(GEN)
    print(f"GenData.from_gen({GEN}) -> {data}")
    print(f"Pokedex entries: {len(data.pokedex)}, moves: {len(data.moves)}")

    # Everything is keyed by Showdown ids: lowercase, no spaces/punctuation
    name = "Great Tusk"
    print(f"\nto_id_str({name!r}) -> {to_id_str(name)!r}")

    entry = data.pokedex[to_id_str(name)]
    print(f"Pokedex keys: {sorted(entry.keys())}")
    print(f"types: {entry['types']}")
    print(f"baseStats: {entry['baseStats']}")
    print(f"abilities: {entry['abilities']}")

    move = data.moves["headlongrush"]
    print(f"\nMove 'headlongrush' raw data keys: {sorted(move.keys())}")
    print(
        f"basePower={move['basePower']} type={move['type']} "
        f"category={move['category']} accuracy={move['accuracy']}"
    )

    print(f"\nType chart: data.type_chart['FIRE'] = {data.type_chart['FIRE']}")
    print("(read as: how much damage each attacking type does TO a Fire type)")


# ---------------------------------------------------------------------------
# 2. Battle objects you can build offline: Move, Pokemon, PokemonType
# ---------------------------------------------------------------------------
def explore_objects():
    section("2. Move / Pokemon / PokemonType objects")

    data = GenData.from_gen(GEN)

    thunderbolt = Move("thunderbolt", gen=GEN)
    print(f"Move('thunderbolt') -> {thunderbolt!r}")
    print(
        f"  base_power={thunderbolt.base_power} type={thunderbolt.type} "
        f"category={thunderbolt.category} accuracy={thunderbolt.accuracy} "
        f"pp={thunderbolt.max_pp} priority={thunderbolt.priority}"
    )

    gyarados = Pokemon(gen=GEN, species="gyarados")
    print(f"\nPokemon(species='gyarados') -> {gyarados!r}")
    print(f"  types={gyarados.types}")
    print(f"  base_stats={gyarados.base_stats}")
    print(
        f"  Thunderbolt vs Gyarados multiplier: "
        f"{gyarados.damage_multiplier(thunderbolt)}"
    )
    print(
        f"  Ground-type vs Gyarados multiplier: "
        f"{gyarados.damage_multiplier(PokemonType.GROUND)}"
    )

    fire = PokemonType.FIRE
    print(f"\nPokemonType.FIRE -> {fire!r}")
    print(
        f"  Fire attacking Grass/Steel (Ferrothorn): "
        f"{fire.damage_multiplier(PokemonType.GRASS, PokemonType.STEEL, type_chart=data.type_chart)}"
    )

    # Real stats from base stats + EVs/IVs/level/nature
    evs = [0, 252, 4, 0, 0, 252]
    ivs = [31] * 6
    raw = compute_raw_stats("greattusk", evs, ivs, 100, "jolly", data)
    print(
        f"\ncompute_raw_stats(Great Tusk, 252 Atk/4 Def/252 Spe, Jolly, Lv100) -> {raw}"
    )
    print("  order: [hp, atk, def, spa, spd, spe]")


# ---------------------------------------------------------------------------
# 3. Teams: showdown text <-> TeambuilderPokemon <-> packed format
# ---------------------------------------------------------------------------
def random_team_text(format: str) -> str:
    """Six random sets from data/showdown_sets, no repeated species."""
    sets: dict[str, str] = {}
    while len(sets) < 6:
        pokemon = choose_random_pokemon_from_format(format).strip()
        species = pokemon.split("\n")[0].split(" @ ")[0]
        sets.setdefault(species, pokemon)
    return "\n\n".join(sets.values())


class RandomSetsTeambuilder(Teambuilder):
    """Gives a new random team (packed format) every battle."""

    def __init__(self, format: str):
        self.format = format

    def yield_team(self) -> str:
        return self.join_team(self.parse_showdown_team(random_team_text(self.format)))


def explore_teams():
    section("3. Teams: showdown text -> TeambuilderPokemon -> packed")

    text = random_team_text(FORMAT)
    print(f"Showdown text (what you'd paste in the teambuilder):\n\n{text}\n")

    mons = Teambuilder.parse_showdown_team(text)
    print(f"parse_showdown_team -> list of {len(mons)} TeambuilderPokemon")
    first = mons[0]
    print(
        f"  first: nickname={first.nickname!r} species={first.species!r} "
        "(species only set when there's a nickname)"
    )
    print(
        f"         item={first.item!r} "
        f"ability={first.ability!r} nature={first.nature!r}"
    )
    print(f"         moves={first.moves} tera_type={first.tera_type!r}")
    print(f"         evs={first.evs} ivs={first.ivs}")

    packed = Teambuilder.join_team(mons)
    print("\njoin_team -> packed format (what Showdown receives), one mon per ']':")
    for mon in packed.split("]"):
        print(f"  {mon}")

    builder = RandomSetsTeambuilder(FORMAT)
    print(
        f"\nA Teambuilder just needs yield_team(); called once per battle:\n"
        f"  {builder.yield_team()[:100]}..."
    )


# ---------------------------------------------------------------------------
# 4. What a Battle object looks like before any messages arrive
# ---------------------------------------------------------------------------
def explore_empty_battle():
    section("4. An empty Battle object")

    import logging

    battle = Battle("battle-gen9ou-test", "me", logging.getLogger("test"), gen=GEN)
    print(f"Battle(...) -> {battle!r}")
    print(f"  battle_tag={battle.battle_tag} turn={battle.turn} finished={battle.finished}")
    print(f"  team={battle.team} active_pokemon={battle.active_pokemon}")
    print(
        "The Player fills this in from Showdown protocol messages during a real battle"
        " (parts 5-7)."
    )


# ---------------------------------------------------------------------------
# 5. A custom Player that prints the battle state every turn
# ---------------------------------------------------------------------------
def describe(pokemon: Pokemon | None) -> str:
    if pokemon is None:
        return "None"
    status = pokemon.status.name if pokemon.status else "-"
    boosts = {k: v for k, v in pokemon.boosts.items() if v}
    return (
        f"{pokemon.species} {pokemon.current_hp_fraction:.0%} HP, "
        f"status={status}, types={[t.name for t in pokemon.types]}, boosts={boosts}"
    )


class VerbosePlayer(Player):
    """Picks the highest damage move (or a random switch), printing everything."""

    def choose_move(self, battle: AbstractBattle):
        assert isinstance(battle, Battle)
        print(f"\n--- {self.username} | {battle.battle_tag} | turn {battle.turn} ---")
        print(f"  me:       {describe(battle.active_pokemon)}")
        print(f"  opponent: {describe(battle.opponent_active_pokemon)}")
        print(f"  weather={battle.weather} fields={battle.fields}")
        print(f"  my side conditions={battle.side_conditions}")
        print(
            f"  available_moves="
            f"{[(m.id, m.base_power, m.type.name) for m in battle.available_moves]}"
        )
        print(f"  available_switches={[p.species for p in battle.available_switches]}")
        print(f"  can_tera={battle.can_tera} force_switch={battle.force_switch}")
        print(f"  valid_orders ({len(battle.valid_orders)}): {battle.valid_orders[:5]}...")

        opponent = battle.opponent_active_pokemon
        if battle.available_moves and opponent is not None:
            move = max(
                battle.available_moves,
                key=lambda m: m.base_power * opponent.damage_multiplier(m),
            )
            order = self.create_order(move)
        elif battle.available_switches:
            order = self.create_order(choice(battle.available_switches))
        else:
            order = self.choose_random_move(battle)

        print(f"  -> chosen order: {order}")
        return order

    def _battle_finished_callback(self, battle: AbstractBattle):
        result = "WON" if battle.won else "LOST" if battle.lost else "TIED"
        print(f"\n*** {self.username} {result} {battle.battle_tag} in {battle.turn} turns")
        print(f"    my team at the end:       {battle.team}")
        print(f"    opponent team at the end: {battle.opponent_team}")


async def explore_battle():
    section("5. One verbose battle: VerbosePlayer vs RandomPlayer")

    me = VerbosePlayer(battle_format=FORMAT, team=RandomSetsTeambuilder(FORMAT))
    opponent = RandomPlayer(battle_format=FORMAT, team=RandomSetsTeambuilder(FORMAT))
    print(f"players: {me.username} vs {opponent.username}")

    await me.battle_against(opponent, n_battles=1)

    print(f"\nme.battles: {list(me.battles)}")
    print(f"won={me.n_won_battles} finished={me.n_finished_battles}")


# ---------------------------------------------------------------------------
# 6. Many quiet battles between the built-in baselines
# ---------------------------------------------------------------------------
async def explore_baselines(n_battles: int = 20):
    section(f"6. Baselines: {n_battles} battles, SimpleHeuristicsPlayer vs RandomPlayer")

    heuristic = SimpleHeuristicsPlayer(
        battle_format=FORMAT, team=RandomSetsTeambuilder(FORMAT)
    )
    random = RandomPlayer(battle_format=FORMAT, team=RandomSetsTeambuilder(FORMAT))

    await heuristic.battle_against(random, n_battles=n_battles)
    print(
        f"SimpleHeuristicsPlayer won {heuristic.n_won_battles}/"
        f"{heuristic.n_finished_battles} (win rate {heuristic.win_rate:.0%})"
    )


# ---------------------------------------------------------------------------
# 7. The Gymnasium side: SinglesEnv + SingleAgentWrapper
# ---------------------------------------------------------------------------
class PlaygroundEnv(SinglesEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 4 move base powers + 4 move multipliers + fainted fractions (mine, opponent)
        low = np.array([-1] * 4 + [0] * 4 + [0, 0], dtype=np.float32)
        high = np.array([3] * 4 + [4] * 4 + [1, 1], dtype=np.float32)
        self.observation_spaces = {  # type: ignore[assignment]
            agent: Box(low, high, dtype=np.float32) for agent in self.possible_agents
        }

    def calc_reward(self, battle: AbstractBattle) -> float:
        return self.reward_computing_helper(
            battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        base_power = -np.ones(4)
        multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves[:4]):
            base_power[i] = move.base_power / 100
            if battle.opponent_active_pokemon is not None:
                multiplier[i] = battle.opponent_active_pokemon.damage_multiplier(move)

        fainted = sum(p.fainted for p in battle.team.values()) / 6
        opp_fainted = sum(p.fainted for p in battle.opponent_team.values()) / 6
        return np.concatenate(
            [base_power, multiplier, [fainted, opp_fainted]]
        ).astype(np.float32)


def explore_env(max_steps: int = 10):
    section("7. Gymnasium env: SinglesEnv wrapped in SingleAgentWrapper")

    env = PlaygroundEnv(
        battle_format=FORMAT,
        team=RandomSetsTeambuilder(FORMAT),
        log_level=40,
        choose_on_teampreview=False,
        strict=False,
    )
    print(f"agents (two Showdown accounts): {env.possible_agents}")
    print(f"action space per agent: {env.action_spaces[env.possible_agents[0]]}")

    wrapped = SingleAgentWrapper(env, SimpleHeuristicsPlayer(start_listening=False))
    print(f"\nwrapped.observation_space: {wrapped.observation_space}")
    print(f"wrapped.action_space: {wrapped.action_space}")

    obs, info = wrapped.reset()
    print(f"\nreset() -> obs keys {list(obs)}")
    print(f"  observation={obs['observation']}")
    print(f"  action_mask={obs['action_mask']}")
    print("  (0-5 switch, 6-9 move, 10+ move with a gimmick; 1 = legal now)")

    for step in range(max_steps):
        action = wrapped.action_space.sample(obs["action_mask"])
        obs, reward, terminated, truncated, info = wrapped.step(action)
        print(
            f"\nstep {step}: action={action} reward={reward:.2f} "
            f"terminated={terminated} truncated={truncated}"
        )
        print(f"  observation={obs['observation']}")
        print(f"  legal actions={np.flatnonzero(obs['action_mask']).tolist()}")
        print(f"  info={info}")
        if terminated or truncated:
            break

    wrapped.close()


def main():
    explore_data()
    explore_objects()
    explore_teams()
    explore_empty_battle()

    if not server_is_up():
        section("Parts 5-7 skipped: no Showdown server on localhost:8000")
        print(__doc__)
        return

    asyncio.run(explore_battle())
    asyncio.run(explore_baselines())
    explore_env()


if __name__ == "__main__":
    main()
