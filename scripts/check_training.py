"""
Offline checks of the training code, no Showdown server needed: config loading, GAE,
reward, random teams, the opponent pool, battle stats, the TensorBoard logger,
PolicyPlayer and a PPO update.

Run from the project root:
    .venv/bin/python -m scripts.check_training
"""

import json
import shutil
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import torch
from poke_env.battle import Battle
from poke_env.data import GenData, to_id_str
from poke_env.environment import SinglesEnv
from poke_env.teambuilder import Teambuilder
from torch.distributions import Categorical

from scripts.check_observation import build_battle
from src.environments.singles import SinglesBattleEnv
from src.models.singles import SinglesPolicy, to_tensors
from src.observations.singles import SinglesObservation
from src.observations.vocab import Vocab
from src.players.policy_player import PolicyPlayer
from src.training.config import (
    DEFAULT_CONFIG,
    TrainConfig,
    load_config,
    read_toml,
    save_config,
)
from src.training.logger import RunLogger
from src.training.opponents import OpponentPool
from src.training.ppo import PPOConfig, Trajectory, compute_gae, make_batch, ppo_update
from src.training.stats import BattleStats, action_mix, parse_battle_log
from src.utils.load_pokemon import choose_random_team_from_format
from src.wrappers.teams import RandomTeam

N_ACTIONS = 26  # gen 9

# Turn 1: Garchomp KOs Gholdengo, then Corviknight comes in and faints to Stealth Rock.
# Turn 2: Kingambit bounces a Stealth Rock back with Magic Bounce, then KOs Garchomp.
BATTLE_LOG = [
    ["", "turn", "1"],
    ["", "move", "p1a: Garchomp", "Earthquake", "p2a: Gholdengo"],
    ["", "-damage", "p2a: Gholdengo", "0 fnt"],
    ["", "faint", "p2a: Gholdengo"],
    ["", "switch", "p2a: Corviknight", "Corviknight, L80, M", "100/100"],
    ["", "-damage", "p2a: Corviknight", "0 fnt", "[from] Stealth Rock"],
    ["", "turn", "2"],
    ["", "move", "p2a: Kingambit", "Stealth Rock", "", "[from] ability: Magic Bounce"],
    ["", "move", "p2a: Kingambit", "Sucker Punch", "p1a: Garchomp"],
    ["", "-damage", "p1a: Garchomp", "0 fnt"],
]


def check_config():
    defaults = asdict(TrainConfig())
    # configs/selfplay.toml has every setting but resume
    from_file = read_toml(DEFAULT_CONFIG)
    assert set(from_file) == set(defaults) - {"resume"}, set(from_file) ^ set(defaults)
    # Without --config, new runs read it
    assert asdict(load_config([])) == {**defaults, **from_file}

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "test.toml"
        path.write_text("[run]\nn_envs = 8\nbatch_size = 1024\n[ppo]\nent_coef = 0\n")
        cfg = load_config(["--config", str(path), "--batch-size", "512"])
        # The file beats the defaults, and flags beat the file
        assert (cfg.n_envs, cfg.batch_size, cfg.ent_coef, cfg.lr) == (8, 512, 0.0, 3e-4)

        path.write_text("[run]\nn_env = 8\n")
        try:
            load_config(["--config", str(path)])
        except SystemExit as error:
            assert "n_env" in str(error)
        else:
            raise AssertionError("an unknown setting was accepted")

    # Resuming starts from the run's saved settings
    run_dir = TrainConfig(run_name="check_training").run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        saved = TrainConfig(run_name="check_training", n_envs=3, lr=1e-3)
        save_config(saved, run_dir / "config.json")
        cfg = load_config(["--run-name", "check_training", "--resume", "--lr", "2e-3"])
        assert (cfg.n_envs, cfg.lr, cfg.resume) == (3, 2e-3, True)
    finally:
        shutil.rmtree(run_dir)
    print("config ok")


def check_gae():
    rewards = np.array([0.0, 0.0, 1.0])
    values = np.array([0.5, 0.5, 0.5])
    # Without discounting, every advantage is the final reward minus the value
    assert np.allclose(compute_gae(rewards, values, gamma=1.0, lam=1.0), 0.5)
    # deltas are -0.05, -0.05, 0.5, and each advantage adds 0.72 times the next one
    advantages = compute_gae(rewards, values, gamma=0.9, lam=0.8)
    assert np.allclose(advantages, [0.1732, 0.31, 0.5]), advantages
    print("gae ok")


def check_trajectory():
    trajectory = Trajectory()
    trajectory.add_reward(5.0)  # before any decision, dropped
    trajectory.add({}, np.ones(1), 0, 0.0, 0.0)
    trajectory.add_reward(0.25)
    trajectory.add_reward(0.5)
    trajectory.add({}, np.ones(1), 1, 0.0, 0.0)
    trajectory.add_reward(1.0)
    assert trajectory.rewards == [0.75, 1.0], trajectory.rewards
    print("trajectory rewards ok")


def check_reward(vocab: Vocab):
    env = SinglesBattleEnv(vocab=vocab, battle_format="gen9ou", start_listening=False)
    battle = build_battle(team_preview=True)
    assert np.isfinite(env.calc_reward(battle))
    assert env.calc_reward(battle) == 0.0  # nothing changed since the last call
    battle.won_by("me")
    assert env.calc_reward(battle) == 1.0
    print("reward ok")


def check_teams():
    pokedex = GenData.from_gen(9).pokedex
    for _ in range(20):
        team = Teambuilder.parse_showdown_team(choose_random_team_from_format("gen9ou"))
        species = [to_id_str(mon.species or mon.nickname or "") for mon in team]
        numbers = {pokedex.get(s, {}).get("num", s) for s in species}
        assert len(team) == 6 and len(numbers) == 6, species
    packed = RandomTeam("gen9ou").yield_team()
    assert len(Teambuilder.parse_packed_team(packed)) == 6
    print("random teams ok")


def check_pool(vocab: Vocab):
    with tempfile.TemporaryDirectory() as directory:
        pool = OpponentPool(Path(directory), vocab, "gen9ou", max_snapshots=2, seed=0)
        # Without snapshots, only the latest policy and the scripted bots
        assert {pool.sample().kind for _ in range(200)} == {"latest", "scripted"}

        policy = SinglesPolicy(vocab, N_ACTIONS)
        for update in (1, 2, 3):
            pool.add_snapshot(policy, update)
        assert [s.name for s in pool.snapshots] == [
            "snapshot_000002",
            "snapshot_000003",
        ]
        assert len(list(Path(directory).glob("*.pt"))) == 2
        loaded = pool.policy_for(pool.snapshots[0])
        for a, b in zip(policy.state_dict().values(), loaded.state_dict().values()):
            assert torch.equal(a, b)
        assert {pool.sample().kind for _ in range(200)} == {
            "latest",
            "snapshot",
            "scripted",
        }

        # PFSP picks the snapshot the learner loses to more often
        pool.p_latest, pool.p_scripted, pool.sampling = 0.0, 0.0, "pfsp"
        beaten, unbeaten = pool.snapshots
        for _ in range(20):
            pool.record(beaten, won=True)
            pool.record(unbeaten, won=False)
        counts = Counter(pool.sample() for _ in range(500))
        assert counts[unbeaten] > 5 * counts[beaten], counts
    print("opponent pool ok")


def check_action_mix():
    mix = action_mix(np.array([0, 5, 6, 9, 22, 25]), N_ACTIONS)
    assert mix == {
        "switch_rate": 2 / 6,
        "move_rate": 2 / 6,
        "mega_rate": 0.0,
        "z_move_rate": 0.0,
        "dynamax_rate": 0.0,
        "tera_rate": 2 / 6,
    }, mix
    # Gens 1 to 5 have no gimmicks
    assert set(action_mix(np.array([0, 6]), 10)) == {"switch_rate", "move_rate"}
    print("action mix ok")


def fake_battle(role: str, names: list[str], log: list[list[str]]) -> Battle:
    """
    Just what BattleStats reads from a finished battle.
    """
    team = {
        f"{role}: {name}": SimpleNamespace(
            species=to_id_str(name), revealed=True, fainted=False
        )
        for name in names
    }
    return cast(Battle, SimpleNamespace(player_role=role, team=team, _replay_data=log))


def check_stats():
    species_by_ident = {
        f"{side}: {name}": to_id_str(name)
        for side, names in (
            ("p1", ["Garchomp"]),
            ("p2", ["Gholdengo", "Corviknight", "Kingambit"]),
        )
        for name in names
    }
    log = parse_battle_log(BATTLE_LOG, species_by_ident)
    assert log.move_uses["p1"] == Counter({("garchomp", "earthquake"): 1})
    # The Stealth Rock was called by Magic Bounce, so it doesn't count
    assert log.move_uses["p2"] == Counter({("kingambit", "suckerpunch"): 1})
    # The Stealth Rock KO on Corviknight isn't credited to Earthquake
    assert log.kos["p1"] == Counter({("garchomp", "earthquake"): 1})
    assert log.kos["p2"] == Counter({("kingambit", "suckerpunch"): 1})

    battles = (
        fake_battle("p1", ["Garchomp", "Dragapult"], BATTLE_LOG),
        fake_battle("p2", ["Gholdengo", "Corviknight", "Kingambit"], BATTLE_LOG),
    )
    stats = BattleStats()
    stats.record(battles, learner_seats=(0,), won=True)
    # Self-play battle: the learner played both sides, seat 1 won this time
    stats.record(battles, learner_seats=(0, 1), won=False)
    garchomp = stats.species["garchomp"]
    assert (garchomp["games"], garchomp["wins"], garchomp["kos"]) == (2, 1, 2)
    assert stats.species["kingambit"]["wins"] == 1
    assert stats.moves["suckerpunch"]["kos"] == 1
    assert stats.opponents["kingambit"]["kos"] == 2
    assert stats.opponents["garchomp"]["kos"] == 1

    tables = stats.tables(top_k=5, min_games=1)
    assert "garchomp" in tables["best_pokemon"]
    assert tables["toughest_opponents"].startswith("| opponent |")
    assert "Not enough games" in stats.tables(top_k=5, min_games=100)["best_pokemon"]

    with tempfile.TemporaryDirectory() as directory:
        stats.write_csv(Path(directory))
        for name in ("species", "moves", "opponents"):
            header = (Path(directory) / f"{name}.csv").read_text().splitlines()[0]
            assert header.startswith(name) and "win_rate" in header, header
        stats.save(Path(directory) / "state.json")
        loaded = BattleStats.load(Path(directory) / "state.json")
        assert loaded.species == stats.species and loaded.moves == stats.moves
    print("battle stats ok")


def check_logger():
    with tempfile.TemporaryDirectory() as directory:
        logger = RunLogger(Path(directory))
        logger.scalars(1, {"loss/policy": 0.5, "matches/wins": 3})
        actions = np.array([0, 6, 6, 22])
        logger.histogram("policy/actions", actions, 1, bins=np.arange(27) - 0.5)
        logger.text("stats/best_pokemon", "| Pokémon | games |\n|---|---|", 1)
        logger.close()

        assert list((Path(directory) / "tensorboard").glob("events.out.tfevents.*"))
        line = json.loads((Path(directory) / "log.jsonl").read_text())
        assert line == {"update": 1, "loss/policy": 0.5, "matches/wins": 3}, line
    print("logger ok")


def check_policy_player(vocab: Vocab, observation: SinglesObservation):
    policy = SinglesPolicy(vocab, N_ACTIONS)
    player = PolicyPlayer(
        policy, observation, battle_format="gen9ou", start_listening=False
    )
    battle = build_battle(team_preview=True)
    valid_orders = {str(order) for order in battle.valid_orders}
    for _ in range(10):
        assert str(player.choose_move(battle)) in valid_orders
    print("policy player ok")


def check_ppo(vocab: Vocab, observation: SinglesObservation):
    torch.manual_seed(0)
    policy = SinglesPolicy(vocab, N_ACTIONS)
    trajectories = []
    for length in (3, 5, 8):
        trajectory = Trajectory()
        for i in range(length):
            battle = build_battle(team_preview=i % 2 == 0)
            obs = observation.embed(battle)
            mask = np.array(SinglesEnv.get_action_mask(battle), dtype=np.int8)
            with torch.no_grad():
                logits, value = policy(to_tensors([obs]), torch.as_tensor(mask[None]))
            dist = Categorical(logits=logits)
            action = dist.sample()
            trajectory.add(
                obs,
                mask,
                int(action.item()),
                dist.log_prob(action).item(),
                value.item(),
            )
            trajectory.add_reward(float(np.random.randn()) / 10)
        trajectories.append(trajectory)

    batch = make_batch(trajectories, gamma=0.99, lam=0.95)
    assert len(batch) == 16 and batch.observations["pokemon_feats"].shape[0] == 16

    before = [p.detach().clone() for p in policy.parameters()]
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    stats = ppo_update(
        policy, optimizer, batch, PPOConfig(epochs=2, minibatch_size=8, target_kl=0)
    )
    assert all(np.isfinite(value) for value in stats.values()), stats
    assert any(not torch.equal(b, p) for b, p in zip(before, policy.parameters()))
    print("ppo update ok:", {key: round(value, 4) for key, value in stats.items()})


def main():
    vocab = Vocab.load()
    observation = SinglesObservation(vocab)
    check_config()
    check_gae()
    check_trajectory()
    check_reward(vocab)
    check_teams()
    check_pool(vocab)
    check_action_mix()
    check_stats()
    check_logger()
    check_policy_player(vocab, observation)
    check_ppo(vocab, observation)


if __name__ == "__main__":
    main()
