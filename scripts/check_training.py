"""
Offline checks of the training code, no Showdown server needed: GAE, reward, random
teams, the opponent pool, PolicyPlayer and a PPO update.

Run from the project root:
    .venv/bin/python -m scripts.check_training
"""

import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
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
from src.training.opponents import OpponentPool
from src.training.ppo import PPOConfig, Trajectory, compute_gae, make_batch, ppo_update
from src.utils.load_pokemon import choose_random_team_from_format
from src.wrappers.teams import RandomTeam

N_ACTIONS = 26  # gen 9


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
    check_gae()
    check_trajectory()
    check_reward(vocab)
    check_teams()
    check_pool(vocab)
    check_policy_player(vocab, observation)
    check_ppo(vocab, observation)


if __name__ == "__main__":
    main()
