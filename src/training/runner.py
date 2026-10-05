"""
Plays many self-play battles at once.

Each env is poke-env's two-player env: seat 0 is always the learner, and seat 1 an
opponent sampled from the pool for every battle. A seat only acts when poke-env says it
has a decision to make (e.g. only one side picks a switch after a faint). Only the
learner's decisions are recorded, from both seats when it plays against itself.
"""

import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from poke_env.battle import Battle
from poke_env.environment import SinglesEnv
from poke_env.player import BattleOrder
from torch.distributions import Categorical

from src.environments.singles import SinglesBattleEnv
from src.models.singles import SinglesPolicy, to_tensors
from src.observations.vocab import Vocab
from src.training.opponents import LATEST, Opponent, OpponentPool
from src.training.ppo import Trajectory
from src.wrappers.teams import team_for_format

# Sent for seats without a decision to make, poke-env ignores it
NO_ACTION = np.int64(-2)


@dataclass
class BattleResult:
    opponent: Opponent
    won: bool | None  # seat 0's result, None for ties and battles cut at max_turns
    turns: int
    # Wall-clock time from the battle's start to its end
    seconds: float
    # Both seats' views of the battle, and the seats the learner played
    battles: tuple[Battle, Battle]
    learner_seats: tuple[int, ...]


@dataclass
class _Slot:
    """
    An env and the state of the battle it's playing.
    """

    env: SinglesBattleEnv
    opponent: Opponent = LATEST
    opponent_policy: SinglesPolicy | None = None
    started_at: float = 0.0
    observations: dict[str, Any] = field(default_factory=dict)
    trajectories: dict[int, Trajectory] = field(default_factory=dict)
    actions: list[np.int64] = field(default_factory=lambda: [NO_ACTION, NO_ACTION])


@dataclass
class _StepOutcome:
    observations: dict[str, Any]  # of the next battle if this one finished
    rewards: list[float]
    finished: bool
    won: bool | None
    turns: int
    battles: tuple[Battle, Battle]
    # When this battle ended and the next one started, if it finished
    ended_at: float = 0.0
    next_started_at: float = 0.0


class SelfPlayRunner:
    def __init__(
        self,
        battle_format: str,
        n_envs: int,
        pool: OpponentPool,
        vocab: Vocab,
        *,
        device: str = "cpu",
        max_turns: int = 300,
        env_kwargs: dict[str, Any] | None = None,
    ):
        """
        env_kwargs go to every SinglesBattleEnv, e.g. the reward weights.
        """
        self.pool = pool
        self.device = device
        self.max_turns = max_turns
        self.slots = [
            _Slot(
                SinglesBattleEnv(
                    vocab=vocab,
                    battle_format=battle_format,
                    team=team_for_format(battle_format),
                    choose_on_teampreview=False,
                    log_level=40,
                    open_timeout=None,
                    strict=False,
                    **(env_kwargs or {}),
                )
            )
            for _ in range(n_envs)
        ]
        # env.step and env.reset block until Showdown answers, so all envs run at once
        self.executor = ThreadPoolExecutor(max_workers=n_envs)
        resets = self.executor.map(lambda slot: slot.env.reset()[0], self.slots)
        for slot, observations in zip(self.slots, resets):
            self._start_battle(slot, observations)

    def collect(
        self, policy: SinglesPolicy, n_transitions: int
    ) -> tuple[list[Trajectory], list[BattleResult]]:
        """
        Plays until the finished battles have at least n_transitions learner decisions.
        Battles still going on carry over to the next call.
        """
        trajectories: list[Trajectory] = []
        results: list[BattleResult] = []
        n_collected = 0
        while n_collected < n_transitions:
            self._choose_actions(policy)
            outcomes = self.executor.map(self._step, self.slots)
            for slot, outcome in zip(self.slots, outcomes):
                for seat, trajectory in slot.trajectories.items():
                    trajectory.add_reward(outcome.rewards[seat])
                if not outcome.finished:
                    slot.observations = outcome.observations
                    continue

                finished = [t for t in slot.trajectories.values() if len(t)]
                trajectories.extend(finished)
                n_collected += sum(len(t) for t in finished)
                results.append(
                    BattleResult(
                        slot.opponent,
                        outcome.won,
                        outcome.turns,
                        seconds=outcome.ended_at - slot.started_at,
                        battles=outcome.battles,
                        learner_seats=tuple(slot.trajectories),
                    )
                )
                self.pool.record(slot.opponent, outcome.won)
                self._start_battle(slot, outcome.observations, outcome.next_started_at)
        return trajectories, results

    def set_shaping_weight(self, weight: float):
        for slot in self.slots:
            slot.env.shaping_weight = weight

    def close(self):
        for slot in self.slots:
            slot.env.close()
        self.executor.shutdown()

    def _start_battle(
        self,
        slot: _Slot,
        observations: dict[str, Any],
        started_at: float | None = None,
    ):
        slot.started_at = started_at or time.perf_counter()
        slot.opponent = self.pool.sample()
        # Loaded now, so the snapshot can leave the pool while this battle goes on
        slot.opponent_policy = (
            self.pool.policy_for(slot.opponent)
            if slot.opponent.kind == "snapshot"
            else None
        )
        slot.observations = observations
        slot.trajectories = {0: Trajectory()}
        if slot.opponent == LATEST:
            slot.trajectories[1] = Trajectory()

    def _choose_actions(self, policy: SinglesPolicy):
        # Decisions grouped by who makes them, to batch the forward passes
        groups: dict[Opponent, list[tuple[_Slot, int]]] = defaultdict(list)
        for slot in self.slots:
            slot.actions = [NO_ACTION, NO_ACTION]
            to_move = (slot.env.agent1_to_move, slot.env.agent2_to_move)
            for seat in (0, 1):
                if to_move[seat]:
                    groups[LATEST if seat == 0 else slot.opponent].append((slot, seat))

        for opponent, decisions in groups.items():
            if opponent.kind == "scripted":
                for slot, seat in decisions:
                    slot.actions[seat] = self._scripted_action(opponent, slot)
                continue
            model = policy if opponent == LATEST else decisions[0][0].opponent_policy
            assert model is not None
            self._model_actions(model, decisions)

    def _model_actions(self, model: SinglesPolicy, decisions: list[tuple[_Slot, int]]):
        observations = [
            slot.observations[slot.env.possible_agents[seat]]
            for slot, seat in decisions
        ]
        obs = to_tensors([o["observation"] for o in observations], self.device)
        masks = torch.as_tensor(
            np.stack([o["action_mask"] for o in observations]), device=self.device
        )
        with torch.no_grad():
            logits, values = model(obs, masks)
            dist = Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

        for i, (slot, seat) in enumerate(decisions):
            action = int(actions[i].item())
            slot.actions[seat] = np.int64(action)
            # Only the learner's seats have trajectories
            trajectory = slot.trajectories.get(seat)
            if trajectory is not None:
                trajectory.add(
                    observations[i]["observation"],
                    observations[i]["action_mask"],
                    action,
                    log_probs[i].item(),
                    values[i].item(),
                )

    def _scripted_action(self, opponent: Opponent, slot: _Slot) -> np.int64:
        battle = slot.env.battle2
        assert isinstance(battle, Battle)
        order = self.pool.player_for(opponent).choose_move(battle)
        assert isinstance(order, BattleOrder)
        return SinglesEnv.order_to_action(order, battle, strict=False)

    def _step(self, slot: _Slot) -> _StepOutcome:
        """
        Runs in a worker thread: steps the env, and starts the next battle if this one
        finished or went past max_turns (forfeiting it).
        """
        env = slot.env
        agents = env.possible_agents
        observations, rewards, _, _, _ = env.step(dict(zip(agents, slot.actions)))

        # Kept before env.reset() replaces them, for the battle stats
        battle, other_battle = env.battle1, env.battle2
        assert isinstance(battle, Battle) and isinstance(other_battle, Battle)
        outcome = _StepOutcome(
            observations=observations,
            rewards=[rewards[agent] for agent in agents],
            finished=battle.finished or battle.turn >= self.max_turns,
            won=battle.won if battle.finished else None,
            turns=battle.turn,
            battles=(battle, other_battle),
        )
        if outcome.finished:
            outcome.ended_at = time.perf_counter()
            outcome.observations = env.reset()[0]
            outcome.next_started_at = time.perf_counter()
        return outcome
