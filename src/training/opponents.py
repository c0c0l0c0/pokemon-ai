"""
Opponents for self-play: the latest policy, frozen snapshots of past policies, and
poke-env's scripted bots.

Sampling past snapshots uniformly is fictitious self-play: the learner plays against the
average of its past selves, which keeps it from going in circles (rock-paper-scissors
style) and from forgetting how to beat older strategies.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from poke_env.player import (
    MaxBasePowerPlayer,
    Player,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)

from src.models.singles import SinglesPolicy, load_policy, save_policy
from src.observations.vocab import Vocab

SCRIPTED_PLAYERS: dict[str, type[Player]] = {
    "random": RandomPlayer,
    "max_base_power": MaxBasePowerPlayer,
    "heuristics": SimpleHeuristicsPlayer,
}


@dataclass(frozen=True)
class Opponent:
    kind: str  # "latest", "snapshot" or "scripted"
    name: str


LATEST = Opponent("latest", "latest")


class OpponentPool:
    def __init__(
        self,
        directory: Path,
        vocab: Vocab,
        battle_format: str,
        *,
        p_latest: float = 0.5,
        p_scripted: float = 0.1,
        sampling: str = "uniform",
        max_snapshots: int = 50,
        device: str = "cpu",
        seed: int | None = None,
    ):
        """
        Each battle is against the latest policy with probability p_latest, a scripted
        bot with probability p_scripted, and a past snapshot otherwise. Snapshots are
        sampled uniformly (fictitious self-play), or with "pfsp" more often the more
        they beat the learner (prioritized fictitious self-play, from AlphaStar).
        """
        assert sampling in ("uniform", "pfsp"), sampling
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.vocab = vocab
        self.p_latest = p_latest
        self.p_scripted = p_scripted
        self.sampling = sampling
        self.max_snapshots = max_snapshots
        self.device = device
        self.rng = np.random.default_rng(seed)

        # Snapshots already on disk, e.g. when resuming a run
        self.snapshots = [
            Opponent("snapshot", path.stem)
            for path in sorted(self.directory.glob("*.pt"))
        ]
        # Scripted bots only pick moves for env battles, so they don't connect
        self.scripted = {
            name: cls(battle_format=battle_format, start_listening=False, log_level=40)
            for name, cls in SCRIPTED_PLAYERS.items()
        }
        self.wins: dict[Opponent, float] = defaultdict(float)
        self.games: dict[Opponent, int] = defaultdict(int)
        self._policies: dict[Opponent, SinglesPolicy] = {}

    def sample(self) -> Opponent:
        r = self.rng.random()
        if r < self.p_scripted:
            return Opponent("scripted", str(self.rng.choice(list(self.scripted))))
        if r < self.p_scripted + self.p_latest or not self.snapshots:
            return LATEST

        if self.sampling == "pfsp":
            # Weight (1 - learner win rate)^2: focus on the snapshots that beat us
            win_rates = np.array([self.win_rate(s) for s in self.snapshots])
            weights = (1 - win_rates) ** 2 + 1e-3
            i = self.rng.choice(len(self.snapshots), p=weights / weights.sum())
        else:
            i = self.rng.integers(len(self.snapshots))
        return self.snapshots[int(i)]

    def add_snapshot(self, policy: SinglesPolicy, update: int) -> Opponent:
        snapshot = Opponent("snapshot", f"snapshot_{update:06d}")
        save_policy(policy, self._path(snapshot))
        self.snapshots.append(snapshot)

        while len(self.snapshots) > self.max_snapshots:
            oldest = self.snapshots.pop(0)
            self._policies.pop(oldest, None)
            self._path(oldest).unlink(missing_ok=True)
        return snapshot

    def policy_for(self, opponent: Opponent) -> SinglesPolicy:
        if opponent not in self._policies:
            self._policies[opponent] = load_policy(
                self._path(opponent), self.vocab, self.device
            )
        return self._policies[opponent]

    def player_for(self, opponent: Opponent) -> Player:
        return self.scripted[opponent.name]

    def record(self, opponent: Opponent, won: bool | None):
        """
        Records the learner's result against the opponent, ties count as half a win.
        """
        self.games[opponent] += 1
        self.wins[opponent] += 1.0 if won else 0.5 if won is None else 0.0

    def win_rate(self, opponent: Opponent) -> float:
        # Starts at 0.5 for opponents without games
        return (self.wins[opponent] + 1) / (self.games[opponent] + 2)

    def _path(self, opponent: Opponent) -> Path:
        return self.directory / f"{opponent.name}.pt"
