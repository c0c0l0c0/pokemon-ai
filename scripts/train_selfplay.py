"""
Self-play PPO training for SinglesPolicy.

Needs a local Showdown server: node pokemon-showdown start --no-security

Run from the project root:
    .venv/bin/python -m scripts.train_selfplay --run-name run1
    .venv/bin/python -m scripts.train_selfplay --run-name run1 --resume
    .venv/bin/python -m scripts.train_selfplay --help

Writes checkpoints/<run-name>/latest.pt, log.jsonl and the pool/ of past snapshots.
"""

import argparse
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch
from poke_env.data import GenData
from poke_env.environment import SinglesEnv

from src.models.singles import SinglesPolicy, save_policy
from src.observations.singles import SinglesObservation
from src.observations.vocab import ROOT, Vocab
from src.training.evaluate import Evaluator
from src.training.opponents import OpponentPool
from src.training.ppo import PPOConfig, make_batch, ppo_update
from src.training.runner import BattleResult, SelfPlayRunner
from src.utils.server import check_showdown_server


@dataclass
class TrainConfig(PPOConfig):
    run_name: str = "selfplay"
    battle_format: str = "gen9randombattle"
    updates: int = 1000
    n_envs: int = 16
    # Learner decisions from finished battles per update
    batch_size: int = 4096
    lr: float = 3e-4
    # Opponents: the rest of the battles are against past snapshots
    p_latest: float = 0.5
    p_scripted: float = 0.1
    # "uniform" (fictitious self-play) or "pfsp" (more games vs snapshots that win)
    sampling: str = "uniform"
    snapshot_every: int = 10
    max_snapshots: int = 50
    eval_every: int = 25
    eval_battles: int = 50
    shaping_weight: float = 1.0
    # Anneals the reward shaping to 0 over this many updates, 0 keeps it constant
    shaping_anneal_updates: int = 0
    max_turns: int = 300
    device: str = "cpu"
    seed: int = 0
    resume: bool = False


def parse_config() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for config_field in fields(TrainConfig):
        flag = f"--{config_field.name.replace('_', '-')}"
        default = config_field.default
        if isinstance(default, bool):
            parser.add_argument(
                flag, action=argparse.BooleanOptionalAction, default=default
            )
        else:
            parser.add_argument(flag, type=type(default), default=default)
    return TrainConfig(**vars(parser.parse_args()))


def shaping_weight(cfg: TrainConfig, update: int) -> float:
    if cfg.shaping_anneal_updates <= 0:
        return cfg.shaping_weight
    return cfg.shaping_weight * max(0.0, 1 - update / cfg.shaping_anneal_updates)


def win_rates(results: list[BattleResult]) -> dict[str, float]:
    """
    The learner's win rate against each kind of opponent, ties count as half a win.
    """
    scores: dict[str, list[float]] = defaultdict(list)
    for result in results:
        opponent = result.opponent
        name = opponent.name if opponent.kind == "scripted" else opponent.kind
        score = 1.0 if result.won else 0.5 if result.won is None else 0.0
        scores[f"win_rate_{name}"].append(score)
    return {key: float(np.mean(values)) for key, values in scores.items()}


def format_log(log: dict[str, float]) -> str:
    parts = [
        f"update {log['update']}",
        f"{log['battles']} battles",
        f"{log['decisions_per_second']:.0f} decisions/s",
    ]
    parts += [
        f"{key.removeprefix('win_rate_')} {value:.2f}"
        for key, value in log.items()
        if key.startswith("win_rate_")
    ]
    parts += [
        f"{key} {log[key]:.3f}"
        for key in ("policy_loss", "value_loss", "entropy", "approx_kl")
    ]
    parts += [
        f"eval vs {key.removeprefix('eval_')} {value:.2f}"
        for key, value in log.items()
        if key.startswith("eval_")
    ]
    return " | ".join(parts)


def main():
    cfg = parse_config()
    check_showdown_server()
    torch.manual_seed(cfg.seed)
    run_dir = ROOT / "checkpoints" / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "latest.pt"

    vocab = Vocab.load()
    gen = GenData.from_format(cfg.battle_format).gen
    policy = SinglesPolicy(vocab, SinglesEnv.get_action_space_size(gen)).to(cfg.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5)

    start = 0
    if cfg.resume:
        if not checkpoint_path.exists():
            raise SystemExit(f"Nothing to resume, {checkpoint_path} doesn't exist")
        checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
        policy.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start = checkpoint["update"]
        print(f"Resuming {cfg.run_name} after update {start}")

    pool = OpponentPool(
        run_dir / "pool",
        vocab,
        cfg.battle_format,
        p_latest=cfg.p_latest,
        p_scripted=cfg.p_scripted,
        sampling=cfg.sampling,
        max_snapshots=cfg.max_snapshots,
        device=cfg.device,
        seed=cfg.seed,
    )
    print(f"Starting {cfg.n_envs} envs on the local Showdown server...")
    runner = SelfPlayRunner(
        cfg.battle_format,
        cfg.n_envs,
        pool,
        vocab,
        device=cfg.device,
        max_turns=cfg.max_turns,
    )
    evaluator = Evaluator(policy, SinglesObservation(vocab), cfg.battle_format)

    try:
        for update in range(start + 1, cfg.updates + 1):
            runner.set_shaping_weight(shaping_weight(cfg, update))
            started = time.perf_counter()
            trajectories, results = runner.collect(policy, cfg.batch_size)
            collect_seconds = time.perf_counter() - started

            batch = make_batch(trajectories, cfg.gamma, cfg.gae_lambda)
            log: dict[str, float] = {
                "update": update,
                "battles": len(results),
                "transitions": len(batch),
                "decisions_per_second": len(batch) / collect_seconds,
                "mean_turns": float(np.mean([result.turns for result in results])),
                "shaping_weight": shaping_weight(cfg, update),
                **win_rates(results),
                **ppo_update(policy, optimizer, batch, cfg, cfg.device),
            }
            if update % cfg.snapshot_every == 0:
                pool.add_snapshot(policy, update)
                log["snapshots"] = len(pool.snapshots)
            if update % cfg.eval_every == 0:
                log.update(evaluator.run(cfg.eval_battles))

            save_policy(
                policy,
                checkpoint_path,
                optimizer=optimizer.state_dict(),
                update=update,
                config=asdict(cfg),
            )
            with open(run_dir / "log.jsonl", "a") as f:
                f.write(json.dumps(log) + "\n")
            print(format_log(log))
    finally:
        runner.close()


if __name__ == "__main__":
    main()
