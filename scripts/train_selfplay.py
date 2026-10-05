"""
Self-play PPO training for SinglesPolicy.

Needs a local Showdown server: node pokemon-showdown start --no-security

Settings come from configs/selfplay.toml, where every option is documented, and any of
them can be overridden with a flag. Run from the project root:
    .venv/bin/python -m scripts.train_selfplay --run-name run1
    .venv/bin/python -m scripts.train_selfplay --run-name run1 --resume
    .venv/bin/python -m scripts.train_selfplay --config configs/other.toml --batch-size 2048
    .venv/bin/python -m scripts.train_selfplay --help

Writes to checkpoints/<run-name>/: latest.pt, config.json, log.jsonl, tensorboard/,
stats/ (Pokémon and move tables) and pool/ (past snapshots). Watch it with:
    .venv/bin/tensorboard --logdir checkpoints

To stop after the current update, with everything saved, create a STOP file:
    touch checkpoints/<run-name>/STOP
"""

import time
from collections import defaultdict

import numpy as np
import torch
from poke_env.data import GenData
from poke_env.environment import SinglesEnv

from src.models.singles import SinglesPolicy, save_policy
from src.observations.singles import SinglesObservation
from src.observations.vocab import ROOT, Vocab
from src.training.config import TrainConfig, config_markdown, load_config, save_config
from src.training.evaluate import Evaluator
from src.training.logger import RunLogger
from src.training.opponents import OpponentPool
from src.training.ppo import make_batch, ppo_update
from src.training.runner import BattleResult, SelfPlayRunner
from src.training.stats import BattleStats, action_mix
from src.utils.server import check_showdown_server


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
        scores[name].append(1.0 if result.won else 0.5 if result.won is None else 0.0)
    return {name: float(np.mean(values)) for name, values in scores.items()}


def format_log(update: int, metrics: dict[str, float]) -> str:
    total_battles = metrics["matches/total_battles"]
    parts = [
        f"update {update}",
        f"{metrics['rollout/battles']:.0f} battles ({total_battles:.0f} total)",
        f"{metrics['time/total_hours']:.1f} h",
        f"{metrics['rollout/battle_seconds']:.0f} s/battle",
        f"{metrics['rollout/battles_per_minute']:.0f} battles/min",
        f"{metrics['rollout/decisions_per_second']:.0f} decisions/s",
    ]
    parts += [
        f"{tag.removeprefix('win_rate/')} {value:.2f}"
        for tag, value in metrics.items()
        if tag.startswith("win_rate/")
    ]
    parts += [
        f"{tag.split('/')[1]} {metrics[tag]:.3f}"
        for tag in ("loss/policy", "loss/value", "loss/entropy", "ppo/approx_kl")
    ]
    parts += [
        f"eval {tag.removeprefix('eval/')} {value:.2f}"
        for tag, value in metrics.items()
        if tag.startswith("eval/")
    ]
    return " | ".join(parts)


def main():
    cfg = load_config()
    check_showdown_server()
    torch.manual_seed(cfg.seed)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = cfg.run_dir / "latest.pt"
    stats_dir = cfg.run_dir / "stats"

    vocab = Vocab.load()
    n_actions = SinglesEnv.get_action_space_size(
        GenData.from_format(cfg.battle_format).gen
    )
    policy = SinglesPolicy(
        vocab,
        n_actions,
        d_model=cfg.d_model,
        d_move=cfg.d_move,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
    ).to(cfg.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5)
    stats = BattleStats()
    # Since the start of the run: battles and wins (seat 0), learner decisions trained
    # on, and hours of training
    totals = {"battles": 0, "wins": 0, "decisions": 0, "hours": 0.0}

    start = 0
    if cfg.resume:
        if not checkpoint_path.exists():
            raise SystemExit(f"Nothing to resume, {checkpoint_path} doesn't exist")
        checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
        policy.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # The saved optimizer state has the old learning rate, --lr should win
        for group in optimizer.param_groups:
            group["lr"] = cfg.lr
        start = checkpoint["update"]
        totals.update(checkpoint.get("totals", {}))
        if (stats_dir / "state.json").exists():
            stats = BattleStats.load(stats_dir / "state.json")
        print(f"Resuming {cfg.run_name} after update {start}")

    save_config(cfg, cfg.run_dir / "config.json")
    logger = RunLogger(cfg.run_dir, tensorboard=cfg.tensorboard)
    logger.text("config", config_markdown(cfg), start)
    # Left over from a previous stop
    stop_file = cfg.run_dir / "STOP"
    stop_file.unlink(missing_ok=True)

    pool = OpponentPool(
        cfg.run_dir / "pool",
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
    print(f"To stop after the current update: touch {stop_file.relative_to(ROOT)}")
    runner = SelfPlayRunner(
        cfg.battle_format,
        cfg.n_envs,
        pool,
        vocab,
        device=cfg.device,
        max_turns=cfg.max_turns,
        env_kwargs={
            "fainted_value": cfg.fainted_value,
            "hp_value": cfg.hp_value,
            "status_value": cfg.status_value,
        },
    )
    evaluator = Evaluator(
        policy,
        SinglesObservation(vocab),
        cfg.battle_format,
        greedy=cfg.eval_greedy,
        concurrent_battles=cfg.eval_concurrency,
    )

    try:
        for update in range(start + 1, cfg.updates + 1):
            update_started = time.perf_counter()
            weight = shaping_weight(cfg, update)
            runner.set_shaping_weight(weight)

            started = time.perf_counter()
            trajectories, results = runner.collect(policy, cfg.batch_size)
            collect_seconds = time.perf_counter() - started

            started = time.perf_counter()
            batch = make_batch(trajectories, cfg.gamma, cfg.gae_lambda)
            ppo_stats = ppo_update(policy, optimizer, batch, cfg, cfg.device)
            update_seconds = time.perf_counter() - started

            for result in results:
                stats.record(result.battles, result.learner_seats, result.won)
            wins = sum(result.won is True for result in results)
            losses = sum(result.won is False for result in results)
            totals["battles"] += len(results)
            totals["wins"] += wins
            totals["decisions"] += len(batch)

            metrics: dict[str, float] = {
                "matches/wins": wins,
                "matches/losses": losses,
                "matches/ties": len(results) - wins - losses,
                "matches/total_battles": totals["battles"],
                "matches/total_wins": totals["wins"],
                "matches/total_decisions": totals["decisions"],
                **{
                    f"win_rate/vs_{name}": rate
                    for name, rate in win_rates(results).items()
                },
                "loss/policy": ppo_stats["policy_loss"],
                "loss/value": ppo_stats["value_loss"],
                "loss/entropy": ppo_stats["entropy"],
                "ppo/approx_kl": ppo_stats["approx_kl"],
                "ppo/clip_fraction": ppo_stats["clip_fraction"],
                "ppo/explained_variance": ppo_stats["explained_variance"],
                "ppo/learning_rate": optimizer.param_groups[0]["lr"],
                **{
                    f"policy/{name}": rate
                    for name, rate in action_mix(batch.actions, n_actions).items()
                },
                "rollout/battles": len(results),
                "rollout/battles_per_minute": len(results) / collect_seconds * 60,
                # Wall-clock time per battle, with n_envs of them played at once
                "rollout/battle_seconds": float(np.mean([r.seconds for r in results])),
                "rollout/mean_turns": float(np.mean([r.turns for r in results])),
                "rollout/decisions": len(batch),
                "rollout/decisions_per_second": len(batch) / collect_seconds,
                "rollout/shaping_weight": weight,
                "time/collect_seconds": collect_seconds,
                "time/update_seconds": update_seconds,
            }
            # One bin per action index
            action_bins = np.arange(n_actions + 1) - 0.5
            logger.histogram("policy/actions", batch.actions, update, bins=action_bins)
            logger.histogram("value/predicted", batch.values, update)
            logger.histogram("value/returns", batch.returns, update)

            if update % cfg.snapshot_every == 0:
                pool.add_snapshot(policy, update)
            metrics["pool/snapshots"] = len(pool.snapshots)

            if update % cfg.eval_every == 0:
                started = time.perf_counter()
                for name, rate in evaluator.run(cfg.eval_battles).items():
                    metrics[f"eval/vs_{name}"] = rate
                metrics["time/eval_seconds"] = time.perf_counter() - started

            if update % cfg.stats_every == 0:
                tables = stats.tables(cfg.stats_top_k, cfg.stats_min_games)
                for title, table in tables.items():
                    logger.text(f"stats/{title}", table, update)
                stats.write_csv(stats_dir)

            totals["hours"] += (time.perf_counter() - update_started) / 3600
            metrics["time/total_hours"] = totals["hours"]
            save_policy(
                policy,
                checkpoint_path,
                optimizer=optimizer.state_dict(),
                update=update,
                totals=totals,
            )
            stats.save(stats_dir / "state.json")
            logger.scalars(update, metrics)
            print(format_log(update, metrics))

            if stop_file.exists():
                stop_file.unlink()
                print(f"Stopped after update {update}, continue it with --resume")
                break
    finally:
        logger.close()
        runner.close()


if __name__ == "__main__":
    main()
