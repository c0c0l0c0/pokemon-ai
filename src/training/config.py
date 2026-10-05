"""
Training settings. configs/selfplay.toml documents every option.

Later sources override earlier ones:
    1. the defaults below
    2. the run's saved config.json, when resuming (--resume)
    3. a TOML file: --config path.toml, or configs/selfplay.toml for new runs
    4. flags passed on the command line, e.g. --batch-size 2048
"""

import argparse
import json
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from src.observations.vocab import ROOT
from src.training.ppo import PPOConfig

CHECKPOINTS_DIR = ROOT / "checkpoints"
DEFAULT_CONFIG = ROOT / "configs" / "selfplay.toml"


@dataclass
class TrainConfig(PPOConfig):
    # Run
    run_name: str = "selfplay"
    battle_format: str = "gen9randombattle"
    updates: int = 1000
    n_envs: int = 16
    batch_size: int = 4096
    max_turns: int = 300
    device: str = "cpu"
    seed: int = 0
    resume: bool = False
    lr: float = 3e-4

    # Opponents
    p_latest: float = 0.5
    p_scripted: float = 0.1
    sampling: str = "uniform"
    snapshot_every: int = 10
    max_snapshots: int = 50

    # Reward, the original ratios (2, 1 and 0.5 for a win worth 30) with a win = 1
    fainted_value: float = 2 / 30
    hp_value: float = 1 / 30
    status_value: float = 0.5 / 30
    shaping_weight: float = 1.0
    shaping_anneal_updates: int = 0

    # Model
    d_model: int = 128
    d_move: int = 64
    n_layers: int = 2
    n_heads: int = 4

    # Evaluation against the scripted bots
    eval_every: int = 25
    eval_battles: int = 50
    eval_concurrency: int = 10
    eval_greedy: bool = False

    # Logging and Pokémon / move stats
    tensorboard: bool = True
    stats_every: int = 10
    stats_top_k: int = 15
    stats_min_games: int = 20

    @property
    def run_dir(self) -> Path:
        return CHECKPOINTS_DIR / self.run_name


def load_config(argv: list[str] | None = None) -> TrainConfig:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        description="Self-play PPO training. Every setting is documented in "
        "configs/selfplay.toml, and any of them can be overridden with a flag.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="TOML settings file (new runs use configs/selfplay.toml by default)",
    )
    for config_field in fields(TrainConfig):
        default = getattr(defaults, config_field.name)
        flag = "--" + config_field.name.replace("_", "-")
        # SUPPRESS leaves flags that weren't passed out of the parsed values
        if isinstance(default, bool):
            parser.add_argument(
                flag,
                action=argparse.BooleanOptionalAction,
                default=argparse.SUPPRESS,
                help=f"default: {default}",
            )
        else:
            parser.add_argument(
                flag,
                type=type(default),
                default=argparse.SUPPRESS,
                help=f"default: {default}",
            )

    from_cli = vars(parser.parse_args(argv))
    config_file: Path | None = from_cli.pop("config")
    resume = from_cli.get("resume", False)
    # A resumed run keeps its own settings, unless a file is given explicitly
    if config_file is None and not resume and DEFAULT_CONFIG.exists():
        config_file = DEFAULT_CONFIG
    from_file = read_toml(config_file) if config_file is not None else {}

    values = asdict(defaults)
    if resume:
        run_name = from_cli.get(
            "run_name", from_file.get("run_name", defaults.run_name)
        )
        saved = CHECKPOINTS_DIR / run_name / "config.json"
        if saved.exists():
            saved_values = json.loads(saved.read_text())
            values.update({k: v for k, v in saved_values.items() if k in values})
    values.update(from_file)
    values.update(from_cli)
    values["resume"] = resume
    return TrainConfig(**_check_types(values))


def read_toml(path: Path) -> dict[str, Any]:
    """
    Reads a settings file. Its [sections] are only for readability, so they're flattened.
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)

    values: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            values.update(value)
        else:
            values[key] = value

    valid = {config_field.name for config_field in fields(TrainConfig)}
    unknown = sorted(set(values) - valid)
    if unknown:
        raise SystemExit(
            f"Unknown settings in {path}: {', '.join(unknown)}.\n"
            f"Valid ones: {', '.join(sorted(valid))}"
        )
    return values


def save_config(cfg: TrainConfig, path: Path):
    path.write_text(json.dumps(asdict(cfg), indent=2) + "\n")


def config_markdown(cfg: TrainConfig) -> str:
    rows = [f"| {key} | {value} |" for key, value in asdict(cfg).items()]
    return "\n".join(["| setting | value |", "|---|---|", *rows])


def _check_types(values: dict[str, Any]) -> dict[str, Any]:
    defaults = asdict(TrainConfig())
    for key, value in values.items():
        expected = type(defaults[key])
        if expected is float and type(value) is int:
            values[key] = float(value)
        elif type(value) is not expected:
            raise SystemExit(f"{key} should be a {expected.__name__}, got {value!r}")
    return values
