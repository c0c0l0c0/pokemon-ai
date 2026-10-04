"""
Plays a trained policy on the local Showdown server.

Against poke-env's scripted bots (random, max base power, heuristics):
    .venv/bin/python -m scripts.play_policy checkpoints/run1/latest.pt --battles 100

Against you: open http://localhost:8000, pick any username and challenge the bot:
    .venv/bin/python -m scripts.play_policy checkpoints/run1/latest.pt --human --battles 3
"""

import argparse
import asyncio
from pathlib import Path

from src.models.singles import load_policy
from src.observations.singles import SinglesObservation
from src.players.policy_player import PolicyPlayer
from src.training.evaluate import Evaluator
from src.utils.server import check_showdown_server
from src.wrappers.teams import team_for_format


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--format", default="gen9randombattle")
    parser.add_argument("--battles", type=int, default=100)
    parser.add_argument("--human", action="store_true", help="accept your challenges")
    parser.add_argument(
        "--greedy", action="store_true", help="always pick the top move"
    )
    args = parser.parse_args()
    check_showdown_server()

    policy = load_policy(args.checkpoint)
    observation = SinglesObservation()

    if args.human:
        player = PolicyPlayer(
            policy,
            observation,
            greedy=args.greedy,
            battle_format=args.format,
            team=team_for_format(args.format),
            log_level=40,
        )
        print(
            f'Challenge "{player.username}" to {args.format} on http://localhost:8000'
        )
        asyncio.run(player.accept_challenges(None, args.battles))
        print(f"The bot won {player.n_won_battles}/{player.n_finished_battles}")
        return

    evaluator = Evaluator(policy, observation, args.format, greedy=args.greedy)
    for name, win_rate in evaluator.run(args.battles).items():
        print(f"{name.removeprefix('eval_')}: {win_rate:.1%} wins")


if __name__ == "__main__":
    main()
