"""
Plays a trained policy on the local Showdown server.

Against poke-env's scripted bots (random, max base power, heuristics):
    .venv/bin/python -m scripts.play_policy checkpoints/run1/latest.pt --battles 100

Against you: run this, then open http://localhost:8000, pick any username and
challenge "Pokemon AI Bot" (or the --name you gave it) to [Gen 9] Random Battle:
    .venv/bin/python -m scripts.play_policy checkpoints/run1/latest.pt --human --battles 3
"""

import argparse
import asyncio
from pathlib import Path

from poke_env.player import Player
from poke_env.ps_client import AccountConfiguration

from src.models.singles import load_policy
from src.observations.singles import SinglesObservation
from src.players.policy_player import PolicyPlayer
from src.training.evaluate import Evaluator
from src.utils.server import check_showdown_server
from src.wrappers.teams import team_for_format


async def wait_for_login(player: Player, seconds: float = 10) -> bool:
    for _ in range(int(seconds * 10)):
        if player.ps_client.logged_in.is_set():
            return True
        await asyncio.sleep(0.1)
    return False


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--format", default="gen9randombattle")
    parser.add_argument("--battles", type=int, default=100)
    parser.add_argument("--human", action="store_true", help="accept your challenges")
    parser.add_argument(
        "--name",
        default="Pokemon AI Bot",
        help="the bot's username with --human, at most 18 characters",
    )
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
            account_configuration=AccountConfiguration(args.name, None),
            battle_format=args.format,
            team=team_for_format(args.format),
            log_level=40,
        )
        if not asyncio.run(wait_for_login(player)):
            raise SystemExit(
                f'Couldn\'t log in as "{args.name}", the name may be taken. '
                "Pick another one with --name"
            )
        print(
            f'"{args.name}" is waiting for {args.battles} challenge(s). Open '
            f"http://localhost:8000, pick any username, and challenge it to "
            f"{args.format}."
        )
        asyncio.run(player.accept_challenges(None, args.battles))
        print(f"The bot won {player.n_won_battles}/{player.n_finished_battles}")
        return

    evaluator = Evaluator(policy, observation, args.format, greedy=args.greedy)
    for name, win_rate in evaluator.run(args.battles).items():
        print(f"vs {name}: {win_rate:.1%} wins")


if __name__ == "__main__":
    main()
