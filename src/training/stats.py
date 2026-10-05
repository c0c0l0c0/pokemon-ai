"""
Which Pokémon and moves win: counters over the learner's battles, shown as tables in
TensorBoard and saved as CSVs.

KOs are approximate. A KO goes to the last move the other side used that turn, and only
if the finishing damage came from a move, so hazard, weather, status and recoil KOs
aren't credited to anyone.
"""

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from poke_env.battle import Battle
from poke_env.data import to_id_str

from src.observations.singles import N_MOVES, N_TEAM

# Gimmick action blocks after the plain moves, in poke-env's order
GIMMICKS = ("mega", "z_move", "dynamax", "tera")

SIDES = ("p1", "p2")

# A table column: header and how to show it from an entry's counters and rates
Column = tuple[str, Callable[[dict[str, float]], str]]

GAMES: Column = ("games", lambda c: f"{c['games']:.0f}")
WIN_RATE: Column = ("win rate", lambda c: f"{c['win_rate']:.0%}")
KOS: Column = ("KOs", lambda c: f"{c['kos']:.0f}")
KOS_PER_GAME: Column = ("KOs / game", lambda c: f"{c['kos_per_game']:.2f}")
BROUGHT_IN: Column = ("brought in", lambda c: f"{c['brought_in_rate']:.0%}")
USES: Column = ("uses", lambda c: f"{c['uses']:.0f}")
KOS_PER_USE: Column = ("KOs / use", lambda c: f"{c['kos'] / max(c['uses'], 1):.2f}")


@dataclass
class BattleLog:
    """
    What each side ("p1" or "p2") did in a battle: uses and KOs per (species, move).
    """

    move_uses: dict[str, Counter[tuple[str, str]]]
    kos: dict[str, Counter[tuple[str, str]]]


def parse_battle_log(
    log: list[list[str]], species_by_ident: dict[str, str]
) -> BattleLog:
    """
    Reads Showdown protocol messages, split on "|" like poke-env does.
    species_by_ident maps team keys like "p1: Great Tusk" to species.
    """
    move_uses: dict[str, Counter[tuple[str, str]]] = {side: Counter() for side in SIDES}
    kos: dict[str, Counter[tuple[str, str]]] = {side: Counter() for side in SIDES}
    last_move: dict[str, tuple[str, str]] = {}

    for event in log:
        if len(event) < 2:
            continue
        if event[1] == "turn":
            last_move = {}
        elif event[1] == "move" and len(event) > 3 and not _has_from(event):
            # Moves with [from] were called by something else, e.g. Magic Bounce
            side = event[2][:2]
            used = (_species(event[2], species_by_ident), to_id_str(event[3]))
            move_uses[side][used] += 1
            last_move[side] = used
        elif (
            event[1] == "-damage"
            and len(event) > 3
            and event[3].endswith("fnt")
            and not _has_from(event)
        ):
            attacker = "p2" if event[2].startswith("p1") else "p1"
            if attacker in last_move:
                kos[attacker][last_move[attacker]] += 1
    return BattleLog(move_uses, kos)


def action_mix(actions: np.ndarray, n_actions: int) -> dict[str, float]:
    """
    Fraction of switches, plain moves and each gimmick the generation has.
    """
    n_gimmicks = (n_actions - N_TEAM) // N_MOVES - 1
    names = ["switch", "move", *GIMMICKS[:n_gimmicks]]
    blocks = np.where(actions < N_TEAM, 0, 1 + (actions - N_TEAM) // N_MOVES)
    counts = np.bincount(blocks, minlength=len(names))
    return {
        f"{name}_rate": float(count) / max(len(actions), 1)
        for name, count in zip(names, counts)
    }


class BattleStats:
    """
    Counters over the learner's sides of finished battles:
        species    its team: games, wins, brought in, fainted, KOs
        moves      its moves: uses, games used, wins in those games, KOs
        opponents  the other team: games, learner wins, KOs scored on the learner
    Ties (and battles cut at max_turns) count as half a win.
    """

    def __init__(self):
        self.species: dict[str, dict[str, float]] = defaultdict(_counters)
        self.moves: dict[str, dict[str, float]] = defaultdict(_counters)
        self.opponents: dict[str, dict[str, float]] = defaultdict(_counters)

    def record(
        self,
        battles: tuple[Battle, Battle],
        learner_seats: tuple[int, ...],
        won: bool | None,
    ):
        """
        battles are both seats' views of a finished battle, won is from seat 0's side.
        """
        species_by_ident = {
            key: mon.species for battle in battles for key, mon in battle.team.items()
        }
        # poke-env keeps the protocol messages for replays, with no public accessor.
        # Both seats see the same public messages, so seat 0's log is enough.
        log = parse_battle_log(
            getattr(battles[0], "_replay_data", []), species_by_ident
        )
        seat_0_score = 1.0 if won else 0.5 if won is None else 0.0

        for seat in learner_seats:
            mine, theirs = battles[seat], battles[1 - seat]
            side = mine.player_role or SIDES[seat]
            other_side = "p2" if side == "p1" else "p1"
            score = seat_0_score if seat == 0 else 1 - seat_0_score

            for mon in mine.team.values():
                counters = self.species[mon.species]
                counters["games"] += 1
                counters["wins"] += score
                counters["brought_in"] += mon.revealed
                counters["fainted"] += mon.fainted

            uses_per_move: Counter[str] = Counter()
            for (_, move), n in log.move_uses[side].items():
                uses_per_move[move] += n
            for move, n in uses_per_move.items():
                counters = self.moves[move]
                counters["uses"] += n
                counters["games"] += 1
                counters["wins"] += score
            for (species, move), n in log.kos[side].items():
                self.species[species]["kos"] += n
                self.moves[move]["kos"] += n

            for mon in theirs.team.values():
                counters = self.opponents[mon.species]
                counters["games"] += 1
                counters["wins"] += score
            for (species, _), n in log.kos[other_side].items():
                self.opponents[species]["kos"] += n

    def tables(self, top_k: int, min_games: int) -> dict[str, str]:
        """
        Markdown tables for TensorBoard's text tab.
        """
        species = _rows(self.species, min_games)
        moves = _rows(self.moves, 1)
        opponents = _rows(self.opponents, min_games)
        not_enough = f"_Not enough games yet, each one needs at least {min_games}._"

        return {
            "best_pokemon": _table(
                "Pokémon",
                species,
                sort_by="win_rate",
                columns=[GAMES, WIN_RATE, KOS_PER_GAME, BROUGHT_IN],
                top_k=top_k,
                empty=not_enough,
            ),
            "most_kos": _table(
                "Pokémon",
                species,
                sort_by="kos_per_game",
                columns=[GAMES, KOS, KOS_PER_GAME, WIN_RATE],
                top_k=top_k,
                empty=not_enough,
            ),
            "most_used_moves": _table(
                "move",
                moves,
                sort_by="uses",
                columns=[USES, GAMES, ("win rate when used", WIN_RATE[1]), KOS],
                top_k=top_k,
                empty="_No moves used yet._",
            ),
            "best_moves_by_kos": _table(
                "move",
                moves,
                sort_by="kos",
                columns=[KOS, USES, KOS_PER_USE],
                top_k=top_k,
                empty="_No KOs yet._",
            ),
            "toughest_opponents": _table(
                "opponent",
                opponents,
                sort_by="win_rate",
                columns=[
                    GAMES,
                    ("learner win rate", WIN_RATE[1]),
                    ("KOs on the learner", KOS[1]),
                ],
                top_k=top_k,
                empty=not_enough,
                ascending=True,
            ),
        }

    def write_csv(self, directory: Path):
        """
        Every counter and rate, e.g. for pandas: species.csv, moves.csv, opponents.csv.
        """
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("species", "moves", "opponents"):
            rows = _rows(getattr(self, name), 1)
            columns = sorted({column for _, row in rows for column in row})
            with open(directory / f"{name}.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([name, *columns])
                for key, row in sorted(rows, key=lambda r: -r[1]["games"]):
                    writer.writerow([key, *(round(row[c], 4) for c in columns)])

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        tables = {
            name: getattr(self, name) for name in ("species", "moves", "opponents")
        }
        # Written next to it first, so stopping mid-save can't leave a corrupt file
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(tables))
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> "BattleStats":
        stats = cls()
        for name, saved in json.loads(path.read_text()).items():
            table = getattr(stats, name)
            for key, counters in saved.items():
                table[key].update(counters)
        return stats


def _counters() -> dict[str, float]:
    return defaultdict(float)


def _rows(
    table: dict[str, dict[str, float]], min_games: int
) -> list[tuple[str, dict[str, float]]]:
    """
    Entries with at least min_games games, with their rates. Missing counters are 0.
    """
    rows = []
    for key, counters in table.items():
        games = counters.get("games", 0)
        if games < max(min_games, 1):
            continue
        row = defaultdict(float, counters)
        row["win_rate"] = row["wins"] / games
        row["kos_per_game"] = row["kos"] / games
        # Only the learner's own Pokémon have it
        if "brought_in" in counters:
            row["brought_in_rate"] = counters["brought_in"] / games
        rows.append((key, row))
    return rows


def _table(
    name: str,
    rows: list[tuple[str, dict[str, float]]],
    *,
    sort_by: str,
    columns: list[Column],
    top_k: int,
    empty: str,
    ascending: bool = False,
) -> str:
    if not rows:
        return empty
    rows = sorted(rows, key=lambda r: r[1][sort_by], reverse=not ascending)[:top_k]
    lines = [
        "| " + " | ".join([name, *(header for header, _ in columns)]) + " |",
        "|" + "---|" * (len(columns) + 1),
    ]
    for key, row in rows:
        cells = [key, *(show(row) for _, show in columns)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _species(ident: str, species_by_ident: dict[str, str]) -> str:
    # "p1a: Great Tusk" (with the field position) -> "p1: Great Tusk"
    key = ident[:2] + ident[3:] if len(ident) > 3 and ident[2] != ":" else ident
    return species_by_ident.get(key, to_id_str(ident.split(": ", 1)[-1]))


def _has_from(event: list[str]) -> bool:
    return any(part.startswith("[from]") for part in event[3:])
