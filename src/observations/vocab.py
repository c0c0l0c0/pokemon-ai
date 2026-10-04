import json
from dataclasses import dataclass
from pathlib import Path

from poke_env.data import GenData, to_id_str

ROOT = Path(__file__).resolve().parents[2]
VOCAB_PATH = ROOT / "data" / "vocab.json"
SETS_DIR = ROOT / "data" / "showdown_sets"

# Reserved ids, real entries start at N_RESERVED
PAD = 0  # empty slot
UNKNOWN = 1  # not revealed yet, or out of vocabulary
NONE = 2  # known to be empty, e.g. item knocked off
N_RESERVED = 3


@dataclass(frozen=True)
class Vocab:
    """
    Maps species, moves, items and abilities to stable integer ids for nn.Embedding.

    Frozen to data/vocab.json so ids don't shift between runs or poke-env updates.
    """

    species: dict[str, int]
    moves: dict[str, int]
    items: dict[str, int]
    abilities: dict[str, int]

    @classmethod
    def load(cls, path: Path = VOCAB_PATH) -> "Vocab":
        """
        Loads the vocab from disk, building and saving it first if it doesn't exist.
        """
        if not path.exists():
            save_vocab(build_vocab(), path)
        with open(path) as f:
            names = json.load(f)
        return cls(
            **{
                table: {name: i + N_RESERVED for i, name in enumerate(names[table])}
                for table in ("species", "moves", "items", "abilities")
            }
        )

    @property
    def n_species(self) -> int:
        return len(self.species) + N_RESERVED

    @property
    def n_moves(self) -> int:
        return len(self.moves) + N_RESERVED

    @property
    def n_items(self) -> int:
        return len(self.items) + N_RESERVED

    @property
    def n_abilities(self) -> int:
        return len(self.abilities) + N_RESERVED

    def species_id(self, species: str | None) -> int:
        return UNKNOWN if species is None else self.species.get(species, UNKNOWN)

    def move_id(self, move: str | None) -> int:
        return UNKNOWN if move is None else self.moves.get(move, UNKNOWN)

    def ability_id(self, ability: str | None) -> int:
        # poke-env uses None for both "not revealed" and "removed"
        return UNKNOWN if ability is None else self.abilities.get(ability, UNKNOWN)

    def item_id(self, item: str | None) -> int:
        # poke-env uses "unknown_item" until revealed, and None / "" once it's gone
        if item == GenData.UNKNOWN_ITEM:
            return UNKNOWN
        if not item:
            return NONE
        return self.items.get(item, UNKNOWN)


def build_vocab() -> dict[str, list[str]]:
    """
    Collects every species, move and ability from the gen 9 data (a superset of older
    gens), and every item used in data/showdown_sets, since poke-env has no item list.
    """
    data = GenData.from_gen(9)

    abilities = {
        to_id_str(ability)
        for entry in data.pokedex.values()
        for ability in entry.get("abilities", {}).values()
    }

    items = set()
    for sets_file in SETS_DIR.glob("*.txt"):
        for line in sets_file.read_text().splitlines():
            if " @ " in line:
                items.add(to_id_str(line.split(" @ ", 1)[1]))
    items.discard("")

    return {
        "species": sorted(data.pokedex),
        "moves": sorted(data.moves),
        "items": sorted(items),
        "abilities": sorted(abilities),
    }


def save_vocab(names: dict[str, list[str]], path: Path = VOCAB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(names, f, indent=1)
