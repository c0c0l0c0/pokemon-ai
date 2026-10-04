"""
Rebuilds data/vocab.json (species, moves, items, abilities -> embedding ids).

Changing the vocab changes the ids, so models trained with the old one won't load.

Run from the project root:
    .venv/bin/python -m scripts.build_vocab
"""

from src.observations.vocab import VOCAB_PATH, build_vocab, save_vocab


def main():
    names = build_vocab()
    save_vocab(names)
    sizes = ", ".join(f"{table}={len(entries)}" for table, entries in names.items())
    print(f"Wrote {VOCAB_PATH}: {sizes}")


if __name__ == "__main__":
    main()
