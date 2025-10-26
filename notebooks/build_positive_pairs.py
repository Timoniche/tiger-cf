"""Generate a dataset of positive item interaction pairs.

This script reads user interaction histories from a JSON file and produces
adjacent item pairs using a sliding window of configurable length. The result
is saved to a plain text file where each line contains a pair of item IDs
separated by a single space.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple


History = Sequence[int]
Interactions = Dict[str, History]
Pair = Tuple[int, int]


def load_interactions(path: Path) -> Interactions:
    """Load interaction histories from *path*.

    The expected format is a JSON object that maps user identifiers to lists of
    item identifiers. User identifiers are left as strings to avoid accidental
    loss of leading zeros or other formatting.
    """

    with path.open("r", encoding="utf-8") as source:
        data = json.load(source)

    if not isinstance(data, dict):
        raise ValueError("Interactions JSON must be an object mapping user IDs to histories")

    # Validate that every value is a list of integers. While we do not modify the
    # data in place, we raise a helpful error if the structure is unexpected.
    for user_id, history in data.items():
        if not isinstance(history, list):
            raise ValueError(f"History for user {user_id!r} is not a list")
        if not all(isinstance(item_id, int) for item_id in history):
            raise ValueError(f"History for user {user_id!r} must contain only integers")

    return data


def positive_pairs(history: History, window_size: int = 2) -> Iterator[Pair]:
    """Yield sliding window pairs from *history*.

    Parameters
    ----------
    history:
        An ordered sequence of item identifiers.
    window_size:
        Size of the sliding window. For example, ``window_size=2`` yields
        adjacent pairs.
    """

    if window_size < 2:
        raise ValueError("Window size must be at least 2 to form a pair")

    if len(history) < window_size:
        return

    for start in range(len(history) - window_size + 1):
        window = history[start : start + window_size]
        yield window[0], window[-1]


def build_dataset(interactions: Interactions, window_size: int = 2) -> List[Pair]:
    """Create a list of positive pairs for every user in *interactions*."""

    pairs: List[Pair] = []
    for history in interactions.values():
        pairs.extend(positive_pairs(history, window_size=window_size))
    return pairs


def write_pairs(pairs: Iterable[Pair], output_path: Path) -> None:
    """Write *pairs* to *output_path*, one pair per line."""

    with output_path.open("w", encoding="utf-8") as destination:
        for first, second in pairs:
            destination.write(f"{first} {second}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "interactions_path",
        type=Path,
        help="Path to the JSON file with user-item interaction histories.",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Where to save the generated positive pairs (text file).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=2,
        help="Sliding window size. Defaults to 2 for adjacent pairs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    interactions = load_interactions(args.interactions_path)
    pairs = build_dataset(interactions, window_size=args.window_size)

    write_pairs(pairs, args.output_path)

    print(f"Processed {len(interactions)} users and wrote {len(pairs)} pairs to {args.output_path}")


if __name__ == "__main__":
    main()
