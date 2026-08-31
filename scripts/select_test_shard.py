#!/usr/bin/env python3
"""Select deterministic pytest file shards using measured duration weights."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_FILE_SECONDS = 1.0

# Current Core v2 hotspot measurements. Only material outliers are pinned; new
# and ordinary files receive DEFAULT_FILE_SECONDS. The LPT scheduler is
# deterministic, so every retained current-v2 file is assigned exactly once.
HISTORICAL_FILE_SECONDS = {
    "tests/molly/test_core02_agent_loop.py": 8.0,
    "tests/molly/test_core03_acquisition.py": 8.0,
    "tests/molly/test_core04_documents.py": 8.0,
    "tests/molly/test_core05_oled_evidence.py": 8.0,
    "tests/molly/test_core06_br1_plugin.py": 5.0,
    "tests/molly/test_core06_remote_compute.py": 5.0,
    "tests/molly/test_core07_cli.py": 4.0,
    "tests/molly/test_core07_inspection.py": 4.0,
    "tests/molly/test_core07_observability.py": 4.0,
    "tests/molly/test_core08_cutover.py": 4.0,
}


def discover_test_files(tests_root: Path = Path("tests")) -> list[str]:
    repository_root = Path.cwd().resolve()
    discovered: list[str] = []
    for path in tests_root.rglob("test_*.py"):
        if not path.is_file():
            continue
        resolved = path.resolve()
        try:
            discovered.append(resolved.relative_to(repository_root).as_posix())
        except ValueError:
            discovered.append(resolved.as_posix())
    return sorted(discovered)


def file_weight(path: str) -> float:
    return HISTORICAL_FILE_SECONDS.get(path, DEFAULT_FILE_SECONDS)


def assign_test_files(files: Iterable[str], shard_count: int) -> tuple[list[list[str]], list[float]]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    unique_files = sorted(set(files))
    shards: list[list[str]] = [[] for _ in range(shard_count)]
    totals = [0.0] * shard_count
    for path in sorted(unique_files, key=lambda item: (-file_weight(item), item)):
        shard_index = min(range(shard_count), key=lambda index: (totals[index], index))
        shards[shard_index].append(path)
        totals[shard_index] += file_weight(path)
    for shard in shards:
        shard.sort()
    return shards, totals


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests-root", type=Path, default=Path("tests"))
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--shard", type=int)
    parser.add_argument(
        "--validate",
        action="store_true",
        help="validate complete, non-overlapping assignment before printing",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    files = discover_test_files(args.tests_root)
    shards, _totals = assign_test_files(files, args.shards)
    flattened = [path for shard in shards for path in shard]
    if args.validate and (len(flattened) != len(set(flattened)) or sorted(flattened) != files):
        raise SystemExit("test shard assignment is incomplete or overlapping")
    if args.shard is None:
        if not args.validate:
            raise SystemExit("--shard is required unless --validate is used")
        return 0
    if args.shard < 0 or args.shard >= args.shards:
        raise SystemExit(f"--shard must be between 0 and {args.shards - 1}")
    for path in shards[args.shard]:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
