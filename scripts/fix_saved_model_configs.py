#!/usr/bin/env python3
"""Rewrite saved model config.yaml paths to match current artifact locations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO_ROOT / "artifacts" / "models"
KNOWN_WEIGHT_FILES = {"model.pt", "embedder.pt", "text_encoder.pt", "circuit_encoder.pt"}


def to_repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def maybe_fix_save_path(node: dict[str, Any], config_dir: Path) -> bool:
    save_path = node.get("save_path")
    if not isinstance(save_path, str) or not save_path:
        return False

    filename = Path(save_path).name
    if filename not in KNOWN_WEIGHT_FILES:
        return False

    candidate = config_dir / filename
    if not candidate.exists():
        return False

    new_value = to_repo_relative(candidate)
    if new_value == save_path:
        return False

    node["save_path"] = new_value
    return True


def walk_and_fix(node: Any, config_dir: Path) -> int:
    updates = 0
    if isinstance(node, dict):
        updates += int(maybe_fix_save_path(node, config_dir))
        for value in node.values():
            updates += walk_and_fix(value, config_dir)
    elif isinstance(node, list):
        for value in node:
            updates += walk_and_fix(value, config_dir)
    return updates


def process_config(path: Path, apply: bool) -> int:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return 0

    updates = walk_and_fix(data, path.parent)
    if updates and apply:
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return updates


def iter_configs(root: Path) -> list[Path]:
    return sorted(root.rglob("config.yaml"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory to scan for saved model configs.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes in place. Without this flag, run in dry-run mode.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    configs = iter_configs(root)
    changed_files = 0
    changed_entries = 0

    for config_path in configs:
        updates = process_config(config_path, apply=args.apply)
        if updates:
            changed_files += 1
            changed_entries += updates
            mode = "updated" if args.apply else "would update"
            print(f"{mode}: {to_repo_relative(config_path)} ({updates} entries)")

    summary = "updated" if args.apply else "would update"
    print(f"{summary} {changed_entries} entries across {changed_files} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
