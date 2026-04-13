#!/usr/bin/env python3
"""Audit thesis experiment bookkeeping from experiments/registry.yaml."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO_ROOT / "experiments" / "registry.yaml"


def flatten_paths(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            items.extend(flatten_paths(item))
        return items
    if isinstance(value, dict):
        items: list[str] = []
        for nested in value.values():
            items.extend(flatten_paths(nested))
        return items
    return []


def resolve_existing(paths: Iterable[str]) -> tuple[list[str], list[str]]:
    existing: list[str] = []
    missing: list[str] = []
    for raw_path in paths:
        target = REPO_ROOT / raw_path
        if target.exists():
            existing.append(raw_path)
        else:
            missing.append(raw_path)
    return existing, missing


def load_registry() -> dict:
    with REGISTRY_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def format_text(registry: dict) -> str:
    lines = []
    for experiment in registry.get("experiments", []):
        config_paths = flatten_paths(experiment.get("configs", {}))
        evidence_paths = flatten_paths(experiment.get("evidence", {}))
        artifact_paths = flatten_paths(experiment.get("expected_artifacts", {}))

        existing_configs, missing_configs = resolve_existing(config_paths)
        existing_evidence, missing_evidence = resolve_existing(evidence_paths)
        existing_artifacts, missing_artifacts = resolve_existing(artifact_paths)

        lines.append(f"[{experiment['id']}] {experiment['title']}")
        lines.append(f"  Declared status: {experiment['status']['overall']}")
        lines.append(f"  Thesis goal: {experiment['thesis_goal']}")
        lines.append(
            "  Configs: "
            f"{len(existing_configs)}/{len(config_paths)} present"
        )
        lines.append(
            "  Evidence: "
            f"{len(existing_evidence)}/{len(evidence_paths)} present"
        )
        lines.append(
            "  Artifacts: "
            f"{len(existing_artifacts)}/{len(artifact_paths)} present"
        )

        if missing_configs:
            lines.append("  Missing configs:")
            for path in missing_configs:
                lines.append(f"    - {path}")
        if missing_evidence:
            lines.append("  Missing evidence:")
            for path in missing_evidence:
                lines.append(f"    - {path}")
        if missing_artifacts:
            lines.append("  Missing artifacts:")
            for path in missing_artifacts:
                lines.append(f"    - {path}")

        note = experiment["status"].get("notes")
        if note:
            lines.append(f"  Notes: {note}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_markdown(registry: dict) -> str:
    lines = [
        "| Experiment | Declared Status | Configs | Evidence | Artifacts |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for experiment in registry.get("experiments", []):
        config_paths = flatten_paths(experiment.get("configs", {}))
        evidence_paths = flatten_paths(experiment.get("evidence", {}))
        artifact_paths = flatten_paths(experiment.get("expected_artifacts", {}))

        existing_configs, _ = resolve_existing(config_paths)
        existing_evidence, _ = resolve_existing(evidence_paths)
        existing_artifacts, _ = resolve_existing(artifact_paths)

        lines.append(
            f"| `{experiment['id']}` | {experiment['status']['overall']} | "
            f"{len(existing_configs)}/{len(config_paths)} | "
            f"{len(existing_evidence)}/{len(evidence_paths)} | "
            f"{len(existing_artifacts)}/{len(artifact_paths)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=("text", "markdown"),
        default="text",
        help="Output format.",
    )
    args = parser.parse_args()

    registry = load_registry()
    if args.format == "markdown":
        print(format_markdown(registry), end="")
    else:
        print(format_text(registry), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
