from __future__ import annotations

import os
import sys
from pathlib import Path


def find_project_root(start: str | Path | None = None) -> Path:
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "src").exists() and (candidate / "conf").exists():
            return candidate
    raise FileNotFoundError("Could not locate repo root containing src/ and conf/.")


def setup_notebook_paths(start: str | Path | None = None) -> Path:
    project_root = find_project_root(start)
    os.chdir(project_root)

    additions = [
        project_root,
        project_root / "src",
        project_root / "notebooks",
    ]
    for path in additions:
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    return project_root
