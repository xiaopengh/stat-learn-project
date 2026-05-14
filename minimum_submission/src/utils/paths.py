from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Return the repository root by walking upward until pyproject.toml is found."""
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate

    msg = f"Could not find project root from {current}"
    raise FileNotFoundError(msg)


PROJECT_ROOT = find_project_root()


def project_path(*parts: str | Path) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def resolve_project_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return PROJECT_ROOT / path_obj


def ensure_dir(path: str | Path) -> Path:
    path_obj = resolve_project_path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj
