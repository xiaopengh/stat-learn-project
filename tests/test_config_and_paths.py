from __future__ import annotations

from src.utils import PROJECT_ROOT, get_path, load_config


def test_load_config_returns_mapping() -> None:
    config = load_config()
    assert isinstance(config, dict)
    assert config["dataset"]["target"] == "critical_temp"


def test_get_path_resolves_project_relative_paths() -> None:
    config = load_config()
    tables_path = get_path(config, "tables")
    assert tables_path == PROJECT_ROOT / "reports" / "tables"
    assert tables_path.is_absolute()
