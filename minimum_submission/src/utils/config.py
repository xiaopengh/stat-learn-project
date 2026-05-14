from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import project_path, resolve_project_path

DEFAULT_CONFIG_PATH = project_path("configs", "default.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_path = resolve_project_path(path)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        msg = f"Expected mapping in config file {config_path}"
        raise TypeError(msg)
    return config


def get_config_value(config: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for key in dotted_key.split("."):
        if not isinstance(current, Mapping) or key not in current:
            msg = f"Missing config key: {dotted_key}"
            raise KeyError(msg)
        current = current[key]
    return current


def get_path(config: Mapping[str, Any], key: str) -> Path:
    paths = get_config_value(config, "paths")
    if key not in paths:
        msg = f"Missing path config key: {key}"
        raise KeyError(msg)
    return resolve_project_path(paths[key])
