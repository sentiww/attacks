from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


def recursive_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep merge where values from override take precedence."""
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = recursive_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _read_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    return loaded


def load_config(base: str | Path, *overrides: str | Path) -> dict[str, Any]:
    """Load a base YAML config and recursively merge zero or more overrides."""
    config = _read_yaml(base)
    for path in overrides:
        config = recursive_merge(config, _read_yaml(path))
    return config


def require_sections(config: Mapping[str, Any], *sections: str) -> None:
    missing = [section for section in sections if not isinstance(config.get(section), Mapping)]
    if missing:
        raise ValueError(f"Missing config section(s): {', '.join(missing)}")


def require_keys(config: Mapping[str, Any], section: str, *keys: str) -> None:
    values = config.get(section)
    if not isinstance(values, Mapping):
        raise ValueError(f"Missing config section: {section}")
    missing = [key for key in keys if key not in values]
    if missing:
        qualified = ", ".join(f"{section}.{key}" for key in missing)
        raise ValueError(f"Missing config key(s): {qualified}")
