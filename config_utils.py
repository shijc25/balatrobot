from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
from typing import Any, MutableMapping

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for older runtimes
    import tomli as tomllib  # type: ignore[assignment]


def load_config_file(path: str | Path | None) -> dict[str, Any]:
    """
    Load a TOML configuration file if it exists, otherwise return an empty dict.
    """
    if path is None:
        return {}

    config_path = Path(path)
    if not config_path.exists():
        return {}

    with config_path.open("rb") as handle:
        return tomllib.load(handle)


def deep_merge(base: MutableMapping[str, Any], overrides: MutableMapping[str, Any]) -> dict[str, Any]:
    """
    Recursively merge two dictionaries, returning a new dictionary.
    """
    result: dict[str, Any] = deepcopy(base)
    for key, value in overrides.items():
        if (
            isinstance(value, MutableMapping)
            and key in result
            and isinstance(result[key], MutableMapping)
        ):
            result[key] = deep_merge(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = deepcopy(value)
    return result


def apply_overrides(config: MutableMapping[str, Any], overrides: list[str]) -> MutableMapping[str, Any]:
    """
    Apply CLI-style overrides of the form section.key=value to the config dict.
    """
    if not overrides:
        return config

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Expected format key=value.")
        key_path, raw_value = override.split("=", 1)
        key_components = [component.strip() for component in key_path.split(".") if component.strip()]
        if not key_components:
            raise ValueError(f"Invalid override '{override}'. No key supplied before '='.")
        value = _coerce_value(raw_value.strip())
        _set_with_dots(config, key_components, value)
    return config


def _coerce_value(raw_value: str) -> Any:
    lowered = raw_value.lower()
    if lowered in {"none", "null", "nil"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return ast.literal_eval(raw_value)
    except (ValueError, SyntaxError):
        return raw_value


def _set_with_dots(config: MutableMapping[str, Any], keys: list[str], value: Any) -> None:
    cursor: MutableMapping[str, Any] = config
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], MutableMapping):
            cursor[key] = {}
        cursor = cursor[key]  # type: ignore[assignment]
    cursor[keys[-1]] = value


__all__ = ["load_config_file", "deep_merge", "apply_overrides"]
