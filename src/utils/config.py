"""YAML configuration loading, merging, and saving utilities."""

import copy
from pathlib import Path
from typing import Any, Dict, Union

import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML configuration file into a dictionary.

    Args:
        config_path: Path to the target YAML file.

    Returns:
        Dictionary representation of configuration.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        ValueError: If file is not valid YAML or does not parse into a dictionary.
    """
    path = Path(config_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"Configuration file must have a .yaml or .yml extension: {path.name}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"Error parsing YAML file {path}: {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root in {path} must be a dictionary, got {type(data).__name__}")

    return data


def merge_configs(base_cfg: Dict[str, Any], override_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override dictionary into a deep copy of base dictionary.

    Args:
        base_cfg: Baseline configuration dictionary.
        override_cfg: Overriding configuration dictionary.

    Returns:
        A new merged dictionary.
    """
    merged = copy.deepcopy(base_cfg)

    for key, value in override_cfg.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)

    return merged


def save_config(config: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Save a dictionary to a YAML configuration file.

    Args:
        config: Configuration dictionary to save.
        output_path: Target output file path.
    """
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)

