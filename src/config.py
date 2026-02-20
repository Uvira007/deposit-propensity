"""
Config loader to load the configuration from a YAML file for the pipeline
"""

from pathlib import Path
from typing import Any

import yaml

def get_project_root() -> Path:
    """Project Root -> Parent of src"""
    return Path(__file__).resolve().parent.parent

def load_config(config_path: Path | None = None):
    """
    Load config from YAML and update the relative paths to absolute paths
    """
    root = get_project_root()
    config_path = config_path or root / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # Update relative paths to absolute paths
    for key in ("data_dir", "model_dir", "output_dir", "plots_dir", "metrics_dir", "dashboard_dir"):
        if key in config.get("paths", {}):
            config["paths"][key] = root / config["paths"][key]
    return config


def get_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Extract paths from config and return as dict of Path objects"""
    paths = config.get("paths", {})
    return {key: Path(value) for key, value in paths.items()}

if __name__ == "__main__":
    cfg = load_config()
    print(cfg)