"""Configuration loading for project skeleton."""

from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_json(path: str) -> dict:
    """Load a JSON config file from project root."""

    config_path = PROJECT_ROOT / path
    with config_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
