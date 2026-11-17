"""
config.py — Load settings from settings.toml
===============================================
Usage:
    from config import settings
    print(settings.llm.api_url)
"""

import os
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # Python 3.9-3.10 (pip install tomli)
    except ImportError:
        # Fallback: manual TOML parser for basic needs
        tomllib = None

SETTINGS_FILE = Path(__file__).parent / "settings.toml"


def _load_settings_fallback(path: Path) -> dict:
    """Minimal TOML fallback for flat/nested key-value pairs."""
    import re
    result = {}
    current_section = result
    current_path = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            section_match = re.match(r"^\[(.+)\]$", line)
            if section_match:
                parts = section_match.group(1).split(".")
                current_section = result
                current_path = parts
                for p in parts:
                    if p not in current_section:
                        current_section[p] = {}
                    current_section = current_section[p]
                continue

            kv_match = re.match(r'^(\w+)\s*=\s*(.+)$', line)
            if kv_match:
                key = kv_match.group(1)
                value = kv_match.group(2).strip()
                value = _parse_toml_value(value)
                current_section[key] = value

    return result


def _parse_toml_value(value: str):
    """Parse a TOML value string into Python type."""
    import re
    # String
    if (value.startswith('"') and value.endswith('"')) or \
       (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    # Boolean
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    # Null
    if value.lower() in ("null", "none"):
        return None
    # Integer
    try:
        return int(value)
    except ValueError:
        pass
    # Float
    try:
        return float(value)
    except ValueError:
        pass
    return value


class Settings:
    """Dot-access wrapper for nested dict from TOML."""
    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, name):
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        try:
            val = self._data[name]
        except KeyError:
            raise AttributeError(f"Setting '{name}' not found")
        if isinstance(val, dict):
            return Settings(val)
        return val

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def values(self):
        return self._data.values()

    def get(self, key, default=None):
        val = self._data.get(key, default)
        if isinstance(val, dict):
            return Settings(val)
        return val

    def __repr__(self):
        return f"Settings({self._data})"

    def __iter__(self):
        return iter(self._data)


def load_settings(path: Path | str | None = None) -> Settings:
    """Load settings.toml and return a dot-access Settings object."""
    path = Path(path) if path else SETTINGS_FILE

    if not path.exists():
        raise FileNotFoundError(f"Settings file not found: {path}")

    if tomllib is not None:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    else:
        data = _load_settings_fallback(path)

    return Settings(data)


# Singleton — import and use directly
settings = load_settings()
