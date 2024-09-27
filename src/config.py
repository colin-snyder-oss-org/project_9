# src/config.py
import yaml
import os
from typing import Dict, Any
from collections.abc import Mapping

class Config(Mapping):
    def __init__(self, config_file: str):
        self._config = self._load_config(config_file)

    def _load_config(self, config_file: str) -> Dict[str, Any]:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def __getitem__(self, key):
        return self._config[key]

    def __iter__(self):
        return iter(self._config)

    def __len__(self):
        return len(self._config)
