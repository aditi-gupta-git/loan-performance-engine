"""Configuration management for Loan Performance Intelligence Engine."""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class Settings:
    """Application settings loaded from YAML config."""
    data: Dict[str, Any] = field(default_factory=dict)
    synthetic: Dict[str, Any] = field(default_factory=dict)
    split: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, Any] = field(default_factory=dict)
    modeling: Dict[str, Any] = field(default_factory=dict)
    anomaly: Dict[str, Any] = field(default_factory=dict)
    scenarios: Dict[str, Any] = field(default_factory=dict)
    llm: Dict[str, Any] = field(default_factory=dict)
    reproducibility: Dict[str, Any] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, config_path: str = "config/settings.yaml") -> "Settings":
        """Load settings from YAML file."""
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config file {config_path} not found, using defaults")
            return cls()
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)

    def get(self, key: str, default: Any = None) -> Any:
        """Get nested config value using dot notation."""
        keys = key.split('.')
        value = self.__dict__
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get global settings instance (singleton)."""
    global _settings
    if _settings is None:
        _settings = Settings.load()
    return _settings


def reload_settings(config_path: str = "config/settings.yaml") -> Settings:
    """Reload settings from file."""
    global _settings
    _settings = Settings.load(config_path)
    return _settings