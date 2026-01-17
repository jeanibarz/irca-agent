"""
IRCA-Agent Configuration Module

Centralized configuration using Pydantic Settings.
All configuration can be set via environment variables or .env file.
"""

from .settings import Settings, clear_settings_cache, get_settings

__all__ = ["Settings", "clear_settings_cache", "get_settings"]
