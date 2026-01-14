"""
IRCA-Agent Configuration Module

Centralized configuration using Pydantic Settings.
All configuration can be set via environment variables or .env file.
"""

from .settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
