"""Pydantic-Settings configuration object.

A single :class:`CausaSettings` instance is the source of truth for every
tunable knob.  Settings are loaded from environment variables and `.env`
files via :mod:`pydantic_settings`.  Tests use :func:`load_settings` with an
explicit override map.
"""

from __future__ import annotations

from causa.config.settings import CausaSettings, load_settings

__all__ = ["CausaSettings", "load_settings"]
