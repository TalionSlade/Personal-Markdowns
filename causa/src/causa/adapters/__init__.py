"""Concrete adapters behind the protocols defined in :mod:`causa.ports`.

Adapters are organised one sub-package per port:

- :mod:`causa.adapters.llm`        — LLM clients (Anthropic, mock)
- :mod:`causa.adapters.estimators` — DoWhy estimators (linear, propensity, DR)
- :mod:`causa.adapters.history`    — observation history (pandas, sqlite TBD)
- :mod:`causa.adapters.tools`      — debugging tools (stubbed and real)
"""

from __future__ import annotations
