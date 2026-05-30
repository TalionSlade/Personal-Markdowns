"""Causa command-line interface.

The ``causa`` CLI is the dissertation's reproducibility surface:

- ``causa scm-show``       — render the canonical debugging SCM.
- ``causa scm-extract``    — run the LLM graph extractor on a description.
- ``causa run``            — execute one agent over a task file.
- ``causa eval``           — run an arm of the §J3 ablation suite.
- ``causa trace-audit``    — summarise a JSON Lines trace file.

Implementation lives in :mod:`causa.cli.main`; this module is intentionally
empty so plugin discovery (and the ``causa`` console script) keeps working.
"""

from __future__ import annotations

from causa.cli.main import app

__all__ = ["app"]
