"""
tests/test_cli_smoke.py
========================
Pytest wrapper around the CLI smoke_test() function.
Verifies the function returns 0 (all assertions pass) without
any network calls or LLM access.
"""

from __future__ import annotations

from run_cli import smoke_test


def test_smoke_test_passes():
    """smoke_test() should return 0 when all local checks pass."""
    assert smoke_test() == 0
