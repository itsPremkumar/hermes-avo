"""Placeholder smoke test for the multi-agent coordination test subpackage.

This module contains a single trivially passing test decorated with the
``coordination`` category marker. It exists so that ``pytest --collect-only``
reports at least one test per category before real coordination tests
are added.
"""


import pytest


@pytest.mark.coordination
def test_placeholder_coordination():
    """Smoke test ensuring the coordination subpackage is collectable."""
    assert True
