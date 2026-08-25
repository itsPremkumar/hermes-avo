"""Placeholder smoke test for the planning engine test subpackage.

This module contains a single trivially passing test decorated with the
``planning`` category marker. It exists so that ``pytest --collect-only``
reports at least one test per category before real planning tests are
added.
"""


import pytest


@pytest.mark.planning
def test_placeholder_planning():
    """Smoke test ensuring the planning subpackage is collectable."""
    assert True
