"""Placeholder smoke test for the end-to-end test subpackage.

This module contains a single trivially passing test decorated with the
``e2e`` category marker. It exists so that ``pytest --collect-only``
reports at least one test per category before real e2e tests are added.
"""


import pytest


@pytest.mark.e2e
def test_placeholder_e2e():
    """Smoke test ensuring the e2e subpackage is collectable."""
    assert True
