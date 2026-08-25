"""Placeholder smoke test for the DeepAgent wrapper test subpackage.

This module contains a single trivially passing test decorated with the
``wrapper`` category marker. It exists so that ``pytest --collect-only``
reports at least one test per category before real wrapper tests are
added.
"""


import pytest


@pytest.mark.wrapper
def test_placeholder_wrapper():
    """Smoke test ensuring the wrapper subpackage is collectable."""
    assert True
