"""Test ProgramCard widget initialization safety with invalid strategies."""

import pytest
from src.tui.run.helpers import Programa
from src.tui.run.widgets import ProgramCard


def test_program_card_with_valid_strategy():
    """Test that ProgramCard initializes correctly with a valid strategy."""
    programa = Programa(
        nombre="test-prog",
        dataset="N4A",
        patron="patron-1",
        estrategia="basic",  # Valid strategy
    )

    # Should not raise an exception
    card = ProgramCard(programa)
    assert card is not None


def test_program_card_with_invalid_strategy():
    """Test that ProgramCard initializes correctly with an invalid strategy."""
    programa = Programa(
        nombre="test-prog",
        dataset="N4A",
        patron="patron-1",
        estrategia="nonexistent",  # Invalid strategy
    )

    # Should not raise an exception
    card = ProgramCard(programa)
    assert card is not None


def test_program_card_with_empty_strategy():
    """Test that ProgramCard initializes correctly with an empty strategy."""
    programa = Programa(
        nombre="test-prog",
        dataset="N4A",
        patron="patron-1",
        estrategia="",  # Empty strategy
    )

    # Should not raise an exception
    card = ProgramCard(programa)
    assert card is not None


def test_program_card_with_obsolete_strategy():
    """Test that ProgramCard initializes correctly with an obsolete strategy."""
    programa = Programa(
        nombre="test-prog",
        dataset="N4A",
        patron="patron-1",
        estrategia="base",  # Obsolete strategy (should be migrated to "basic")
    )

    # Should not raise an exception
    card = ProgramCard(programa)
    assert card is not None
