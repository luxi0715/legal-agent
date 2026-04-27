"""Test version module."""

from legal_agent.core.version import get_version


def test_get_version() -> None:
    """Version should match the package version."""
    assert get_version() == "0.1.0"
