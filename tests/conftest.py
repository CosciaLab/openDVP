"""Shared pytest configuration.

openDVP's plotting functions call `plt.show()` whenever `return_fig` is False, which opens a real
window on a developer machine with a GUI. Force a non-interactive backend for the whole session
so running the tests never steals focus, and close whatever figures a test leaves behind so a
long run does not accumulate hundreds of them.

This module is imported before any test module, which is what makes `matplotlib.use` effective.
"""

import matplotlib
import pytest

matplotlib.use("Agg", force=True)


@pytest.fixture(autouse=True)
def _close_figures():
    """Close any figures a test leaves open."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
