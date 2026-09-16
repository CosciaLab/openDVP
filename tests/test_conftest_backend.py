"""Guard the headless-backend setup in `conftest.py`.

Without it, every `pl` test that exercises the `return_fig=False` branch opens a window.
"""

import matplotlib


def test_backend_is_non_interactive():
    assert matplotlib.get_backend().lower() == "agg"


def test_pyplot_is_not_interactive():
    import matplotlib.pyplot as plt

    assert not plt.isinteractive()
