"""The `plotting` -> `pl` rename shim, kept until openDVP 1.0."""

import importlib
import subprocess
import sys
import warnings

import pytest

import opendvp


def test_plotting_alias_is_pl() -> None:
    """The deprecated `plotting` alias resolves to the same module object as `pl`."""
    with pytest.warns(DeprecationWarning):
        plotting = opendvp.plotting
    assert plotting is opendvp.pl
    assert plotting.volcano is opendvp.pl.volcano


def test_plotting_alias_warns_with_migration_hint() -> None:
    """Touching `plotting` warns and names the replacement."""
    with pytest.warns(DeprecationWarning, match=r"`opendvp\.plotting` is deprecated.*`opendvp\.pl`"):
        _ = opendvp.plotting


@pytest.mark.parametrize(
    "statement",
    [
        "import opendvp; opendvp.plotting.volcano",
        "from opendvp import plotting; plotting.volcano",
        "import opendvp.plotting; opendvp.plotting.volcano",
        "from opendvp.plotting import volcano",
        "from opendvp.plotting.volcano import volcano",
    ],
)
def test_every_deprecated_import_form_still_works_and_warns(statement: str) -> None:
    """Attribute access is not the only way users reach `plotting`; all forms must keep working.

    `from opendvp.plotting import volcano` in particular was used by openDVP's own docstrings,
    so a shim that only covered `opendvp.plotting` attribute access would be a breaking change.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exec(statement, {})
    assert [w for w in caught if issubclass(w.category, DeprecationWarning)], (
        f"{statement!r} raised no DeprecationWarning"
    )


def test_deprecated_submodule_forwards_to_the_real_module() -> None:
    """`opendvp.plotting.volcano` must hand back the object from `opendvp.pl.volcano`."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = importlib.import_module("opendvp.plotting.volcano")
        new = importlib.import_module("opendvp.pl.volcano")
        assert old.volcano is new.volcano


def test_pl_does_not_warn() -> None:
    """The canonical name must never warn."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert opendvp.pl.volcano is not None


def test_importing_opendvp_is_silent() -> None:
    """Registering the alias must not warn at import time, only on use."""
    result = subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", "-c", "import opendvp; opendvp.pl.volcano"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_unknown_attribute_still_raises_attribute_error() -> None:
    """The alias shim must not swallow genuine typos."""
    with pytest.raises(AttributeError, match="has no attribute 'plottting'"):
        opendvp.plottting  # noqa: B018


def test_dir_lists_canonical_and_deprecated_names() -> None:
    names = dir(opendvp)
    assert "pl" in names
    assert "plotting" in names
