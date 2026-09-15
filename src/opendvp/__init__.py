# removed imaging because rasterio is giving issues with gdal and python
from . import imaging, io, metrics, pl, pp, tl, utils
from ._deprecated import register_module_alias as _register_module_alias
from ._deprecated import warn_renamed as _warn_renamed

try:
    from importlib.metadata import version as _version
except ImportError:
    from importlib_metadata import version as _version  # type: ignore

__version__ = _version("openDVP")

__all__ = [
    "io",
    "tl",
    "pl",
    "imaging",
    "metrics",
    "pp",
    "utils",
]

_DEPRECATED_MODULES = {"plotting": "pl"}

for _old, _new in _DEPRECATED_MODULES.items():
    _register_module_alias(f"{__name__}.{_old}", f"{__name__}.{_new}")
del _old, _new


def __getattr__(name: str):
    """Resolve deprecated module aliases on attribute access.

    Kept in ``__getattr__`` rather than bound eagerly so that ``import opendvp`` stays silent;
    the warning only fires when the old name is actually touched.
    """
    if name in _DEPRECATED_MODULES:
        new_name = _DEPRECATED_MODULES[name]
        _warn_renamed(f"{__name__}.{name}", f"{__name__}.{new_name}", stacklevel=3)
        return globals()[new_name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*__all__, *_DEPRECATED_MODULES])
