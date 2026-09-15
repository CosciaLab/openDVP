"""Machinery for module names kept around for backwards compatibility.

`plotting` was renamed to `pl` in 0.8.0 to match scverse conventions. Registering the old name
in `sys.modules` keeps every import form working, not just attribute access on the package, so
`from opendvp.plotting import volcano` behaves as before while pointing users at the new name.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
import warnings
from types import ModuleType

REMOVED_IN = "1.0"


class DeprecatedModuleAlias(ModuleType):
    """Forward attribute access to a renamed module, warning on each public lookup.

    Dunder and private names are forwarded silently: the import machinery reads `__path__`,
    `__spec__` and friends, and warning on those would fire without any user code touching
    the deprecated name.
    """

    def __init__(self, old_name: str, new_name: str, target: ModuleType) -> None:
        super().__init__(old_name, f"Deprecated alias for :mod:`{new_name}`.")
        self.__dict__["_target"] = target
        self.__dict__["_new_name"] = new_name

    def __getattr__(self, name: str) -> object:
        if not name.startswith("_"):
            warn_renamed(self.__name__, self.__dict__["_new_name"])
        return getattr(self.__dict__["_target"], name)

    def __dir__(self) -> list[str]:
        return dir(self.__dict__["_target"])


def warn_renamed(old_name: str, new_name: str, stacklevel: int = 3) -> None:
    """Emit the standard rename warning for `old_name`."""
    warnings.warn(
        f"`{old_name}` is deprecated and will be removed in openDVP {REMOVED_IN}. Use `{new_name}` instead.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def register_module_alias(old_name: str, new_name: str) -> None:
    """Register `old_name`, and each of its submodules, as aliases of `new_name`."""
    target = importlib.import_module(new_name)
    sys.modules.setdefault(old_name, DeprecatedModuleAlias(old_name, new_name, target))
    for info in pkgutil.iter_modules(target.__path__):
        sub_old, sub_new = f"{old_name}.{info.name}", f"{new_name}.{info.name}"
        sys.modules.setdefault(sub_old, DeprecatedModuleAlias(sub_old, sub_new, importlib.import_module(sub_new)))
