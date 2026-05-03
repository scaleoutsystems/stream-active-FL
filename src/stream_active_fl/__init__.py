"""
Stream-Active Federated Learning for object detection.

Submodules are imported lazily on first attribute access so that
analysis-only consumers (e.g. notebooks under `analysis/`) can use the
package without pulling in heavy optional dependencies such as PyTorch.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.2.0"

__all__ = [
    "analysis",
    "core",
    "evaluation",
    "memory",
    "models",
    "policies",
    "runtime",
    "tracking",
    "training",
    "utils",
]

_SUBMODULES = frozenset(__all__)


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:  # pragma: no cover - type-checker hint only
    from stream_active_fl import (  # noqa: F401
        analysis,
        core,
        evaluation,
        memory,
        models,
        policies,
        runtime,
        tracking,
        training,
        utils,
    )
