"""Shared base for immutable, strictly-validated domain value objects.

M0 defined its own private ``_Frozen`` inside ``contracts.py`` (which the
roadmap freezes). The indexing domain added in M1 reuses the same posture --
frozen, ``extra="forbid"``, whitespace-stripped -- through this public base so
the two never drift in strictness.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Frozen(BaseModel):
    """Immutable, strictly-validated value object.

    ``extra="forbid"`` matters here for the same reason it does on ``Finding``:
    parser and store output is validated into these types, and silently
    swallowing an unknown key would hide a schema drift bug.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )
