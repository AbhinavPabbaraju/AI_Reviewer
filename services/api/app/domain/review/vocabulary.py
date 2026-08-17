"""What ``SYMBOL_RESOLVES`` checks a model's claims against.

``VerificationContext.known_symbols`` is documented as "symbol fqns and bare
names from the symbol table **and the context pack**". Only the first half was
ever built, and the second half is not a nicety -- it is what makes the gate
measure the right thing.

The gate exists to catch a model *inventing* code: naming a helper that exists
nowhere, to make its reasoning sound checked. A symbol table alone answers a
narrower question, because it holds modules, classes, functions and methods and
nothing else. Parameters, locals, fields and module-level constants are real
code that the model was shown and is entitled to name, and under a
symbols-only vocabulary an explanation that says "slicing to ``limit``..." is
charged 0.30 confidence for quoting the parameter it is talking about. The M6
harness measured that cost: three true findings demoted in a 30-PR run, one of
them landing exactly on its severity floor with no margin left.

Harvesting identifiers from the retrieved chunks fixes it from the right end.
The context pack is, by construction, *the code the model actually saw*; a name
that appears in it was read, not invented. A name that appears in neither the
symbol table nor the retrieved source is still unaccounted for, which is the
claim the gate was always trying to test.

This deliberately does not read the repository at large. The question is not
"does this name exist somewhere" -- it is "could this model have known this
name", and the answer to that is exactly the pack it was given.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Final

from app.domain.retrieval.models import ContextPack

__all__ = ["build_vocabulary", "identifiers_in"]

_IDENTIFIER: Final = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")

_MAX_IDENTIFIER_CHARS: Final = 200
"""Longer than any real identifier. A minified or generated file that slipped
through the indexer's filters would otherwise contribute one enormous token per
line, which is memory spent to recognise nothing."""


def identifiers_in(source: str) -> frozenset[str]:
    """Every identifier-shaped token in a piece of source.

    Deliberately lexical rather than parsed. A parse would distinguish a
    definition from a mention, and the distinction is not wanted here: a name
    the model saw *mentioned* in its context pack is a name it did not invent,
    which is the only question this vocabulary answers.
    """
    return frozenset(
        token
        for token in _IDENTIFIER.findall(source)
        if len(token) <= _MAX_IDENTIFIER_CHARS
    )


def build_vocabulary(
    *,
    symbols: Iterable[str] = (),
    packs: Iterable[ContextPack] = (),
) -> frozenset[str]:
    """The names a review may cite without being suspected of inventing them.

    ``symbols`` are fqns; each contributes its full path *and* its bare tail,
    because explanations say ``MemoryRepository.put`` and never
    ``shop.store.memory.MemoryRepository.put``. Writing the full dotted path in
    prose would be unreadable, and demanding it would punish good explanations.

    An empty result disables the gate rather than failing every finding -- see
    ``VerificationContext.known_symbols``. That is the correct behaviour for "we
    have nothing to check against", and it is why this returns a set rather than
    raising when handed nothing.
    """
    names: set[str] = set()
    for fqn in symbols:
        names.add(fqn)
        names.add(_tail(fqn))
    for pack in packs:
        for item in pack.items:
            if item.chunk.symbol_fqn:
                names.add(item.chunk.symbol_fqn)
                names.add(_tail(item.chunk.symbol_fqn))
            names |= identifiers_in(item.chunk.content)
    return frozenset(names)


def _tail(fqn: str) -> str:
    """The bare name at the end of an fqn, in either language's convention.

    Python separates with ``.`` and TypeScript's fqns are ``<path>::<Name>``
    with dotted members inside, so both separators are split on rather than the
    language being dispatched on.
    """
    return fqn.replace("::", ".").rsplit(".", 1)[-1]
