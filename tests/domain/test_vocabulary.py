"""The vocabulary ``SYMBOL_RESOLVES`` checks claims against.

Two failure modes, pulling in opposite directions, and the tests here are
written to hold both at once:

* too narrow and the gate punishes correct reviews for naming a parameter --
  the defect the M6 harness measured, three demotions in a 30-PR run;
* too wide and the gate stops catching invented symbols, which is the only
  thing it exists to do.
"""

from __future__ import annotations

from app.domain.contracts import CodeSpan
from app.domain.indexing.models import Chunk, Language
from app.domain.retrieval.models import (
    ContextItem,
    ContextPack,
    Provenance,
    RetrievalStats,
)
from app.domain.review.vocabulary import build_vocabulary, identifiers_in

STATS = RetrievalStats(
    anchors=1,
    graph_candidates=1,
    semantic_candidates=0,
    dropped_by_budget=0,
    tokens_used=10,
    token_budget=100,
    duration_ms=1,
)


def _pack(*sources: tuple[str, str | None]) -> ContextPack:
    """A pack of ``(content, symbol_fqn)`` chunks."""
    items = tuple(
        ContextItem(
            chunk=Chunk(
                content_hash=f"{index:064x}",
                symbol_fqn=fqn,
                span=CodeSpan(path="app/m.py", line_start=1, line_end=3),
                language=Language.PYTHON,
                token_count=10,
                content=content,
            ),
            provenance=Provenance.GRAPH,
            score=1.0,
            graph_distance=1,
            reason="test fixture",
        )
        for index, (content, fqn) in enumerate(sources)
    )
    return ContextPack(
        repository_id="r",
        hunks=(CodeSpan(path="app/m.py", line_start=1, line_end=3),),
        items=items,
        stats=STATS,
    )


class TestIdentifiersIn:
    def test_picks_out_identifier_shaped_tokens(self) -> None:
        found = identifiers_in("def truncate(value: str, limit: int) -> str:")
        assert {"truncate", "value", "limit", "str", "int"} <= found

    def test_ignores_literals_and_punctuation(self) -> None:
        """String contents and numbers are not names the model can be said to
        have read as code."""
        found = identifiers_in('total = 5000 + price("gbp")')
        assert {"total", "price"} <= found
        assert "5000" not in found

    def test_dollar_and_underscore_are_identifier_characters(self) -> None:
        """TypeScript and Python respectively. A vocabulary that dropped either
        would demote correct reviews of half the supported languages."""
        found = identifiers_in("const $el = _private_helper();")
        assert {"$el", "_private_helper"} <= found


class TestBuildVocabulary:
    def test_a_symbol_contributes_its_fqn_and_its_bare_name(self) -> None:
        """Explanations say ``MemoryRepository.put``, never the full dotted
        path; demanding the fqn in prose would punish good writing."""
        names = build_vocabulary(symbols=["shop.store.memory.MemoryRepository.put"])
        assert "shop.store.memory.MemoryRepository.put" in names
        assert "put" in names

    def test_typescript_fqns_split_on_the_module_separator(self) -> None:
        names = build_vocabulary(symbols=["src/util/text::truncate"])
        assert "truncate" in names

    def test_parameters_and_constants_come_from_the_retrieved_code(self) -> None:
        """The fix the harness forced. ``limit`` is a parameter and
        ``FREE_SHIPPING_THRESHOLD`` a module constant, so neither is in the
        symbol table -- but both are in the code the model was shown, so naming
        them is quoting rather than inventing.
        """
        pack = _pack(
            (
                "FREE_SHIPPING_THRESHOLD = 5000\n"
                "def truncate(value, limit):\n"
                "    return value[:limit]",
                "shop.util.text.truncate",
            )
        )
        names = build_vocabulary(symbols=["shop.util.text.truncate"], packs=[pack])
        assert {"limit", "value", "FREE_SHIPPING_THRESHOLD"} <= names

    def test_a_name_in_neither_source_stays_unknown(self) -> None:
        """The whole point of the gate. A helper that appears in no symbol table
        and in no retrieved chunk is unaccounted for, and a widened vocabulary
        must not quietly start accepting it."""
        pack = _pack(("def truncate(value, limit): ...", "shop.util.text.truncate"))
        names = build_vocabulary(symbols=["shop.util.text.truncate"], packs=[pack])
        assert "normalize_and_check" not in names
        assert "Sanitizer" not in names

    def test_an_unattributed_chunk_still_contributes_its_code(self) -> None:
        """A chunk with no ``symbol_fqn`` -- module-level code, a config file --
        is still code the model read."""
        names = build_vocabulary(packs=[_pack(("DEFAULT_DSN = 'sqlite://'", None))])
        assert "DEFAULT_DSN" in names

    def test_nothing_in_means_nothing_out(self) -> None:
        """An empty vocabulary disables the gate rather than failing every
        finding: a gate with nothing to check against must not manufacture
        verdicts. Returning empty is what ``VerificationContext`` reads as
        'disabled', so this must not raise."""
        assert build_vocabulary() == frozenset()

    def test_absurdly_long_tokens_are_dropped(self) -> None:
        """A minified file that slipped past the indexer's filters would
        otherwise contribute one enormous token per line -- memory spent to
        recognise nothing."""
        names = build_vocabulary(packs=[_pack((f"a{'b' * 500} = 1", None))])
        assert names == frozenset()
