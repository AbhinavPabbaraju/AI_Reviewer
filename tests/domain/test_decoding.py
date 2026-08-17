"""Constrained decoding of model output into ``Finding``s.

Model output is the least trustworthy input in the system, so the bulk of these
tests are about malformed, hostile, or subtly-wrong responses. The decoder must
be total: every one of these produces a counted rejection, never an exception.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from app.domain.contracts import (
    Category,
    EvidenceRole,
    FindingSource,
    Severity,
    VerificationStatus,
)
from app.domain.review.decoding import decode_findings, draft_schema

RUN_ID = UUID("22222222-2222-2222-2222-222222222222")
PATH = "app/store.py"


def draft(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "severity": "high",
        "category": "correctness",
        "title": "Unnormalized path join allows directory traversal",
        "explanation": "The joined path is never normalized, so a traversal "
        "segment escapes the configured root directory.",
        "path": PATH,
        "line_start": 9,
        "line_end": 9,
        "confidence": 0.8,
        "evidence": [
            {
                "path": PATH,
                "line_start": 9,
                "line_end": 9,
                "role": "defect_site",
                "excerpt": "path = os.path.join(self.root, name)",
            }
        ],
    }
    base.update(overrides)
    return base


def payload(*drafts: dict[str, Any]) -> str:
    return json.dumps({"findings": list(drafts)})


def decode(text: str, **kwargs: Any) -> Any:
    kwargs.setdefault("run_id", RUN_ID)
    kwargs.setdefault("prompt_version", "reviewer/v1")
    return decode_findings(text, **kwargs)


class TestHappyPath:
    def test_decodes_a_well_formed_finding(self) -> None:
        result = decode(payload(draft()))
        assert result.rejects == ()
        [finding] = result.findings
        assert finding.severity is Severity.HIGH
        assert finding.category is Category.CORRECTNESS
        assert finding.location.path == PATH
        assert finding.confidence == 0.8

    def test_empty_findings_array_is_success_not_failure(self) -> None:
        """"I found nothing" is the correct answer more often than not, and must
        not look like a decode failure."""
        result = decode('{"findings": []}')
        assert result.findings == ()
        assert result.rejects == ()
        assert result.reject_rate == 0.0

    def test_decodes_several_findings(self) -> None:
        result = decode(
            payload(draft(), draft(title="Handle is leaked when read raises"))
        )
        assert len(result.findings) == 2

    def test_bare_array_is_accepted(self) -> None:
        result = decode(json.dumps([draft()]))
        assert len(result.findings) == 1


class TestPipelineOwnedFields:
    """The model supplies a claim; the pipeline supplies identity and
    attribution. These are the fields a model must never be able to set."""

    def test_source_and_attribution_are_assigned_not_taken(self) -> None:
        result = decode(payload(draft()), model="qwen2.5-coder:7b")
        [finding] = result.findings
        assert finding.source is FindingSource.LLM
        assert finding.prompt_version == "reviewer/v1"
        assert finding.model == "qwen2.5-coder:7b"
        assert finding.run_id == RUN_ID

    def test_model_cannot_declare_itself_verified(self) -> None:
        result = decode(
            payload(draft(verification={"status": "verified"})),
        )
        # `extra="forbid"` on the draft rejects the whole finding rather than
        # silently dropping the field -- an attempt to set it is a signal.
        assert result.findings == ()
        assert "verification" in result.rejects[0]

    def test_model_cannot_claim_a_deterministic_source(self) -> None:
        result = decode(payload(draft(source="semgrep")))
        assert result.findings == ()
        assert result.rejects

    def test_findings_start_unverified(self) -> None:
        [finding] = decode(payload(draft())).findings
        assert finding.verification.status is VerificationStatus.PENDING


class TestMalformedOutput:
    def test_markdown_fence_is_tolerated(self) -> None:
        """Models wrap JSON in fences despite instructions. Losing a valid
        finding over three backticks would be a self-inflicted recall loss."""
        result = decode(f"```json\n{payload(draft())}\n```")
        assert len(result.findings) == 1

    def test_bare_fence_without_language_is_tolerated(self) -> None:
        result = decode(f"```\n{payload(draft())}\n```")
        assert len(result.findings) == 1

    def test_surrounding_prose_is_tolerated(self) -> None:
        result = decode(f"Here is my review:\n{payload(draft())}\nHope that helps!")
        assert len(result.findings) == 1

    def test_invalid_json_is_a_counted_reject(self) -> None:
        result = decode("{not json at all")
        assert result.findings == ()
        assert "invalid JSON" in result.rejects[0]

    def test_empty_response_is_a_counted_reject(self) -> None:
        result = decode("   ")
        assert result.findings == ()
        assert result.rejects == ("empty response",)

    def test_missing_findings_key_is_a_counted_reject(self) -> None:
        result = decode('{"results": []}')
        assert "no 'findings' key" in result.rejects[0]

    def test_findings_not_an_array_is_a_counted_reject(self) -> None:
        result = decode('{"findings": "none"}')
        assert "must be an array" in result.rejects[0]

    def test_one_bad_finding_does_not_lose_the_good_ones(self) -> None:
        result = decode(payload(draft(), {"severity": "high"}, draft()))
        assert len(result.findings) == 2
        assert len(result.rejects) == 1
        assert result.reject_rate == 1 / 3


class TestContractViolations:
    """The contract's own validators, surfaced as decode rejects."""

    def test_vague_title_is_rejected(self) -> None:
        result = decode(payload(draft(title="This could be improved somewhat")))
        assert result.findings == ()
        assert result.rejects

    def test_evidence_not_overlapping_the_location_is_rejected(self) -> None:
        """The model pointed the comment at one place and reasoned about
        another -- subtle, and exactly what this validator exists to catch."""
        result = decode(
            payload(
                draft(
                    line_start=9,
                    line_end=9,
                    evidence=[
                        {
                            "path": PATH,
                            "line_start": 40,
                            "line_end": 41,
                            "role": "defect_site",
                            "excerpt": "somewhere else entirely",
                        }
                    ],
                )
            )
        )
        assert result.findings == ()
        assert "does not overlap" in result.rejects[0]

    def test_missing_defect_site_role_is_rejected(self) -> None:
        result = decode(
            payload(
                draft(
                    evidence=[
                        {
                            "path": PATH,
                            "line_start": 9,
                            "line_end": 9,
                            "role": "caller",
                            "excerpt": "store.read(name)",
                        }
                    ]
                )
            )
        )
        assert result.findings == ()
        assert "DEFECT_SITE" in result.rejects[0]

    def test_empty_evidence_is_rejected(self) -> None:
        result = decode(payload(draft(evidence=[])))
        assert result.findings == ()
        assert result.rejects

    def test_traversal_path_is_rejected(self) -> None:
        """Model output reaches a filesystem read later in the pipeline; the
        span type refuses traversal at the boundary."""
        result = decode(payload(draft(path="../../etc/passwd")), allowed_paths=None)
        assert result.findings == ()
        assert result.rejects

    def test_patch_without_prose_keeps_the_finding_and_drops_the_patch(self) -> None:
        """A formatting slip, not a bad claim -- the same trade PATCH_PARSES
        makes later in the gate."""
        result = decode(payload(draft(improved_code="x = 1", suggested_fix=None)))
        [finding] = result.findings
        assert finding.improved_code is None
        assert finding.suggested_fix is None

    def test_patch_with_prose_survives(self) -> None:
        result = decode(
            payload(draft(improved_code="x = 1", suggested_fix="Normalize it."))
        )
        [finding] = result.findings
        assert finding.improved_code == "x = 1"

    def test_confidence_out_of_range_is_rejected(self) -> None:
        assert decode(payload(draft(confidence=1.5))).findings == ()

    def test_unknown_severity_is_rejected(self) -> None:
        assert decode(payload(draft(severity="catastrophic"))).findings == ()

    def test_inverted_line_range_is_repaired_not_rejected(self) -> None:
        """`line_end < line_start` is a transposition, not a fabricated
        location; the span is normalized rather than the finding lost."""
        [finding] = decode(payload(draft(line_start=9, line_end=4))).findings
        assert finding.location.line_start == 9
        assert finding.location.line_end == 9


class TestPathContainment:
    def test_finding_about_another_file_is_rejected(self) -> None:
        result = decode(payload(draft(path="app/other.py")), allowed_paths=(PATH,))
        assert result.findings == ()
        assert "not among the files under review" in result.rejects[0]

    def test_finding_about_the_reviewed_file_passes(self) -> None:
        result = decode(payload(draft()), allowed_paths=(PATH,))
        assert len(result.findings) == 1


class TestSchema:
    def test_schema_describes_the_findings_envelope(self) -> None:
        schema = draft_schema()
        assert schema["required"] == ["findings"]
        assert schema["properties"]["findings"]["type"] == "array"

    def test_schema_is_generated_from_the_validator(self) -> None:
        """Hand-writing the schema separately is how a schema and its validator
        drift apart."""
        item = draft_schema()["properties"]["findings"]["items"]
        assert set(EvidenceRole) and "properties" in item
        assert "severity" in item["properties"]
        assert "evidence" in item["properties"]
