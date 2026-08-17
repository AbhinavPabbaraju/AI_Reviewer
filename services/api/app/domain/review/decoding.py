"""Turn model output into ``Finding``s, or into a counted rejection.

The model emits a **draft**, never a ``Finding``. A draft carries the claim --
severity, category, prose, location, evidence, confidence -- and nothing else.
Identity (``id``, ``run_id``), attribution (``source``, ``prompt_version``,
``model``) and verification status are assigned here, by the pipeline. That split
is not bookkeeping: a model that could write ``"verification": "verified"`` would
be grading its own work, and a model that could choose ``"source": "semgrep"``
could launder an unverifiable claim as a deterministic one.

Everything here is total. Model output is the least trustworthy input in the
system, and a decoder that raised on malformed JSON would turn a bad response
into a failed review. Every rejection is captured with a reason and counted --
``decode_reject_rate`` is a companion to the verification gate's drop rate, and
a prompt change that starts producing unparseable output should be visible as a
number rather than as a mysteriously quiet reviewer.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final
from uuid import UUID

from pydantic import Field, ValidationError

from app.domain.base import Frozen
from app.domain.contracts import (
    Category,
    CodeSpan,
    Evidence,
    EvidenceRole,
    Finding,
    FindingSource,
    Severity,
)

__all__ = [
    "DecodeResult",
    "DraftEvidence",
    "DraftFinding",
    "decode_findings",
    "draft_schema",
]

# Models wrap JSON in markdown fences with some regularity, instructions to the
# contrary. Tolerated on input rather than fought in the prompt: it is a
# formatting habit, not a content defect, and rejecting a well-formed finding
# over a ``` would be a self-inflicted recall loss.
_FENCED: Final = re.compile(
    r"^\s*```(?:json)?\s*\n(?P<body>.*?)\n?\s*```\s*$", re.DOTALL
)


class DraftEvidence(Frozen):
    """One cited span, as the model claims it."""

    path: str = Field(min_length=1)
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    role: EvidenceRole
    excerpt: str = Field(min_length=1)


class DraftFinding(Frozen):
    """One claimed defect, exactly as far as the model is trusted.

    Deliberately *not* a subset of ``Finding`` by inheritance: the fields the
    model must not control are absent from the type, so there is no path by
    which a model-supplied value reaches them.
    """

    severity: Severity
    category: Category
    title: str
    explanation: str
    path: str = Field(min_length=1)
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: tuple[DraftEvidence, ...] = Field(min_length=1)
    suggested_fix: str | None = None
    improved_code: str | None = None


@dataclass(frozen=True, slots=True)
class DecodeResult:
    """Findings that survived decoding, and why the others did not."""

    findings: tuple[Finding, ...]
    rejects: tuple[str, ...] = ()

    @property
    def reject_rate(self) -> float:
        total = len(self.findings) + len(self.rejects)
        return len(self.rejects) / total if total else 0.0


def draft_schema() -> dict[str, Any]:
    """JSON schema for the expected response, for providers that constrain
    decoding against one. Generated from the model rather than hand-written so
    the schema and the validator can never disagree."""
    return {
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "items": DraftFinding.model_json_schema(),
            }
        },
        "required": ["findings"],
    }


def decode_findings(
    payload: str,
    *,
    run_id: UUID,
    prompt_version: str,
    model: str | None = None,
    allowed_paths: Sequence[str] | None = None,
) -> DecodeResult:
    """Decode a model response into findings. Never raises.

    ``allowed_paths`` is an early, cheap containment check: a finding about a
    file the review was not looking at is dropped here rather than being carried
    to the verification gate. The gate would reject it anyway -- this only keeps
    the gate's per-file reads for claims that are at least plausible.
    """
    text = _unwrap(payload)
    if not text:
        return DecodeResult(findings=(), rejects=("empty response",))

    parsed, parse_error = _load(text)
    if parse_error is not None:
        return DecodeResult(findings=(), rejects=(f"invalid JSON: {parse_error}",))

    raw_findings, envelope_error = _findings_array(parsed)
    if envelope_error is not None:
        return DecodeResult(findings=(), rejects=(envelope_error,))

    findings: list[Finding] = []
    rejects: list[str] = []
    for index, entry in enumerate(raw_findings):
        outcome = _decode_one(
            entry,
            index=index,
            run_id=run_id,
            prompt_version=prompt_version,
            model=model,
            allowed_paths=allowed_paths,
        )
        if isinstance(outcome, Finding):
            findings.append(outcome)
        else:
            rejects.append(outcome)
    return DecodeResult(findings=tuple(findings), rejects=tuple(rejects))


def _unwrap(payload: str) -> str:
    """Strip a surrounding markdown fence, if there is one."""
    text = payload.strip()
    fenced = _FENCED.match(text)
    return fenced.group("body").strip() if fenced else text


def _load(text: str) -> tuple[object, str | None]:
    """Parse, falling back to the largest embedded JSON object.

    Parsing as-is comes first because narrowing to the outermost ``{...}`` is
    destructive on a response that is legitimately a top-level array -- it would
    strip the brackets and leave one element. The narrowing is only a rescue for
    a model that prefixed "Here is my review:" despite being told not to.
    """
    try:
        return json.loads(text), None
    except json.JSONDecodeError as error:
        first_error = str(error)

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1]), None
        except json.JSONDecodeError:
            pass
    return None, first_error


def _findings_array(parsed: object) -> tuple[list[Any], str | None]:
    """Accept the documented envelope, and a bare array as a courtesy."""
    if isinstance(parsed, list):
        return list(parsed), None
    if not isinstance(parsed, dict):
        return [], f"expected a JSON object, got {type(parsed).__name__}"
    raw = parsed.get("findings")
    if raw is None:
        return [], "response object has no 'findings' key"
    if not isinstance(raw, list):
        return [], f"'findings' must be an array, got {type(raw).__name__}"
    return raw, None


def _decode_one(
    entry: object,
    *,
    index: int,
    run_id: UUID,
    prompt_version: str,
    model: str | None,
    allowed_paths: Sequence[str] | None,
) -> Finding | str:
    """One draft in, one ``Finding`` or one rejection reason out."""
    if not isinstance(entry, dict):
        return f"findings[{index}]: expected an object, got {type(entry).__name__}"

    try:
        draft = DraftFinding.model_validate(entry)
    except ValidationError as error:
        return f"findings[{index}]: {_summarize(error)}"

    if allowed_paths is not None and draft.path not in allowed_paths:
        return (
            f"findings[{index}]: cites {draft.path!r}, which is not among the "
            f"files under review"
        )

    try:
        location = CodeSpan(
            path=draft.path,
            line_start=draft.line_start,
            line_end=max(draft.line_end, draft.line_start),
        )
        evidence = tuple(
            Evidence(
                span=CodeSpan(
                    path=item.path,
                    line_start=item.line_start,
                    line_end=max(item.line_end, item.line_start),
                ),
                role=item.role,
                excerpt=item.excerpt,
            )
            for item in draft.evidence
        )
    except ValidationError as error:
        return f"findings[{index}]: {_summarize(error)}"

    # `improved_code` without `suggested_fix` is rejected by the contract. It is
    # a formatting slip rather than a bad claim, so the patch is dropped and the
    # finding survives -- the same trade the PATCH_PARSES gate makes later.
    improved_code = draft.improved_code if draft.suggested_fix else None

    try:
        return Finding(
            run_id=run_id,
            severity=draft.severity,
            category=draft.category,
            source=FindingSource.LLM,
            title=draft.title,
            explanation=draft.explanation,
            location=location,
            evidence=evidence,
            suggested_fix=draft.suggested_fix,
            improved_code=improved_code,
            confidence=draft.confidence,
            prompt_version=prompt_version,
            model=model,
        )
    except ValidationError as error:
        # The contract's own validators: a vague title, evidence that does not
        # overlap the location, a missing DEFECT_SITE. These are real quality
        # signals about the response, so they are counted like any other reject.
        return f"findings[{index}]: {_summarize(error)}"


def _summarize(error: ValidationError) -> str:
    """One line per validation problem -- pydantic's full report is far too
    verbose to carry as a metric label."""
    parts: list[str] = []
    for detail in error.errors():
        location = ".".join(str(piece) for piece in detail["loc"]) or "(root)"
        parts.append(f"{location}: {detail['msg']}")
    return "; ".join(parts[:3])
