"""Turning a review into something worth reading in a terminal.

The output is the product here. A reviewer that finds a real bug and buries it
under statistics has not helped anyone, so the findings come first, at full
width, with the evidence quoted from the file; the numbers go at the bottom in
one line.

Two rules the rest of this module follows:

**Say what was suppressed, and why.** A tool that silently drops most of what its
model produced looks either lucky or broken, and the difference matters when you
are deciding whether to trust it. The per-gate counts are the honest version of
"I checked, and here is what did not survive".

**Never colour into a pipe.** Colour is decoration for a human at a terminal;
written into a file or a CI log it is line noise. ``NO_COLOR`` is honoured
because it is the convention users already have.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, TextIO

from app.domain.contracts import Finding, Severity, VerificationGate

__all__ = ["Palette", "render_findings", "render_json", "render_summary"]

_SEVERITY_ORDER: Final = (
    Severity.CRITICAL,
    Severity.HIGH,
    Severity.MEDIUM,
    Severity.LOW,
    Severity.INFO,
)

_GATE_EXPLANATION: Final = {
    VerificationGate.FILE_EXISTS: "cited a file that is not in the tree",
    VerificationGate.LINE_IN_RANGE: "cited a line outside the change",
    VerificationGate.SYMBOL_RESOLVES: "named code that does not exist",
    VerificationGate.PATCH_PARSES: "suggested a patch that does not parse",
    VerificationGate.PATCH_APPLIES: "suggested a patch that will not apply",
    VerificationGate.NOT_DUPLICATE: "repeated another comment",
    VerificationGate.CONFIDENCE_FLOOR: "was not confident enough to be worth it",
}


@dataclass(frozen=True, slots=True)
class Palette:
    """ANSI codes, or empty strings when colour is not wanted."""

    reset: str = ""
    dim: str = ""
    bold: str = ""
    critical: str = ""
    high: str = ""
    medium: str = ""
    low: str = ""
    info: str = ""
    accent: str = ""

    @classmethod
    def for_stream(cls, stream: TextIO, *, force: bool | None = None) -> Palette:
        enabled = (
            force
            if force is not None
            else stream.isatty() and not os.environ.get("NO_COLOR")
        )
        if not enabled:
            return cls()
        return cls(
            reset="\x1b[0m",
            dim="\x1b[2m",
            bold="\x1b[1m",
            critical="\x1b[1;31m",
            high="\x1b[31m",
            medium="\x1b[33m",
            low="\x1b[36m",
            info="\x1b[2m",
            accent="\x1b[35m",
        )

    def severity(self, severity: Severity) -> str:
        return {
            Severity.CRITICAL: self.critical,
            Severity.HIGH: self.high,
            Severity.MEDIUM: self.medium,
            Severity.LOW: self.low,
            Severity.INFO: self.info,
        }[severity]


def render_findings(
    findings: Sequence[Finding],
    *,
    palette: Palette,
    out: TextIO | None = None,
) -> None:
    """One block per finding, most severe first."""
    out = out or sys.stdout
    if not findings:
        return
    ordered = sorted(
        findings, key=lambda f: (-f.severity.rank, -f.confidence, str(f.location))
    )
    for finding in ordered:
        colour = palette.severity(finding.severity)
        print(file=out)
        print(
            f"{colour}{palette.bold}{finding.severity.value.upper():<8}{palette.reset}"
            f"{colour}{finding.location}{palette.reset}"
            f"{palette.dim}  {finding.category.value}"
            f"  confidence {finding.confidence:.2f}{palette.reset}",
            file=out,
        )
        print(f"  {palette.bold}{finding.title}{palette.reset}", file=out)
        for line in _wrap(finding.explanation, width=76):
            print(f"  {line}", file=out)

        for evidence in finding.evidence:
            excerpt = evidence.excerpt.strip().splitlines()
            if not excerpt:
                continue
            label = evidence.role.value.replace("_", " ")
            print(
                f"  {palette.dim}{label} {evidence.span}{palette.reset}", file=out
            )
            for line in excerpt[:6]:
                print(f"  {palette.dim}| {line}{palette.reset}", file=out)

        if finding.suggested_fix:
            print(f"  {palette.accent}fix{palette.reset} {finding.suggested_fix}",
                  file=out)
        if finding.improved_code:
            for line in finding.improved_code.splitlines()[:12]:
                print(f"  {palette.accent}+ {line}{palette.reset}", file=out)


def render_summary(
    *,
    posted: Sequence[Finding],
    suppressed: Sequence[Finding],
    drops: dict[VerificationGate, int],
    decode_rejects: int,
    failures: Sequence[str],
    completeness: float,
    units: int,
    seconds: float,
    model: str,
    palette: Palette,
    out: TextIO | None = None,
) -> None:
    """The one-screen account of what happened, after the findings."""
    out = out or sys.stdout
    print(file=out)
    if posted:
        counts = " · ".join(
            f"{palette.severity(severity)}{sum(1 for f in posted if f.severity is severity)}"
            f" {severity.value}{palette.reset}"
            for severity in _SEVERITY_ORDER
            if any(f.severity is severity for f in posted)
        )
        print(f"{palette.bold}{len(posted)} comment(s){palette.reset}  {counts}",
              file=out)
    else:
        print(
            f"{palette.bold}No comments.{palette.reset} "
            f"{palette.dim}Argus reviewed the change and found nothing it could "
            f"stand behind.{palette.reset}",
            file=out,
        )

    if suppressed:
        reasons = sorted(drops.items(), key=lambda item: -item[1])
        detail = ", ".join(
            f"{count} {_GATE_EXPLANATION.get(gate, gate.value)}"
            for gate, count in reasons
        )
        print(
            f"{palette.dim}{len(suppressed)} suppressed by verification: "
            f"{detail}{palette.reset}",
            file=out,
        )
    if decode_rejects:
        print(
            f"{palette.dim}{decode_rejects} malformed response(s) discarded"
            f"{palette.reset}",
            file=out,
        )
    if failures:
        print(
            f"{palette.medium}{len(failures)} review unit(s) failed; this review "
            f"is incomplete ({completeness:.0%} covered){palette.reset}",
            file=out,
        )
        for failure in failures[:3]:
            print(f"{palette.dim}  {failure}{palette.reset}", file=out)

    print(
        f"{palette.dim}{units} review unit(s) · {model} · {seconds:.0f}s · "
        f"$0.00{palette.reset}",
        file=out,
    )


def render_json(
    *,
    posted: Sequence[Finding],
    suppressed: Sequence[Finding],
    drops: dict[VerificationGate, int],
    completeness: float,
    model: str,
    out: TextIO | None = None,
) -> None:
    """Machine-readable output.

    Findings are serialized through the contract's own ``model_dump`` rather
    than a hand-built dict, so the shape here is the shape the API serves and
    the frontend's generated types describe. One source of truth, three
    consumers.
    """
    out = out or sys.stdout
    payload = {
        "model": model,
        "completeness": completeness,
        "findings": [json.loads(f.model_dump_json()) for f in posted],
        "suppressed": [json.loads(f.model_dump_json()) for f in suppressed],
        "drops_by_gate": {gate.value: count for gate, count in drops.items()},
        "cost_usd": 0.0,
    }
    json.dump(payload, out, indent=2)
    out.write("\n")


def _wrap(text: str, *, width: int) -> list[str]:
    """Greedy wrap. ``textwrap`` would do this, and would also collapse the
    paragraph breaks that make a long explanation readable."""
    lines: list[str] = []
    for paragraph in text.splitlines():
        if not paragraph.strip():
            lines.append("")
            continue
        current = ""
        for word in paragraph.split():
            if current and len(current) + 1 + len(word) > width:
                lines.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        if current:
            lines.append(current)
    return lines
