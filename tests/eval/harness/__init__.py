"""The M6 evaluation harness: run the pipeline on labeled PRs, score the output.

ROADMAP M6 (partial): *fake ``GitHubPort`` + recorded ``LLMPort`` so runs are
deterministic and free; metrics: precision, recall, FP rate, verification drop
rate by gate, cost, latency.*

Three modules, split along the line that matters:

``transcript``
    What the reviewer says. A fixed, hand-authored transcript replayed through
    ``RecordedLLM`` — the system under test's *input*, not its ground truth.
``runner``
    One labeled PR through M1 -> M3 and out to a ``GitHubPort``. No monkeypatching
    and no shortcuts: the same indexer, retriever, reviewer, verifier and budget
    the worker runs.
``metrics``
    Scoring the posted comments against the labels. Knows nothing about the
    transcript, which is the only reason its numbers mean anything.
"""

from __future__ import annotations
