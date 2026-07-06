# Project documentation

This directory holds the durable design and audit-trail artifacts produced by the
[`rig`](https://github.com/guilhermegrahl/rig) `/orchestrate` run that built
`tinyagent`. The orchestrator's per-run scratch space lives under
`.harness/<run-id>/` (gitignored); everything in `docs/` is committed and
shippable.

## Layout

| File | What it is | Source |
|---|---|---|
| [`decisions.md`](decisions.md) | The 6 conflict resolutions (C1–C6) made during design. | T15c |
| [`requirements.md`](requirements.md) | Locked user spec (10 decisions + 9 success criteria). | Phase 1 |
| [`research.md`](research.md) | Upstream `mozilla-ai/tinyagent` investigation, anchor files, known conflicts. | Phase 2 |
| [`plan.md`](plan.md) | Peer-reviewed, approved implementation plan (single-file outline, conflict resolutions, TDD-ordered task breakdown). | Phase 3 |
| [`revisions.md`](revisions.md) | 3 rounds of architect revisions driven by the peer reviewer. | Phase 3 |
| [`peer-review.json`](peer-review.json) | Final peer-review verdict (decision: approve, ship-ready). | Phase 3 |
| [`tasks.md`](tasks.md) | 22-task dependency graph with critical-path and parallel-band analysis. | Phase 4 |
| [`qa-report.md`](qa-report.md) | Validation roll-up: spec coverage, static analysis, behavioral results. | Phase 6 |
| [`qa/`](qa/) | Raw validator JSON outputs. | Phase 6 |

## Status

- **Implementation**: 22/22 tasks complete (`T1`–`T16`, with `T12` split into `a/b/c/d/cross` and `T15` split into `a/b/c`).
- **Tests**: 268 unit tests pass, 6 integration scenarios gated on `ANY_LLM_TEST_MODEL`.
- **License**: Apache-2.0 (forked from `mozilla-ai/tinyagent`, see `NOTICE`).

## How to navigate

- **Want to understand the design?** Start at [`requirements.md`](requirements.md) → [`decisions.md`](decisions.md) → [`plan.md`](plan.md) §0 + §13.
- **Want to understand what shipped?** [`tasks.md`](tasks.md) → [`plan.md`](plan.md) §13 (task breakdown).
- **Want to know what changed during design review?** [`revisions.md`](revisions.md) + [`peer-review.json`](peer-review.json).
- **Want the QA numbers?** [`qa-report.md`](qa-report.md).