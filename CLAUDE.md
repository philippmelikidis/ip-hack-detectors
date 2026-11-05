# ip-hack-detectors

Replication of inoculation prompting (Wichers et al., arXiv 2510.05024) on MBPP and comparison
of several hack detectors (paper metric, differential tests, AST check, LLM judge) against
hand-labeled outputs.

- Language: everything in this repo is English (issues, PRs, commits, code, docs).
- Stack: Python 3.11+, uv, pytest, ruff. Training with TRL/peft, sampling with vLLM (on the GPU
  server: driver 580, CUDA 13, standard Torch wheels).
- Terms: `GLOSSARY.md`.
- Tests: `uv run --with pytest==9.1.1 pytest`. Lint: `uvx ruff@0.17.1 check . && uvx ruff@0.17.1 format --check .`
- The reference code safety-research/inoculation-prompting has no license: read only, copy
  nothing.
- The analysis plan lives in PREREGISTRATION.md and is not changed after it is frozen (dated
  addenda only).
- Merge: Claude may merge PRs in this repo once CI (lint and test) is green (approved by
  Philippos Melikidis on 04.11.2026). Squash, sole author Philippos Melikidis, no co-author,
  no AI attribution.
- The global rules in `~/.claude/CLAUDE.md` apply.

## Agent skills

### Issue tracker

Issues live in this repo's GitHub Issues. See `docs/agents/issue-tracker.md`.

### Triage labels

The five standard labels (needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix).
See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
