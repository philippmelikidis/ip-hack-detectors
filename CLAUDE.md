# ip-hack-detectors

Replikation von Inoculation Prompting (Wichers et al., arXiv 2510.05024) auf MBPP und Vergleich mehrerer Hack-Detektoren
(Paper-Metrik, Differential-Tests, AST-Check, LLM-Judge) gegen handgelabelte Ausgaben.

- Stack: Python 3.11+, uv, pytest, ruff. Training mit TRL/peft, Sampling mit vLLM (auf dem GPU-Server, Torch cu118).
- Tests: `uv run --with pytest==9.1.1 pytest`. Lint: `uvx ruff@0.17.1 check . && uvx ruff@0.17.1 format --check .`
- Der Referenz-Code safety-research/inoculation-prompting hat keine Lizenz: nur lesen, nichts übernehmen.
- Analyseplan steht in PREREGISTRATION.md und wird nach Beginn der Hauptläufe nicht mehr geändert (nur Nachträge mit Datum).
- Es gelten die globalen Regeln aus `~/.claude/CLAUDE.md`.

## Agent skills

### Issue tracker

Issues liegen in GitHub Issues dieses Repos. See `docs/agents/issue-tracker.md`.

### Triage labels

Die fünf Standard-Labels (needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` und `docs/adr/` im Repo-Root. See `docs/agents/domain.md`.
