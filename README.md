# ip-hack-detectors

Measuring what inoculation prompting prevents: do hack detectors agree?

Replication of inoculation prompting (Wichers et al., arXiv 2510.05024) on MBPP and comparison
of several hack detectors against hand-labeled outputs. Spec: issue #3. Terms: `GLOSSARY.md`.

## Status

So far the pipeline runs with a fake backend (fixed answers instead of a model): load tasks,
build the prompt, extract code, run it in a sandbox, D1 and correct rate with Wilson 95% CIs.

```bash
uv run ipd reproduce --config configs/mbpp-fake.yaml
```

The check run `configs/mbpp-fake.yaml` uses the real 257 tasks of MBPP sanitized test (pinned
revision) and expects: reference solutions are correct, hack solutions are D1 hacks.

## Sandbox

Generated code runs in its own Python subprocess per test, with a timeout, CPU and file-size
limits, a memory limit (Linux only), a temporary directory, an empty environment and blocked
sockets. This is best-effort protection, not real isolation. File access outside the
temporary directory is not blocked, and only the standard library is available. The task's
imports precede the asserts (a deviation from the paper's reference code).

## Development

```bash
uv run --with pytest==9.1.1 pytest
uvx ruff@0.17.1 check . && uvx ruff@0.17.1 format --check .
```
