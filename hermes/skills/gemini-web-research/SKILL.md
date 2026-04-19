---
name: gemini-web-research
description: Run current web research through the hermes-gemini-web-research Python package. Use when Hermes or Codex needs multiple parallel Gemini CLI research workers, strict JSON evidence capture, Markdown or JSON report files, custom research angles, or deterministic reconciliation with optional semantic synthesis fallback.
---

# Gemini Web Research

## Overview

Use this repository as a Hermes research tool when a question benefits from multiple independent web-research angles and a persisted report. The package runs Gemini CLI workers concurrently, validates each worker's JSON, reconciles findings deterministically, and can write Markdown or JSON output files.

## Workflow

1. Confirm the local package is installed from the repository root:

```bash
python -m pip install -e ".[dev]"
```

2. Choose angles. Use defaults for general research, pass repeated `--angle` values for ad hoc work, or use `--angle-file` for repeatable angle sets. Read `references/angle-sets.md` when selecting reusable angle patterns.

3. Run the CLI with an output file so Hermes can consume the artifact without scraping stdout:

```bash
hgw-research \
  "What are the current tradeoffs of using Gemini CLI for web research orchestration?" \
  --angle-file examples/angles.json \
  --output-file reports/gemini-web-research.md
```

4. Prefer JSON output when another tool will parse the result:

```bash
hgw-research \
  "What changed in this ecosystem recently?" \
  --format json \
  --output-file reports/gemini-web-research.json
```

5. Inspect the report's `status`, `synthesis_method`, `synthesis_error`, worker errors, limitations, and evidence URLs before using the answer in downstream work.

## Operating Notes

- Keep deterministic reconciliation as the baseline. Use `--semantic-synthesis` when you want the built-in GeminiSemanticSynthesizer to run a second pass over the deterministic result.
- Use `--timeout` and `--max-concurrency` to control latency and Gemini CLI load.
- Use repeated `--gemini-arg` values when a local Gemini wrapper needs a different headless invocation shape.
- Do not assume every worker succeeds. A `partial` report can still be useful when limitations make the gaps clear.

## CLI Examples

Deterministic Markdown report:

```bash
hgw-research \
  "What changed in this ecosystem recently?" \
  --angle-file examples/angles.json \
  --output-file reports/gemini-web-research.md
```

Built-in semantic synthesis with JSON output:

```bash
hgw-research \
  "Should Hermes use Gemini CLI workers for current research?" \
  --semantic-synthesis \
  --format json \
  --output-file reports/gemini-web-research.json
```

## Python API

Use the API when Hermes wants direct control over the runner or wants to swap in a custom synthesizer implementation:

```python
from hermes_gemini_web_research import GeminiSemanticSynthesizer, ResearchOrchestrator, ResearchRequest
from hermes_gemini_web_research.runner import GeminiRunner

runner = GeminiRunner()
request = ResearchRequest(
    question="Should Hermes use Gemini CLI workers for current research?",
    semantic_synthesis=True,
)

result = await ResearchOrchestrator(
    runner,
    synthesizer=GeminiSemanticSynthesizer(runner=runner),
).run(request)
```

Custom synthesizers can still implement `async synthesize(question, deterministic_result, timeout_seconds)`. If a synthesizer raises, the orchestrator returns the deterministic result with `synthesis_error` populated.
