# hermes-gemini-web-research

`hermes-gemini-web-research` is a small Python controller for running multiple headless Gemini CLI research workers in parallel, validating each worker's strict JSON output with Pydantic, and reconciling the results into a Markdown or JSON report file.

The intended use case is Hermes, or another controller, asking 3-6 independent research angles to investigate one question, then collecting evidence, caveats, telemetry, and a synthesized result without depending on brittle free-form model text.

## Status

This is an MVP scaffold. It has a practical subprocess runner, a stable internal contract, deterministic reconciliation, optional semantic synthesis hooks with fallback, and report-file output. The default Gemini command targets headless JSON mode, while still allowing wrapper-specific flags to be overridden.

## Architecture

The package lives under `src/hermes_gemini_web_research/`:

- `models.py` defines Pydantic models for research requests, angles, worker input/output, evidence, wrapper telemetry, worker results, and final reconciliation.
- `prompts.py` builds strict JSON prompts and extracts JSON objects from raw model text.
- `runner.py` runs Gemini CLI via `asyncio.create_subprocess_exec`, applies per-worker timeouts, strips terminal escape sequences, retries bounded recoverable failures with exponential backoff, parses optional wrapper JSON, extracts telemetry, and validates worker JSON.
- `orchestrator.py` runs angles concurrently with a bounded semaphore, passes results to reconciliation, and optionally calls a semantic synthesizer.
- `reconcile.py` performs a conservative deterministic synthesis over successful worker outputs and records limitations from failed or uncertain workers.
- `report.py` renders final structured results as Markdown.
- `cli.py` provides the `hermes-gemini-web-research` and `hgw-research` console commands.

The flow is:

1. A controller submits a `ResearchRequest` containing a question and optional angles.
2. Each angle becomes a `WorkerInput`.
3. The runner builds a strict JSON prompt and invokes Gemini CLI headlessly.
4. Gemini stdout is treated as either raw worker JSON or wrapper JSON containing model text plus telemetry.
5. Worker JSON is validated as `WorkerOutput`.
6. Worker results are reconciled into a `ResearchResult` with `synthesis_method="deterministic"`.
7. If requested through the Python API and a synthesizer is provided, semantic synthesis can refine the deterministic result. Failures fall back to the deterministic result and populate `synthesis_error`.
8. The CLI prints or writes Markdown or JSON.

## Quickstart

Install in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Run with default angles:

```bash
hgw-research "What are the current tradeoffs of using Gemini CLI for web research orchestration?"
```

Run with explicit angles:

```bash
hgw-research \
  "Should Hermes use Gemini CLI workers for current web research?" \
  --angle "Implementation: Check subprocess, JSON, timeout, and telemetry concerns" \
  --angle "Reliability: Look for failure modes and validation risks" \
  --angle "Operations: Assess costs, observability, and debugging workflow"
```

Run with an angle file:

```bash
hgw-research "What is the best MVP architecture?" --angle-file examples/angles.json
```

An angle file can be either a list:

```json
[
  {
    "name": "Current facts",
    "description": "Find the most current factual answer with primary sources."
  },
  {
    "name": "Evidence quality",
    "description": "Check source quality, conflicts, uncertainty, and caveats."
  }
]
```

Or an object:

```json
{
  "angles": [
    {
      "name": "Current facts",
      "description": "Find the most current factual answer with primary sources."
    }
  ]
}
```

Output JSON instead of Markdown:

```bash
hgw-research "Summarize the latest evidence on a topic" --format json
```

Write a report file instead of printing the rendered report to stdout:

```bash
hgw-research \
  "Summarize the latest evidence on a topic" \
  --format markdown \
  --output-file reports/research.md
```

When `--output-file` is used, parent directories are created automatically and the CLI prints the written path to stderr.

Customize the Gemini executable or flags. When omitted, runner arguments default to `--output-format json --approval-mode=yolo --prompt`, which asks Gemini CLI for wrapper JSON, auto-approves tools in headless mode, and passes the generated prompt as the `--prompt` value:

```bash
hgw-research \
  "Research this question" \
  --gemini-command gemini \
  --gemini-arg=--output-format \
  --gemini-arg=json \
  --gemini-arg=--approval-mode=yolo \
  --gemini-arg=--prompt \
  --timeout 180 \
  --max-concurrency 6
```

## Worker Contract

Every worker is instructed to return strict JSON matching `WorkerOutput`:

```json
{
  "angle_name": "Evidence quality",
  "answer": "Short answer from this angle.",
  "key_findings": ["Finding 1", "Finding 2"],
  "evidence": [
    {
      "type": "source",
      "claim": "Evidence-backed claim.",
      "source_title": "Source title",
      "url": "https://example.com",
      "quote": "Short quote when useful.",
      "published_date": "2026-04-19",
      "confidence": 0.8
    }
  ],
  "open_questions": ["What could not be verified?"],
  "confidence": 0.7
}
```

The runner accepts either direct worker JSON or wrapper JSON. Wrapper text can be supplied through fields such as `text`, `output`, `response`, or `stdout`; usage metadata can be supplied through `usage`, `usage_metadata`, `usageMetadata`, or `telemetry`.

Wrapper-level failures are treated as worker failures even when the subprocess exits 0. Supported error fields include `error`, `errors`, and failed `status` values.

ANSI escape sequences are stripped before wrapper and worker JSON parsing so colored CLI output does not poison validation.

## Synthesis

Deterministic reconciliation is always the baseline. It deduplicates findings by normalized text, keeps top evidence, and records worker failures or open questions as limitations. Every `ResearchResult` includes:

- `synthesis_method`: `deterministic` or `semantic`
- `synthesis_error`: the semantic synthesis error when fallback was needed, otherwise `null`

Built-in semantic synthesis is available through `GeminiSemanticSynthesizer`, which runs a second Gemini CLI pass over the deterministic report and refines the summary plus reconciled findings. The CLI can enable it directly:

```bash
hgw-research \
  "Should Hermes use Gemini CLI workers for current research?" \
  --semantic-synthesis \
  --format json \
  --output-file reports/semantic-research.json
```

Python API example:

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

`SemanticSynthesizer` is exported from the package root as a protocol for custom implementations. If semantic synthesis raises or returns invalid JSON, the orchestrator falls back to the deterministic result with `synthesis_method="deterministic"` and `synthesis_error` set.

## Hermes Integration

The repository includes a usable Hermes/Codex skill at `hermes/skills/gemini-web-research/SKILL.md`. It documents when to use the tool, CLI invocation patterns, output-file handling, semantic synthesis, and Python API behavior. Supporting angle-set guidance lives in `hermes/skills/gemini-web-research/references/angle-sets.md`.

## Reliability

`GeminiRunner` retries likely recoverable failures with bounded exponential backoff. Invalid worker JSON is retried because the same prompt can often recover on a second model call. Non-zero exits and wrapper-level errors are retried only when the error text looks transient, such as rate limits, 5xx responses, network failures, temporary unavailability, or overload.

The retry policy is configurable on the runner:

```python
runner = GeminiRunner(
    max_retries=2,
    initial_backoff_seconds=1.0,
    backoff_multiplier=2.0,
    max_backoff_seconds=10.0,
)
```

## CI

GitHub Actions runs `pytest` on Python 3.11 and 3.12 via `.github/workflows/ci.yml`.

## Current Limitations

- The baseline reconciliation step is still deterministic and intentionally simple. Semantic clustering is available through the optional second Gemini pass, but it adds extra latency and model calls.
- Gemini CLI flag conventions can still vary by wrapper. The default command shape is `gemini --output-format json --approval-mode=yolo --prompt "<prompt>"`; adjust `--gemini-command` and repeated `--gemini-arg` values as needed.
- Tests do not invoke Gemini CLI or any network service.

## Development

Useful commands:

```bash
python -m pip install -e ".[dev]"
pytest
hgw-research "Question" --format json
```

This repository targets Python 3.11+.
