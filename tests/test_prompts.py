import json
import sys

import pytest

from hermes_gemini_web_research.models import ResearchAngle, WorkerInput, WorkerStatus
from hermes_gemini_web_research.prompts import (
    build_worker_prompt,
    extract_json_object,
    parse_worker_output,
    strip_ansi_escape_sequences,
)
from hermes_gemini_web_research.runner import GeminiRunner


def test_worker_prompt_demands_strict_json_and_schema():
    prompt = build_worker_prompt(
        WorkerInput(
            question="What changed in Python packaging?",
            angle=ResearchAngle(name="Standards", description="Look for PEP and PyPA guidance."),
        )
    )

    assert "Return STRICT JSON only" in prompt
    assert "Required JSON schema:" in prompt
    assert "WorkerOutput" in prompt
    assert "What changed in Python packaging?" in prompt


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"angle_name":"A","answer":"B"}', '{"angle_name":"A","answer":"B"}'),
        ('```json\n{"angle_name":"A","answer":"B"}\n```', '{"angle_name":"A","answer":"B"}'),
        ('prefix {"angle_name":"A","answer":"B"} suffix', '{"angle_name":"A","answer":"B"}'),
    ],
)
def test_extract_json_object(raw, expected):
    assert extract_json_object(raw) == expected


def test_extract_json_object_strips_ansi_sequences():
    raw = '\x1b[32m{"angle_name":"A","answer":"B"}\x1b[0m'

    assert strip_ansi_escape_sequences(raw) == '{"angle_name":"A","answer":"B"}'
    assert extract_json_object(raw) == '{"angle_name":"A","answer":"B"}'


def test_parse_worker_output_validates_json():
    raw = json.dumps(
        {
            "angle_name": "Facts",
            "answer": "A short answer.",
            "key_findings": ["One finding"],
            "evidence": [
                {
                    "type": "source",
                    "claim": "The source supports the finding.",
                    "source_title": "Example",
                    "url": "https://example.com/report",
                    "confidence": 0.8,
                }
            ],
            "confidence": 0.7,
        }
    )

    parsed = parse_worker_output(raw)

    assert parsed.angle_name == "Facts"
    assert parsed.evidence[0].url.unicode_string() == "https://example.com/report"


def test_runner_unwraps_wrapper_json_and_usage_metadata():
    stdout = json.dumps(
        {
            "text": '{"angle_name":"Facts","answer":"A"}',
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 20,
                "totalTokenCount": 30,
                "latencyMs": 125.5,
            },
        }
    )

    text, usage = GeminiRunner()._unwrap(stdout)

    assert text == '{"angle_name":"Facts","answer":"A"}'
    assert usage is not None
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 20
    assert usage.total_tokens == 30
    assert usage.latency_ms == 125.5


def test_runner_default_args_request_json_and_auto_approve_tools():
    assert GeminiRunner().args == ["--output-format", "json", "--approval-mode=yolo", "--prompt"]


def test_runner_detects_wrapper_level_error():
    stdout = json.dumps({"error": {"message": "rate limit exceeded"}, "stdout": ""})

    text, usage, error = GeminiRunner()._unwrap_with_error(stdout)

    assert text == stdout
    assert usage is None
    assert "rate limit exceeded" in error


@pytest.mark.asyncio
async def test_runner_retries_invalid_json_and_recovers(tmp_path):
    counter = tmp_path / "attempts.txt"
    script = """
import json
import pathlib
import sys

counter = pathlib.Path(sys.argv[1])
attempts = int(counter.read_text() if counter.exists() else "0")
counter.write_text(str(attempts + 1))

if attempts == 0:
    print("\\x1b[31mnot json\\x1b[0m")
else:
    print(json.dumps({"text": json.dumps({"angle_name": "Retry", "answer": "Recovered"})}))
"""
    runner = GeminiRunner(
        command=sys.executable,
        args=["-c", script, str(counter), "--"],
        max_retries=1,
        initial_backoff_seconds=0,
    )

    result = await runner.run(
        WorkerInput(
            question="Can retries recover?",
            angle=ResearchAngle(name="Retry", description="Exercise retry path."),
        ),
        timeout_seconds=5,
    )

    assert result.status == WorkerStatus.SUCCEEDED
    assert result.output is not None
    assert result.output.answer == "Recovered"
    assert counter.read_text() == "2"
