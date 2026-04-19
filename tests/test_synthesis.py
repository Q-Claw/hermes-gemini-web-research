from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from hermes_gemini_web_research.models import (
    ResearchAngle,
    ReconciledFinding,
    ResearchResult,
    SemanticSynthesisOutput,
    WorkerOutput,
    WorkerResult,
    WorkerStatus,
)
from hermes_gemini_web_research.prompts import parse_json_model
from hermes_gemini_web_research.synthesis import GeminiSemanticSynthesizer


class FakePromptRunner:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[tuple[str, float]] = []

    async def run_prompt(self, prompt: str, timeout_seconds: float):
        self.calls.append((prompt, timeout_seconds))
        return self.response_text, None


def make_result() -> ResearchResult:
    return ResearchResult(
        question="Should Hermes use Gemini CLI workers?",
        status="complete",
        summary="Benefits: Faster research\nRisks: Needs validation",
        findings=[
            ReconciledFinding(
                finding="Gemini CLI can research in parallel",
                supporting_angles=["Benefits"],
                evidence=[],
            )
        ],
        worker_results=[
            WorkerResult(
                angle=ResearchAngle(name="Benefits", description="Assess upside."),
                status=WorkerStatus.SUCCEEDED,
                output=WorkerOutput(
                    angle_name="Benefits",
                    answer="Faster research",
                    key_findings=["Gemini CLI can research in parallel"],
                    confidence=0.8,
                ),
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                duration_seconds=0.1,
            )
        ],
        limitations=["Need more validation"],
        synthesis_method="deterministic",
    )


@pytest.mark.asyncio
async def test_semantic_synthesizer_refines_summary_and_findings():
    runner = FakePromptRunner(
        json.dumps(
            {
                "summary": "Semantic summary",
                "findings": [
                    {
                        "finding": "Merged semantic finding",
                        "supporting_angles": ["Benefits", "Risks"],
                        "evidence": [],
                    }
                ],
            }
        )
    )
    synthesizer = GeminiSemanticSynthesizer(runner=runner)

    result = await synthesizer.synthesize(
        "Should Hermes use Gemini CLI workers?",
        make_result(),
        timeout_seconds=30,
    )

    assert result.summary == "Semantic summary"
    assert result.findings[0].finding == "Merged semantic finding"
    assert result.worker_results[0].angle.name == "Benefits"
    assert result.limitations == ["Need more validation"]
    assert runner.calls
    assert "Gemini CLI can research in parallel" in runner.calls[0][0]
    assert "Should Hermes use Gemini CLI workers?" in runner.calls[0][0]


def test_semantic_synthesis_output_rejects_extra_keys():
    raw = json.dumps(
        {
            "summary": "Semantic summary",
            "findings": [],
            "extra": "should fail",
        }
    )

    with pytest.raises(ValidationError):
        parse_json_model(raw, SemanticSynthesisOutput)
