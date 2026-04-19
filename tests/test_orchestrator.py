from datetime import datetime, timezone

import pytest

from hermes_gemini_web_research.models import (
    ResearchAngle,
    ResearchRequest,
    WorkerInput,
    WorkerOutput,
    WorkerResult,
    WorkerStatus,
)
from hermes_gemini_web_research.orchestrator import ResearchOrchestrator


class MockRunner:
    def __init__(self) -> None:
        self.calls: list[WorkerInput] = []

    async def run(self, worker_input: WorkerInput, timeout_seconds: float) -> WorkerResult:
        self.calls.append(worker_input)
        return WorkerResult(
            angle=worker_input.angle,
            status=WorkerStatus.SUCCEEDED,
            output=WorkerOutput(
                angle_name=worker_input.angle.name,
                answer=f"{worker_input.angle.name} answer",
                key_findings=[f"{worker_input.angle.name} finding"],
                confidence=0.75,
            ),
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            duration_seconds=0.01,
        )


class MockSynthesizer:
    def __init__(self, *, should_fail: bool = False, mutate_before_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.mutate_before_fail = mutate_before_fail
        self.calls = 0

    async def synthesize(self, question, deterministic_result, timeout_seconds: float):
        self.calls += 1
        if self.should_fail:
            if self.mutate_before_fail:
                deterministic_result.summary = "Mutated before failure"
            raise RuntimeError("semantic synthesis unavailable")

        synthesized = deterministic_result.model_copy(deep=True)
        synthesized.summary = "Semantic summary"
        synthesized.findings[0].finding = "Merged semantic finding"
        synthesized.synthesis_method = "semantic"
        synthesized.synthesis_error = None
        return synthesized


@pytest.mark.asyncio
async def test_orchestrator_runs_angles_and_reconciles():
    runner = MockRunner()
    request = ResearchRequest(
        question="Should we use Gemini CLI for research workers?",
        angles=[
            ResearchAngle(name="Benefits", description="Assess upside."),
            ResearchAngle(name="Risks", description="Assess failure modes."),
        ],
        max_concurrency=2,
    )

    result = await ResearchOrchestrator(runner).run(request)

    assert [call.angle.name for call in runner.calls] == ["Benefits", "Risks"]
    assert result.status == "complete"
    assert len(result.findings) == 2
    assert "Benefits answer" in result.summary
    assert result.synthesis_method == "deterministic"


@pytest.mark.asyncio
async def test_orchestrator_can_apply_semantic_synthesis():
    runner = MockRunner()
    synthesizer = MockSynthesizer()
    request = ResearchRequest(
        question="Should we use Gemini CLI for research workers?",
        angles=[ResearchAngle(name="Benefits", description="Assess upside.")],
        semantic_synthesis=True,
    )

    result = await ResearchOrchestrator(runner, synthesizer=synthesizer).run(request)

    assert synthesizer.calls == 1
    assert result.synthesis_method == "semantic"
    assert result.summary == "Semantic summary"
    assert result.findings[0].finding == "Merged semantic finding"


@pytest.mark.asyncio
async def test_orchestrator_keeps_deterministic_result_without_synthesizer():
    runner = MockRunner()
    request = ResearchRequest(
        question="Should we use Gemini CLI for research workers?",
        angles=[ResearchAngle(name="Benefits", description="Assess upside.")],
        semantic_synthesis=True,
    )

    result = await ResearchOrchestrator(runner).run(request)

    assert result.synthesis_method == "deterministic"
    assert result.synthesis_error is None
    assert result.summary == "Benefits: Benefits answer"


@pytest.mark.asyncio
async def test_orchestrator_falls_back_when_semantic_synthesis_fails():
    runner = MockRunner()
    synthesizer = MockSynthesizer(should_fail=True)
    request = ResearchRequest(
        question="Should we use Gemini CLI for research workers?",
        angles=[ResearchAngle(name="Benefits", description="Assess upside.")],
        semantic_synthesis=True,
    )

    result = await ResearchOrchestrator(runner, synthesizer=synthesizer).run(request)

    assert synthesizer.calls == 1
    assert result.synthesis_method == "deterministic"
    assert result.synthesis_error == "semantic synthesis unavailable"
    assert result.summary == "Benefits: Benefits answer"


@pytest.mark.asyncio
async def test_orchestrator_fallback_preserves_deterministic_result_after_mutation():
    runner = MockRunner()
    synthesizer = MockSynthesizer(should_fail=True, mutate_before_fail=True)
    request = ResearchRequest(
        question="Should we use Gemini CLI for research workers?",
        angles=[ResearchAngle(name="Benefits", description="Assess upside.")],
        semantic_synthesis=True,
    )

    result = await ResearchOrchestrator(runner, synthesizer=synthesizer).run(request)

    assert result.synthesis_method == "deterministic"
    assert result.synthesis_error == "semantic synthesis unavailable"
    assert result.summary == "Benefits: Benefits answer"
