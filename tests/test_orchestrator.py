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

