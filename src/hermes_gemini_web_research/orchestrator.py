"""Concurrent orchestration for Gemini CLI research workers."""

from __future__ import annotations

import asyncio
from typing import Protocol

from hermes_gemini_web_research.models import (
    ResearchAngle,
    ResearchRequest,
    ResearchResult,
    WorkerInput,
    WorkerResult,
)
from hermes_gemini_web_research.reconcile import reconcile
from hermes_gemini_web_research.runner import GeminiRunner


DEFAULT_ANGLES = [
    ResearchAngle(name="Current facts", description="Find the most current factual answer with primary sources."),
    ResearchAngle(name="Evidence quality", description="Check source quality, conflicts, uncertainty, and caveats."),
    ResearchAngle(name="Practical implications", description="Explain what the findings mean for a decision maker."),
]


class WorkerRunner(Protocol):
    """Minimal runner protocol used by orchestration and tests."""

    async def run(self, worker_input: WorkerInput, timeout_seconds: float) -> WorkerResult:
        ...


class ResearchOrchestrator:
    """Run research angles concurrently and reconcile validated worker outputs."""

    def __init__(self, runner: WorkerRunner | None = None) -> None:
        self.runner = runner or GeminiRunner()

    async def run(self, request: ResearchRequest) -> ResearchResult:
        angles = request.angles or DEFAULT_ANGLES
        semaphore = asyncio.Semaphore(request.max_concurrency)

        async def run_angle(angle: ResearchAngle) -> WorkerResult:
            async with semaphore:
                return await self.runner.run(
                    WorkerInput(question=request.question, angle=angle),
                    timeout_seconds=request.timeout_seconds,
                )

        worker_results = await asyncio.gather(*(run_angle(angle) for angle in angles))
        return reconcile(request.question, list(worker_results))

