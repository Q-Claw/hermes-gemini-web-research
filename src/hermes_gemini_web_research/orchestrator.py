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


class SemanticSynthesizer(Protocol):
    """Optional second-stage synthesizer for refining deterministic reconciliation."""

    async def synthesize(
        self,
        question: str,
        deterministic_result: ResearchResult,
        timeout_seconds: float,
    ) -> ResearchResult:
        ...


class ResearchOrchestrator:
    """Run research angles concurrently and reconcile validated worker outputs."""

    def __init__(
        self,
        runner: WorkerRunner | None = None,
        *,
        synthesizer: SemanticSynthesizer | None = None,
    ) -> None:
        self.runner = runner or GeminiRunner()
        self.synthesizer = synthesizer

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
        deterministic_result = reconcile(request.question, list(worker_results))

        if not request.semantic_synthesis or self.synthesizer is None:
            return deterministic_result

        try:
            result = await self.synthesizer.synthesize(
                request.question,
                deterministic_result.model_copy(deep=True),
                request.timeout_seconds,
            )
        except Exception as exc:
            deterministic_result.synthesis_method = "deterministic"
            deterministic_result.synthesis_error = str(exc)
            return deterministic_result

        result.synthesis_method = "semantic"
        result.synthesis_error = None
        return result
