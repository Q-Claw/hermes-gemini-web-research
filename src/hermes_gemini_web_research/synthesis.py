"""Optional Gemini-powered semantic synthesis for deterministic research results."""

from __future__ import annotations

import json
from typing import Protocol

from pydantic import ValidationError

from hermes_gemini_web_research.models import ResearchResult, SemanticSynthesisOutput
from hermes_gemini_web_research.prompts import build_semantic_synthesis_prompt, parse_json_model
from hermes_gemini_web_research.runner import GeminiRunner


class PromptRunner(Protocol):
    async def run_prompt(self, prompt: str, timeout_seconds: float) -> tuple[str, object | None]:
        ...


class GeminiSemanticSynthesizer:
    """Run a second Gemini pass to semantically merge deterministic findings."""

    def __init__(self, runner: PromptRunner | None = None, *, max_parse_retries: int = 1) -> None:
        self.runner = runner or GeminiRunner()
        self.max_parse_retries = max(0, max_parse_retries)

    async def synthesize(
        self,
        question: str,
        deterministic_result: ResearchResult,
        timeout_seconds: float,
    ) -> ResearchResult:
        prompt = build_semantic_synthesis_prompt(question, deterministic_result)

        last_error: Exception | None = None
        for _ in range(self.max_parse_retries + 1):
            raw_text, _usage = await self.runner.run_prompt(prompt, timeout_seconds)
            try:
                parsed = parse_json_model(raw_text, SemanticSynthesisOutput)
            except (ValueError, ValidationError, json.JSONDecodeError) as exc:
                last_error = exc
                continue

            return deterministic_result.model_copy(
                update={
                    "summary": parsed.summary,
                    "findings": parsed.findings,
                    "synthesis_method": "semantic",
                    "synthesis_error": None,
                },
                deep=True,
            )

        assert last_error is not None
        raise RuntimeError(f"invalid semantic synthesis JSON: {last_error}")
