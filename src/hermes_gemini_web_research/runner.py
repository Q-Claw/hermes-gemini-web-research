"""Subprocess runner for Gemini CLI workers."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from hermes_gemini_web_research.models import (
    GeminiWrapperOutput,
    ResearchAngle,
    WorkerInput,
    WorkerResult,
    WorkerStatus,
)
from hermes_gemini_web_research.prompts import build_worker_prompt, parse_worker_output, strip_ansi_escape_sequences


@dataclass(frozen=True)
class _AttemptFailure:
    status: WorkerStatus
    error: str
    raw_text: str | None = None
    usage: object | None = None


class GeminiRunner:
    """Run Gemini CLI headlessly and validate strict worker JSON."""

    def __init__(
        self,
        command: str = "gemini",
        *,
        args: list[str] | None = None,
        cwd: Path | None = None,
        max_retries: int = 2,
        initial_backoff_seconds: float = 1.0,
        backoff_multiplier: float = 2.0,
        max_backoff_seconds: float = 10.0,
    ) -> None:
        self.command = command
        self.args = args or ["--output-format", "json", "--approval-mode=yolo", "--prompt"]
        self.cwd = cwd
        self.max_retries = max(0, max_retries)
        self.initial_backoff_seconds = max(0.0, initial_backoff_seconds)
        self.backoff_multiplier = max(1.0, backoff_multiplier)
        self.max_backoff_seconds = max(0.0, max_backoff_seconds)

    async def run(self, worker_input: WorkerInput, timeout_seconds: float) -> WorkerResult:
        """Run one worker and return a validated result or structured failure."""

        started = datetime.now(timezone.utc)
        started_monotonic = time.monotonic()
        prompt = build_worker_prompt(worker_input)

        last_failure: _AttemptFailure | None = None
        for attempt in range(self.max_retries + 1):
            result = await self._run_attempt(worker_input, prompt, timeout_seconds, started, started_monotonic)
            if isinstance(result, WorkerResult):
                return result

            last_failure = result
            if attempt >= self.max_retries or not self._should_retry(result):
                break

            await asyncio.sleep(self._backoff_seconds(attempt))

        assert last_failure is not None
        attempts = attempt + 1
        suffix = f" after {attempts} attempts" if attempts > 1 else ""
        return self._failure(
            worker_input.angle,
            last_failure.status,
            f"{last_failure.error}{suffix}",
            started,
            started_monotonic,
            raw_text=last_failure.raw_text,
            usage=last_failure.usage,
        )

    async def _run_attempt(
        self,
        worker_input: WorkerInput,
        prompt: str,
        timeout_seconds: float,
        started: datetime,
        started_monotonic: float,
    ) -> WorkerResult | _AttemptFailure:
        proc: asyncio.subprocess.Process | None = None

        try:
            proc = await asyncio.create_subprocess_exec(
                self.command,
                *self.args,
                prompt,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            if proc is not None and proc.returncode is None:
                proc.kill()
                await proc.wait()
            return _AttemptFailure(
                WorkerStatus.TIMED_OUT,
                "Gemini CLI worker timed out",
            )
        except OSError as exc:
            return _AttemptFailure(
                WorkerStatus.FAILED,
                f"failed to start Gemini CLI: {exc}",
            )

        stdout = strip_ansi_escape_sequences(stdout_bytes.decode("utf-8", errors="replace"))
        stderr = strip_ansi_escape_sequences(stderr_bytes.decode("utf-8", errors="replace"))
        if proc.returncode != 0:
            return _AttemptFailure(
                WorkerStatus.FAILED,
                f"Gemini CLI exited with {proc.returncode}: {stderr.strip()}",
                raw_text=stdout,
            )

        text, usage, wrapper_error = self._unwrap_with_error(stdout)
        if wrapper_error:
            error = f"Gemini CLI wrapper returned an error: {wrapper_error}"
            if stderr.strip():
                error = f"{error}; stderr: {stderr.strip()}"
            return _AttemptFailure(
                WorkerStatus.FAILED,
                error,
                raw_text=text,
                usage=usage,
            )

        try:
            output = parse_worker_output(text)
        except (ValueError, ValidationError, json.JSONDecodeError) as exc:
            return _AttemptFailure(
                WorkerStatus.INVALID_JSON,
                f"invalid worker JSON: {exc}",
                raw_text=text,
                usage=usage,
            )

        completed = datetime.now(timezone.utc)
        return WorkerResult(
            angle=worker_input.angle,
            status=WorkerStatus.SUCCEEDED,
            output=output,
            raw_text=text,
            started_at=started,
            completed_at=completed,
            duration_seconds=round(time.monotonic() - started_monotonic, 3),
            usage=usage,
        )

    def _unwrap(self, stdout: str):
        text, usage, _ = self._unwrap_with_error(stdout)
        return text, usage

    def _unwrap_with_error(self, stdout: str):
        stdout = strip_ansi_escape_sequences(stdout)
        stripped = stdout.strip()
        if not stripped.startswith("{"):
            return stdout, None, None
        try:
            wrapper = GeminiWrapperOutput.model_validate_json(stripped)
        except ValidationError:
            return stdout, None, None
        text = wrapper.best_text()
        usage = wrapper.best_usage()
        error = wrapper.best_error()
        if text is None:
            return stdout, usage, error
        return text, usage, error

    def _backoff_seconds(self, attempt: int) -> float:
        delay = self.initial_backoff_seconds * (self.backoff_multiplier**attempt)
        return min(delay, self.max_backoff_seconds)

    @staticmethod
    def _should_retry(failure: _AttemptFailure) -> bool:
        if failure.status == WorkerStatus.INVALID_JSON:
            return True
        if failure.status != WorkerStatus.FAILED:
            return False

        error = failure.error.lower()
        transient_markers = (
            "429",
            "500",
            "502",
            "503",
            "504",
            "temporarily unavailable",
            "try again",
            "rate limit",
            "rate-limit",
            "timeout",
            "timed out",
            "connection reset",
            "connection aborted",
            "network",
            "econnreset",
            "etimedout",
            "eai_again",
            "unavailable",
            "overloaded",
        )
        return any(marker in error for marker in transient_markers)

    @staticmethod
    def _failure(
        angle: ResearchAngle,
        status: WorkerStatus,
        error: str,
        started: datetime,
        started_monotonic: float,
        *,
        raw_text: str | None = None,
        usage=None,
    ) -> WorkerResult:
        return WorkerResult(
            angle=angle,
            status=status,
            error=error,
            raw_text=raw_text,
            started_at=started,
            completed_at=datetime.now(timezone.utc),
            duration_seconds=round(time.monotonic() - started_monotonic, 3),
            usage=usage,
        )
