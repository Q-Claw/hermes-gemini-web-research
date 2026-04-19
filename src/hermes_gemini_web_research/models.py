"""Pydantic models for research orchestration and Gemini CLI wrapper output."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


class EvidenceType(str, Enum):
    """Supported evidence categories produced by workers."""

    SOURCE = "source"
    QUOTE = "quote"
    STATISTIC = "statistic"
    CLAIM = "claim"
    CAVEAT = "caveat"


class ResearchAngle(BaseModel):
    """A focused perspective that one Gemini CLI worker should investigate."""

    name: str = Field(..., min_length=1, max_length=80)
    description: str = Field(..., min_length=1, max_length=500)


class ResearchRequest(BaseModel):
    """Top-level controller request."""

    question: str = Field(..., min_length=1)
    angles: list[ResearchAngle] = Field(default_factory=list)
    timeout_seconds: float = Field(default=120.0, gt=0)
    max_concurrency: int = Field(default=4, ge=1, le=12)
    semantic_synthesis: bool = Field(
        default=False,
        description="When true, allow an optional semantic synthesizer to refine deterministic reconciliation.",
    )

    @field_validator("angles")
    @classmethod
    def require_unique_angle_names(cls, angles: list[ResearchAngle]) -> list[ResearchAngle]:
        names = [angle.name.lower() for angle in angles]
        if len(names) != len(set(names)):
            raise ValueError("angle names must be unique")
        return angles


class WorkerInput(BaseModel):
    """Input passed to one Gemini CLI worker."""

    question: str = Field(..., min_length=1)
    angle: ResearchAngle
    output_schema_name: str = "WorkerOutput"


class EvidenceItem(BaseModel):
    """One piece of evidence from a worker."""

    type: EvidenceType
    claim: str = Field(..., min_length=1)
    source_title: str | None = None
    url: HttpUrl | None = None
    quote: str | None = None
    published_date: str | None = None
    confidence: float = Field(default=0.5, ge=0, le=1)


class WorkerOutput(BaseModel):
    """Strict JSON contract requested from Gemini CLI."""

    angle_name: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    key_findings: list[str] = Field(default_factory=list, max_length=12)
    evidence: list[EvidenceItem] = Field(default_factory=list, max_length=30)
    open_questions: list[str] = Field(default_factory=list, max_length=12)
    confidence: float = Field(default=0.5, ge=0, le=1)


class GeminiUsageMetadata(BaseModel):
    """Token and latency details extracted from wrapper JSON when present."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    prompt_tokens: int | None = Field(
        default=None,
        validation_alias=AliasChoices("prompt_tokens", "promptTokenCount", "input_tokens", "inputTokenCount"),
    )
    completion_tokens: int | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "completion_tokens",
            "candidatesTokenCount",
            "output_tokens",
            "outputTokenCount",
        ),
    )
    total_tokens: int | None = Field(
        default=None,
        validation_alias=AliasChoices("total_tokens", "totalTokenCount"),
    )
    latency_ms: float | None = Field(
        default=None,
        validation_alias=AliasChoices("latency_ms", "latencyMs"),
    )
    model: str | None = None


class GeminiWrapperOutput(BaseModel):
    """Flexible parser for common Gemini CLI wrapper JSON envelopes."""

    model_config = ConfigDict(extra="allow")

    error: str | dict[str, Any] | list[Any] | None = None
    errors: list[Any] | None = None
    text: str | None = None
    output: str | None = None
    response: str | dict[str, Any] | None = None
    stdout: str | None = None
    stderr: str | None = None
    status: str | None = None
    usage: GeminiUsageMetadata | None = None
    usage_metadata: GeminiUsageMetadata | None = Field(
        default=None,
        validation_alias=AliasChoices("usage_metadata", "usageMetadata"),
    )
    telemetry: GeminiUsageMetadata | None = None
    exit_code: int | None = None

    def best_text(self) -> str | None:
        """Return the most likely model text field from a wrapper object."""

        for value in (self.text, self.output, self.stdout):
            if value:
                return value
        if isinstance(self.response, str):
            return self.response
        if isinstance(self.response, dict):
            for key in ("text", "output", "content"):
                value = self.response.get(key)
                if isinstance(value, str) and value:
                    return value
        return None

    def best_usage(self) -> GeminiUsageMetadata | None:
        """Return normalized usage metadata from supported wrapper fields."""

        return self.usage or self.usage_metadata or self.telemetry

    def best_error(self) -> str | None:
        """Return a readable wrapper-level error when one is present."""

        if self.error:
            return str(self.error)
        if self.errors:
            return "; ".join(str(error) for error in self.errors)
        if isinstance(self.response, dict):
            for key in ("error", "errors"):
                value = self.response.get(key)
                if value:
                    return str(value)
        if self.status and self.status.lower() in {"error", "failed", "failure"}:
            return self.stderr or f"wrapper status is {self.status}"
        return None


class WorkerStatus(str, Enum):
    """Execution status for a worker."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    INVALID_JSON = "invalid_json"


class WorkerResult(BaseModel):
    """Validated or failed result from one worker."""

    angle: ResearchAngle
    status: WorkerStatus
    output: WorkerOutput | None = None
    raw_text: str | None = None
    error: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    duration_seconds: float | None = None
    usage: GeminiUsageMetadata | None = None


class ReconciledFinding(BaseModel):
    """Deduplicated finding with supporting evidence."""

    finding: str
    supporting_angles: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    source_count: int = Field(default=0, ge=0)
    source_diversity: float = Field(default=0.0, ge=0, le=1)
    consensus_score: float = Field(default=0.0, ge=0, le=1)
    confidence: Literal["high", "medium", "low"] = "low"
    severity: Literal["high", "medium", "low"] = "low"
    best_evidence: EvidenceItem | None = None
    best_evidence_score: float | None = Field(default=None, ge=0, le=1)
    contradicts: list[str] = Field(default_factory=list)


class ReportSummary(BaseModel):
    """Compact report-level severity and confidence counts."""

    finding_count: int = Field(default=0, ge=0)
    confidence_counts: dict[Literal["high", "medium", "low"], int] = Field(
        default_factory=lambda: {"high": 0, "medium": 0, "low": 0}
    )
    severity_counts: dict[Literal["high", "medium", "low"], int] = Field(
        default_factory=lambda: {"high": 0, "medium": 0, "low": 0}
    )
    contradiction_count: int = Field(default=0, ge=0)


class ResearchResult(BaseModel):
    """Final structured synthesis."""

    question: str
    status: Literal["complete", "partial", "failed"]
    summary: str
    findings: list[ReconciledFinding] = Field(default_factory=list)
    report_summary: ReportSummary = Field(default_factory=ReportSummary)
    worker_results: list[WorkerResult] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    synthesis_method: Literal["deterministic", "semantic"] = "deterministic"
    synthesis_error: str | None = None

    @model_validator(mode="after")
    def populate_report_summary(self) -> "ResearchResult":
        self.report_summary = build_report_summary(self.findings)
        return self


class SemanticSynthesisOutput(BaseModel):
    """Structured output requested from the optional semantic synthesis pass."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(..., min_length=1)
    findings: list[ReconciledFinding] = Field(default_factory=list)


def build_report_summary(findings: list[ReconciledFinding]) -> ReportSummary:
    """Build deterministic report-level counts from reconciled findings."""

    confidence_counts: dict[Literal["high", "medium", "low"], int] = {"high": 0, "medium": 0, "low": 0}
    severity_counts: dict[Literal["high", "medium", "low"], int] = {"high": 0, "medium": 0, "low": 0}
    contradiction_pairs: set[tuple[str, str]] = set()

    for finding in findings:
        confidence_counts[finding.confidence] += 1
        severity_counts[finding.severity] += 1
        for contradiction in finding.contradicts:
            contradiction_pairs.add(tuple(sorted((finding.finding, contradiction))))

    return ReportSummary(
        finding_count=len(findings),
        confidence_counts=confidence_counts,
        severity_counts=severity_counts,
        contradiction_count=len(contradiction_pairs),
    )
