"""Simple deterministic reconciliation for worker outputs."""

from __future__ import annotations

from collections import OrderedDict

from hermes_gemini_web_research.models import (
    EvidenceItem,
    ReconciledFinding,
    ResearchResult,
    WorkerResult,
    WorkerStatus,
)


def reconcile(question: str, worker_results: list[WorkerResult]) -> ResearchResult:
    """Synthesize worker outputs into a conservative structured result."""

    succeeded = [result for result in worker_results if result.status == WorkerStatus.SUCCEEDED and result.output]
    failed = [result for result in worker_results if result.status != WorkerStatus.SUCCEEDED]

    if not succeeded:
        return ResearchResult(
            question=question,
            status="failed",
            summary="No worker returned valid research JSON.",
            worker_results=worker_results,
            limitations=[result.error or f"{result.angle.name} failed" for result in failed],
        )

    findings_by_text: OrderedDict[str, ReconciledFinding] = OrderedDict()
    for result in succeeded:
        assert result.output is not None
        angle_name = result.output.angle_name or result.angle.name
        for finding_text in result.output.key_findings or [result.output.answer]:
            normalized = " ".join(finding_text.strip().split())
            if not normalized:
                continue
            finding = findings_by_text.setdefault(
                normalized.lower(),
                ReconciledFinding(finding=normalized),
            )
            if angle_name not in finding.supporting_angles:
                finding.supporting_angles.append(angle_name)
            finding.evidence.extend(_top_evidence(result.output.evidence))

    summary = _build_summary(succeeded)
    limitations = []
    for result in succeeded:
        assert result.output is not None
        limitations.extend(result.output.open_questions)
    limitations.extend(result.error or f"{result.angle.name} failed" for result in failed)

    return ResearchResult(
        question=question,
        status="partial" if failed else "complete",
        summary=summary,
        findings=list(findings_by_text.values()),
        worker_results=worker_results,
        limitations=limitations,
    )


def _top_evidence(evidence: list[EvidenceItem]) -> list[EvidenceItem]:
    return sorted(evidence, key=lambda item: item.confidence, reverse=True)[:3]


def _build_summary(succeeded: list[WorkerResult]) -> str:
    answers = []
    for result in succeeded:
        assert result.output is not None
        answers.append(f"{result.output.angle_name}: {result.output.answer}")
    return "\n".join(answers)

