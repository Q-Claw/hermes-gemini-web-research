from __future__ import annotations

from datetime import datetime, timezone

from hermes_gemini_web_research.models import EvidenceItem, ResearchAngle, WorkerOutput, WorkerResult, WorkerStatus
from hermes_gemini_web_research.reconcile import reconcile


def make_worker_result(
    *,
    angle_name: str,
    answer: str,
    key_findings: list[str],
    evidence: list[EvidenceItem],
) -> WorkerResult:
    return WorkerResult(
        angle=ResearchAngle(name=angle_name, description=f"{angle_name} description"),
        status=WorkerStatus.SUCCEEDED,
        output=WorkerOutput(
            angle_name=angle_name,
            answer=answer,
            key_findings=key_findings,
            evidence=evidence,
            confidence=0.8,
        ),
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        duration_seconds=0.01,
    )


def test_reconcile_clusters_semantically_similar_findings():
    result = reconcile(
        "Should Hermes use Gemini CLI workers?",
        [
            make_worker_result(
                angle_name="Benefits",
                answer="Parallel research is possible.",
                key_findings=["Gemini CLI enables parallel web research workflows"],
                evidence=[],
            ),
            make_worker_result(
                angle_name="Operations",
                answer="Parallel research is useful.",
                key_findings=["Parallel web research workflows are enabled by Gemini CLI"],
                evidence=[],
            ),
        ],
    )

    assert len(result.findings) == 1
    assert set(result.findings[0].supporting_angles) == {"Benefits", "Operations"}


def test_reconcile_clustering_is_transitive_for_close_paraphrases():
    result = reconcile(
        "Should Hermes use Gemini CLI workers?",
        [
            make_worker_result(
                angle_name="Benefits",
                answer="Parallel research is possible.",
                key_findings=["Gemini CLI enables parallel web research workflows"],
                evidence=[],
            ),
            make_worker_result(
                angle_name="Operations",
                answer="Parallel research is useful.",
                key_findings=["Parallel research workflows are enabled by Gemini CLI"],
                evidence=[],
            ),
            make_worker_result(
                angle_name="Evidence quality",
                answer="Parallel research is documented.",
                key_findings=["Gemini CLI enables web research workflows in parallel"],
                evidence=[],
            ),
        ],
    )

    assert len(result.findings) == 1
    assert set(result.findings[0].supporting_angles) == {"Benefits", "Operations", "Evidence quality"}


def test_reconcile_ranks_and_deduplicates_evidence_items():
    high_quality_duplicate = EvidenceItem(
        type="source",
        claim="Gemini CLI supports structured JSON output.",
        source_title="Official docs",
        url="https://example.com/docs",
        quote="Use --output-format json for JSON output.",
        published_date="2026-04-19",
        confidence=0.82,
    )
    low_quality_duplicate = EvidenceItem(
        type="source",
        claim="Gemini CLI supports structured JSON output.",
        source_title="Forum post",
        confidence=0.91,
    )
    strong_secondary = EvidenceItem(
        type="source",
        claim="Gemini CLI can auto-approve tools in yolo mode.",
        source_title="CLI reference",
        url="https://example.com/cli",
        confidence=0.8,
    )

    result = reconcile(
        "How reliable is Gemini CLI for orchestration?",
        [
            make_worker_result(
                angle_name="Current facts",
                answer="It supports JSON output.",
                key_findings=["Gemini CLI supports structured JSON output"],
                evidence=[low_quality_duplicate, strong_secondary],
            ),
            make_worker_result(
                angle_name="Evidence quality",
                answer="Docs confirm JSON output.",
                key_findings=["Gemini CLI supports structured JSON output"],
                evidence=[high_quality_duplicate],
            ),
        ],
    )

    evidence = result.findings[0].evidence

    assert len(evidence) == 3
    assert evidence[0].source_title == "Official docs"
    assert evidence[0].url is not None
    assert evidence[1].claim == "Gemini CLI can auto-approve tools in yolo mode."
    assert evidence[2].source_title == "Forum post"


def test_reconcile_deduplicates_evidence_despite_punctuation_variation():
    result = reconcile(
        "How reliable is Gemini CLI for orchestration?",
        [
            make_worker_result(
                angle_name="Current facts",
                answer="It supports JSON output.",
                key_findings=["Gemini CLI supports structured JSON output"],
                evidence=[
                    EvidenceItem(
                        type="source",
                        claim="Gemini CLI supports structured JSON output.",
                        source_title="Official docs",
                        url="https://example.com/docs",
                        confidence=0.82,
                    )
                ],
            ),
            make_worker_result(
                angle_name="Evidence quality",
                answer="Docs confirm JSON output.",
                key_findings=["Gemini CLI supports structured JSON output"],
                evidence=[
                    EvidenceItem(
                        type="source",
                        claim="Gemini CLI supports structured JSON output",
                        source_title="Official docs",
                        url="https://example.com/docs",
                        confidence=0.75,
                    )
                ],
            ),
        ],
    )

    assert len(result.findings[0].evidence) == 1


def test_reconcile_keeps_distinct_sources_for_same_claim():
    result = reconcile(
        "How reliable is Gemini CLI for orchestration?",
        [
            make_worker_result(
                angle_name="Current facts",
                answer="It supports JSON output.",
                key_findings=["Gemini CLI supports structured JSON output"],
                evidence=[
                    EvidenceItem(
                        type="source",
                        claim="Gemini CLI supports structured JSON output.",
                        source_title="Official docs",
                        url="https://example.com/docs",
                        confidence=0.82,
                    )
                ],
            ),
            make_worker_result(
                angle_name="Evidence quality",
                answer="Docs confirm JSON output.",
                key_findings=["Gemini CLI supports structured JSON output"],
                evidence=[
                    EvidenceItem(
                        type="source",
                        claim="Gemini CLI supports structured JSON output",
                        source_title="Independent analysis",
                        url="https://example.com/analysis",
                        confidence=0.75,
                    )
                ],
            ),
        ],
    )

    assert len(result.findings[0].evidence) == 2


def test_reconcile_keeps_contradictory_findings_separate():
    result = reconcile(
        "Should Hermes use Gemini CLI workers?",
        [
            make_worker_result(
                angle_name="Benefits",
                answer="Parallel research is possible.",
                key_findings=["Gemini CLI enables parallel web research workflows"],
                evidence=[],
            ),
            make_worker_result(
                angle_name="Risks",
                answer="Parallel research is not possible.",
                key_findings=["Gemini CLI does not enable parallel web research workflows"],
                evidence=[],
            ),
        ],
    )

    assert len(result.findings) == 2


def test_reconcile_treats_succeeded_without_output_as_partial_failure():
    broken_worker = WorkerResult(
        angle=ResearchAngle(name="Broken", description="broken output"),
        status=WorkerStatus.SUCCEEDED,
        output=None,
        error="worker returned no structured output",
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        duration_seconds=0.01,
    )

    result = reconcile(
        "Should Hermes use Gemini CLI workers?",
        [
            make_worker_result(
                angle_name="Benefits",
                answer="Parallel research is possible.",
                key_findings=["Gemini CLI enables parallel web research workflows"],
                evidence=[],
            ),
            broken_worker,
        ],
    )

    assert result.status == "partial"
    assert "worker returned no structured output" in result.limitations
