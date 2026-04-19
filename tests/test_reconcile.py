from __future__ import annotations

from datetime import datetime, timezone

import pytest

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


def test_reconcile_marks_deterministic_contradictions_on_both_findings():
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

    by_text = {finding.finding: finding for finding in result.findings}

    assert by_text["Gemini CLI enables parallel web research workflows"].contradicts == [
        "Gemini CLI does not enable parallel web research workflows"
    ]
    assert by_text["Gemini CLI does not enable parallel web research workflows"].contradicts == [
        "Gemini CLI enables parallel web research workflows"
    ]
    assert result.report_summary.contradiction_count == 1


@pytest.mark.parametrize(
    ("positive", "negative"),
    [
        (
            "Gemini CLI can enable parallel web research workflows",
            "Gemini CLI can't enable parallel web research workflows",
        ),
        (
            "Gemini CLI workers do support structured JSON output",
            "Gemini CLI workers don't support structured JSON output",
        ),
        (
            "Gemini CLI output is available for reconciliation",
            "Gemini CLI output isn't available for reconciliation",
        ),
        (
            "Gemini CLI workers will publish source metadata",
            "Gemini CLI workers won't publish source metadata",
        ),
        (
            "Gemini CLI is enabled for web research",
            "Gemini CLI is disabled for web research",
        ),
        (
            "Gemini CLI output is available for reconciliation",
            "Gemini CLI output is unavailable for reconciliation",
        ),
        (
            "Gemini CLI workers support citations",
            "Gemini CLI workers lack citations",
        ),
    ],
)
def test_reconcile_marks_contraction_negations_as_contradictions(positive: str, negative: str):
    result = reconcile(
        "Should Hermes use Gemini CLI workers?",
        [
            make_worker_result(
                angle_name="Benefits",
                answer=positive,
                key_findings=[positive],
                evidence=[],
            ),
            make_worker_result(
                angle_name="Risks",
                answer=negative,
                key_findings=[negative],
                evidence=[],
            ),
        ],
    )

    by_text = {finding.finding: finding for finding in result.findings}

    assert by_text[positive].contradicts == [negative]
    assert by_text[negative].contradicts == [positive]


def test_reconcile_does_not_mark_unrelated_negative_findings_as_contradictions():
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
                angle_name="Pricing",
                answer="Pricing is unclear.",
                key_findings=["Gemini CLI does not publish enterprise pricing"],
                evidence=[],
            ),
        ],
    )

    assert all(not finding.contradicts for finding in result.findings)


def test_reconcile_surfaces_source_diversity_consensus_and_best_evidence():
    official = EvidenceItem(
        type="source",
        claim="Official docs describe JSON output support.",
        source_title="Official docs",
        url="https://example.com/docs",
        quote="Use --output-format json for JSON output.",
        published_date="2026-04-19",
        confidence=0.92,
    )
    independent = EvidenceItem(
        type="source",
        claim="Independent analysis confirms JSON output is reliable.",
        source_title="Independent analysis",
        url="https://example.com/analysis",
        confidence=0.84,
    )

    result = reconcile(
        "How reliable is Gemini CLI for orchestration?",
        [
            make_worker_result(
                angle_name="Current facts",
                answer="It supports JSON output.",
                key_findings=["Gemini CLI supports structured JSON output"],
                evidence=[official],
            ),
            make_worker_result(
                angle_name="Evidence quality",
                answer="Docs confirm JSON output.",
                key_findings=["Gemini CLI supports structured JSON output"],
                evidence=[independent],
            ),
        ],
    )

    finding = result.findings[0]

    assert finding.source_count == 2
    assert finding.source_diversity == 1.0
    assert finding.consensus_score > 0.8
    assert finding.confidence == "high"
    assert finding.severity == "medium"
    assert finding.best_evidence == official
    assert finding.best_evidence_score == 1.0
    assert result.report_summary.finding_count == 1
    assert result.report_summary.confidence_counts == {"high": 1, "medium": 0, "low": 0}
    assert result.report_summary.severity_counts == {"high": 0, "medium": 1, "low": 0}


def test_reconcile_source_count_collapses_trailing_slash_but_preserves_case_sensitive_urls():
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
                        claim="Official docs describe JSON output support.",
                        source_title="Official docs",
                        url="https://example.com/docs",
                        confidence=0.92,
                    ),
                    EvidenceItem(
                        type="source",
                        claim="Official docs describe JSON output support.",
                        source_title="Official docs mirror",
                        url="https://example.com/docs/",
                        confidence=0.9,
                    ),
                    EvidenceItem(
                        type="source",
                        claim="Official docs describe JSON output support.",
                        source_title="Case-sensitive docs",
                        url="https://example.com/Docs",
                        confidence=0.88,
                    ),
                ],
            )
        ],
    )

    finding = result.findings[0]

    assert finding.source_count == 2
    assert [str(evidence.url).rstrip("/") for evidence in finding.evidence] == [
        "https://example.com/docs",
        "https://example.com/Docs",
    ]


def test_reconcile_scores_against_full_deduped_evidence_set_before_presentation_limit():
    evidence = [
        EvidenceItem(
            type="source",
            claim=f"Independent source {index} confirms JSON output support.",
            source_title=f"Independent source {index}",
            url=f"https://source-{index}.example.com/report",
            confidence=0.8,
        )
        for index in range(5)
    ]

    result = reconcile(
        "How reliable is Gemini CLI for orchestration?",
        [
            make_worker_result(
                angle_name="Current facts",
                answer="It supports JSON output.",
                key_findings=["Gemini CLI supports structured JSON output"],
                evidence=evidence,
            )
        ],
    )

    finding = result.findings[0]

    assert len(finding.evidence) == 5
    assert finding.source_count == 5
    assert finding.source_diversity == 1.0
    assert finding.consensus_score == 1.0


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
