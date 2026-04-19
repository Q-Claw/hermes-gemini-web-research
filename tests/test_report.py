from hermes_gemini_web_research.models import EvidenceItem, ReconciledFinding, ResearchResult
from hermes_gemini_web_research.report import render_markdown_report


def make_result() -> ResearchResult:
    return ResearchResult(
        question="Should Hermes use Gemini CLI workers?",
        status="complete",
        summary="Gemini CLI is useful for structured research.",
        findings=[
            ReconciledFinding(
                finding="Gemini CLI supports structured JSON output",
                supporting_angles=["Current facts", "Evidence quality"],
                source_count=2,
                source_diversity=1.0,
                consensus_score=0.91,
                confidence="high",
                severity="medium",
                evidence=[
                    EvidenceItem(
                        type="source",
                        claim="Official docs describe JSON output support.",
                        source_title="Official docs",
                        url="https://example.com/docs",
                        quote="Use --output-format json for JSON output.",
                        published_date="2026-04-19",
                        confidence=0.92,
                    ),
                    EvidenceItem(
                        type="source",
                        claim="Independent analysis confirms JSON output is reliable.",
                        source_title="Independent analysis",
                        url="https://example.com/analysis",
                        confidence=0.84,
                    ),
                ],
            ),
            ReconciledFinding(
                finding="Gemini CLI can auto-approve tools in yolo mode",
                supporting_angles=["Operations"],
                source_count=1,
                source_diversity=1.0,
                consensus_score=0.67,
                confidence="medium",
                severity="low",
                contradicts=["Gemini CLI cannot auto-approve tools"],
                evidence=[
                    EvidenceItem(
                        type="source",
                        claim="Official docs describe yolo approval mode.",
                        source_title="Official docs",
                        url="https://example.com/docs",
                        confidence=0.88,
                    )
                ],
            ),
        ],
        worker_results=[],
        limitations=[],
        synthesis_method="deterministic",
    )


def test_report_renders_inline_citations_and_source_registry():
    report = render_markdown_report(make_result())

    assert "## Sources" in report
    assert "[1] Official docs — https://example.com/docs" in report
    assert "[2] Independent analysis — https://example.com/analysis" in report
    assert report.count("[1] Official docs — https://example.com/docs") == 1
    assert "Official docs describe JSON output support. [1]" in report
    assert "Official docs describe yolo approval mode. [1]" in report
    assert "Independent analysis confirms JSON output is reliable. [2]" in report


def test_report_includes_best_evidence_block_with_quote_and_date():
    report = render_markdown_report(make_result())

    assert "Best evidence:" in report
    assert 'Quote: "Use --output-format json for JSON output."' in report
    assert "Published: 2026-04-19" in report


def test_report_includes_severity_confidence_summary_near_top():
    report = render_markdown_report(make_result())

    assert report.index("## Severity / Confidence Summary") < report.index("## Summary")
    assert "Findings: 2" in report
    assert "Confidence: high 1, medium 1, low 0" in report
    assert "Severity: high 0, medium 1, low 1" in report
    assert "Contradictions: 1" in report


def test_report_renders_scoring_and_contradiction_markers_per_finding():
    report = render_markdown_report(make_result())

    assert "Consensus: 0.91; confidence: high; severity: medium; sources: 2; diversity: 1.00" in report
    assert "Consensus: 0.67; confidence: medium; severity: low; sources: 1; diversity: 1.00" in report
    assert "Contradiction: Gemini CLI cannot auto-approve tools" in report


def test_report_deduplicates_source_registry_by_url_even_if_titles_differ():
    result = ResearchResult(
        question="Should Hermes use Gemini CLI workers?",
        status="complete",
        summary="Gemini CLI is useful for structured research.",
        findings=[
            ReconciledFinding(
                finding="Gemini CLI supports structured JSON output",
                supporting_angles=["Current facts"],
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
                        claim="Mirrored docs describe JSON output support.",
                        source_title="Gemini CLI Reference",
                        url="https://example.com/docs",
                        confidence=0.9,
                    ),
                ],
            )
        ],
        worker_results=[],
        limitations=[],
        synthesis_method="deterministic",
    )

    report = render_markdown_report(result)

    assert report.count("https://example.com/docs") == 1
    assert "Official docs describe JSON output support. [1]" in report
    assert "Mirrored docs describe JSON output support. [1]" in report


def test_report_keeps_distinct_case_sensitive_urls_separate():
    result = ResearchResult(
        question="Should Hermes use Gemini CLI workers?",
        status="complete",
        summary="Gemini CLI is useful for structured research.",
        findings=[
            ReconciledFinding(
                finding="Gemini CLI supports structured JSON output",
                supporting_angles=["Current facts"],
                evidence=[
                    EvidenceItem(
                        type="source",
                        claim="Docs mention one resource.",
                        source_title="Docs A",
                        url="https://example.com/Readme",
                        confidence=0.92,
                    ),
                    EvidenceItem(
                        type="source",
                        claim="Docs mention another resource.",
                        source_title="Docs B",
                        url="https://example.com/readme",
                        confidence=0.9,
                    ),
                ],
            )
        ],
        worker_results=[],
        limitations=[],
        synthesis_method="deterministic",
    )

    report = render_markdown_report(result)

    assert "[1] Docs A — https://example.com/Readme" in report
    assert "[2] Docs B — https://example.com/readme" in report


def test_report_limits_rendered_best_evidence_items():
    result = ResearchResult(
        question="Should Hermes use Gemini CLI workers?",
        status="complete",
        summary="Gemini CLI is useful for structured research.",
        findings=[
            ReconciledFinding(
                finding="Gemini CLI supports structured JSON output",
                supporting_angles=["Current facts"],
                evidence=[
                    EvidenceItem(
                        type="source",
                        claim=f"Evidence item {index}.",
                        source_title=f"Source {index}",
                        url=f"https://example.com/source-{index}",
                        confidence=0.8,
                    )
                    for index in range(5)
                ],
            )
        ],
        worker_results=[],
        limitations=[],
        synthesis_method="deterministic",
    )

    report = render_markdown_report(result)

    assert "Evidence item 0. [1]" in report
    assert "Evidence item 1. [2]" in report
    assert "Evidence item 2. [3]" in report
    assert "Evidence item 3. [4]" not in report
    assert "Evidence item 4. [5]" not in report
