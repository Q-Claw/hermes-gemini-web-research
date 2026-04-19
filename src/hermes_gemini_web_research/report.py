"""Markdown report rendering."""

from __future__ import annotations

from hermes_gemini_web_research.models import EvidenceItem, ResearchResult, WorkerStatus


def render_markdown_report(result: ResearchResult) -> str:
    """Render a human-readable Markdown report from a structured result."""

    source_index = _build_source_index(result)
    lines = [
        f"# Research Report: {result.question}",
        "",
        f"Status: **{result.status}**",
        f"Synthesis: **{result.synthesis_method}**",
        "",
        "## Summary",
        "",
        result.summary,
        "",
        "## Reconciled Findings",
        "",
    ]

    if not result.findings:
        lines.append("No reconciled findings were produced.")
    for index, finding in enumerate(result.findings, start=1):
        lines.append(f"{index}. {finding.finding}")
        if finding.supporting_angles:
            lines.append(f"   - Angles: {', '.join(finding.supporting_angles)}")
        if finding.evidence:
            lines.append("   - Best evidence:")
        for evidence in finding.evidence[:3]:
            citation = _citation_label(evidence, source_index)
            lines.append(f"     - {evidence.claim} {citation}".rstrip())
            if evidence.quote:
                lines.append(f'       - Quote: "{evidence.quote}"')
            if evidence.published_date:
                lines.append(f"       - Published: {evidence.published_date}")

    lines.extend(["", "## Worker Results", ""])
    for worker in result.worker_results:
        status = worker.status.value
        duration = f", {worker.duration_seconds}s" if worker.duration_seconds is not None else ""
        lines.append(f"- **{worker.angle.name}**: {status}{duration}")
        if worker.status == WorkerStatus.SUCCEEDED and worker.output:
            lines.append(f"  - Confidence: {worker.output.confidence:.2f}")
        if worker.error:
            lines.append(f"  - Error: {worker.error}")
        if worker.usage:
            usage = worker.usage
            token_parts = []
            if usage.prompt_tokens is not None:
                token_parts.append(f"prompt={usage.prompt_tokens}")
            if usage.completion_tokens is not None:
                token_parts.append(f"completion={usage.completion_tokens}")
            if usage.total_tokens is not None:
                token_parts.append(f"total={usage.total_tokens}")
            if token_parts:
                lines.append(f"  - Usage: {', '.join(token_parts)}")

    if result.limitations:
        lines.extend(["", "## Limitations", ""])
        for limitation in result.limitations:
            lines.append(f"- {limitation}")

    if source_index:
        lines.extend(["", "## Sources", ""])
        rendered_sources: set[int] = set()
        for finding in result.findings:
            for evidence in finding.evidence:
                number, title = _source_label(evidence, source_index)
                if number is None or number in rendered_sources:
                    continue
                rendered_sources.add(number)
                if evidence.url:
                    lines.append(f"[{number}] {title} — {evidence.url}")
                else:
                    lines.append(f"[{number}] {title}")

    if result.synthesis_error:
        lines.extend(["", "## Synthesis Fallback", "", result.synthesis_error])

    return "\n".join(lines).strip() + "\n"


def _build_source_index(result: ResearchResult) -> dict[tuple[str, str], int]:
    source_index: dict[tuple[str, str], int] = {}
    next_number = 1
    for finding in result.findings:
        for evidence in finding.evidence:
            key = _source_key(evidence)
            if key not in source_index:
                source_index[key] = next_number
                next_number += 1
    return source_index


def _source_key(evidence: EvidenceItem) -> tuple[str, str]:
    title = " ".join((evidence.source_title or "Untitled source").split())
    if evidence.url:
        return ("url", str(evidence.url).rstrip("/"))
    return ("title", title.lower())


def _source_label(evidence: EvidenceItem, source_index: dict[tuple[str, str], int]) -> tuple[int | None, str]:
    number = source_index.get(_source_key(evidence))
    title = evidence.source_title or "Untitled source"
    return number, title


def _citation_label(evidence: EvidenceItem, source_index: dict[tuple[str, str], int]) -> str:
    number, _ = _source_label(evidence, source_index)
    if number is None:
        return ""
    return f"[{number}]"
