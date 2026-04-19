"""Markdown report rendering."""

from __future__ import annotations

from hermes_gemini_web_research.models import ResearchResult, WorkerStatus


def render_markdown_report(result: ResearchResult) -> str:
    """Render a human-readable Markdown report from a structured result."""

    lines = [
        f"# Research Report: {result.question}",
        "",
        f"Status: **{result.status}**",
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
        for evidence in finding.evidence[:3]:
            source = evidence.source_title or "Untitled source"
            url = f" ({evidence.url})" if evidence.url else ""
            lines.append(f"   - Evidence: {evidence.claim} - {source}{url}")

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

    return "\n".join(lines).strip() + "\n"

