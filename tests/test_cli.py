from __future__ import annotations

import json
from datetime import datetime, timezone

from hermes_gemini_web_research.cli import main
from hermes_gemini_web_research.models import (
    EvidenceItem,
    ReconciledFinding,
    ResearchAngle,
    ResearchResult,
    WorkerOutput,
    WorkerResult,
    WorkerStatus,
)


class StubOrchestrator:
    last_instance = None

    def __init__(self, runner=None, *, synthesizer=None) -> None:
        self.runner = runner
        self.synthesizer = synthesizer
        self.last_request = None
        StubOrchestrator.last_instance = self

    async def run(self, request):
        self.last_request = request
        return ResearchResult(
            question=request.question,
            status="complete",
            summary="Current facts: concise summary",
            findings=[
                ReconciledFinding(
                    finding="concise summary",
                    supporting_angles=["Current facts"],
                    evidence=[
                        EvidenceItem(
                            type="source",
                            claim="Docs support concise summary.",
                            source_title="Docs",
                            url="https://example.com/docs",
                            confidence=0.9,
                        )
                    ],
                    source_count=1,
                    source_diversity=1.0,
                    consensus_score=0.74,
                    confidence="medium",
                    severity="low",
                    best_evidence=EvidenceItem(
                        type="source",
                        claim="Docs support concise summary.",
                        source_title="Docs",
                        url="https://example.com/docs",
                        confidence=0.9,
                    ),
                    best_evidence_score=1.0,
                )
            ],
            worker_results=[
                WorkerResult(
                    angle=ResearchAngle(name="Current facts", description="desc"),
                    status=WorkerStatus.SUCCEEDED,
                    output=WorkerOutput(
                        angle_name="Current facts",
                        answer="concise summary",
                        key_findings=["concise summary"],
                        confidence=0.9,
                    ),
                    started_at=datetime.now(timezone.utc),
                    completed_at=datetime.now(timezone.utc),
                    duration_seconds=0.01,
                )
            ],
            limitations=[],
        )


def test_cli_writes_markdown_output_file_without_polluting_stdout(tmp_path, monkeypatch, capsys):
    output_file = tmp_path / "reports" / "result.md"
    monkeypatch.setattr("hermes_gemini_web_research.cli.ResearchOrchestrator", StubOrchestrator)

    exit_code = main([
        "What happened?",
        "--output-file",
        str(output_file),
    ])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert output_file.exists()
    assert output_file.read_text(encoding="utf-8").startswith("# Research Report: What happened?")
    assert captured.out == ""
    assert str(output_file) in captured.err


def test_cli_writes_json_output_file(tmp_path, monkeypatch, capsys):
    output_file = tmp_path / "result.json"
    monkeypatch.setattr("hermes_gemini_web_research.cli.ResearchOrchestrator", StubOrchestrator)

    exit_code = main([
        "What happened?",
        "--format",
        "json",
        "--output-file",
        str(output_file),
    ])

    captured = capsys.readouterr()
    payload = json.loads(output_file.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["question"] == "What happened?"
    assert payload["synthesis_method"] == "deterministic"
    assert payload["report_summary"]["finding_count"] == 1
    assert payload["findings"][0]["best_evidence"]["source_title"] == "Docs"
    assert payload["findings"][0]["best_evidence_score"] == 1.0
    assert payload["findings"][0]["consensus_score"] == 0.74
    assert payload["findings"][0]["source_diversity"] == 1.0
    assert captured.out == ""
    assert str(output_file) in captured.err


def test_cli_enables_semantic_synthesis_with_real_synthesizer_hook(monkeypatch):
    monkeypatch.setattr("hermes_gemini_web_research.cli.ResearchOrchestrator", StubOrchestrator)

    exit_code = main([
        "What happened?",
        "--semantic-synthesis",
    ])

    orchestrator = StubOrchestrator.last_instance

    assert exit_code == 0
    assert orchestrator is not None
    assert orchestrator.last_request is not None
    assert orchestrator.last_request.semantic_synthesis is True
    assert orchestrator.synthesizer is not None


def test_cli_returns_clean_error_when_output_file_write_fails(tmp_path, monkeypatch, capsys):
    output_file = tmp_path / "cannot-write.json"
    monkeypatch.setattr("hermes_gemini_web_research.cli.ResearchOrchestrator", StubOrchestrator)

    def boom(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(type(output_file), "write_text", boom)

    exit_code = main([
        "What happened?",
        "--format",
        "json",
        "--output-file",
        str(output_file),
    ])

    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert "disk full" in captured.err
