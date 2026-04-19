"""Deterministic reconciliation and evidence ranking for worker outputs."""

from __future__ import annotations

import re
from collections import OrderedDict

from hermes_gemini_web_research.models import (
    EvidenceItem,
    ReconciledFinding,
    ResearchResult,
    WorkerResult,
    WorkerStatus,
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}

TOKEN_RE = re.compile(r"[a-z0-9]+")


def reconcile(question: str, worker_results: list[WorkerResult]) -> ResearchResult:
    """Synthesize worker outputs into a conservative structured result."""

    succeeded = [result for result in worker_results if result.status == WorkerStatus.SUCCEEDED and result.output]
    failed = [
        result
        for result in worker_results
        if result.status != WorkerStatus.SUCCEEDED or result.output is None
    ]

    if not succeeded:
        return ResearchResult(
            question=question,
            status="failed",
            summary="No worker returned valid research JSON.",
            worker_results=worker_results,
            limitations=[result.error or f"{result.angle.name} failed" for result in failed],
        )

    clusters: list[dict[str, object]] = []
    for result in succeeded:
        assert result.output is not None
        angle_name = result.output.angle_name or result.angle.name
        evidence_pool = list(result.output.evidence)
        for finding_text in result.output.key_findings or [result.output.answer]:
            normalized = " ".join(finding_text.strip().split())
            if not normalized:
                continue
            tokens = _finding_tokens(normalized)
            cluster = _find_cluster(clusters, normalized, tokens)
            if cluster is None:
                cluster = {
                    "finding": normalized,
                    "tokens": tokens,
                    "supporting_angles": [],
                    "evidence": [],
                }
                clusters.append(cluster)
            if angle_name not in cluster["supporting_angles"]:
                cluster["supporting_angles"].append(angle_name)
            cluster["tokens"] = cluster["tokens"] | tokens
            if len(normalized) > len(cluster["finding"]):
                cluster["finding"] = normalized
            cluster["evidence"].extend(evidence_pool)

    findings = [
        ReconciledFinding(
            finding=cluster["finding"],
            supporting_angles=cluster["supporting_angles"],
            evidence=_rank_and_dedupe_evidence(cluster["evidence"]),
        )
        for cluster in clusters
    ]
    findings.sort(
        key=lambda finding: (
            -len(finding.supporting_angles),
            -(_evidence_score(finding.evidence[0]) if finding.evidence else 0.0),
            finding.finding.lower(),
        )
    )

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
        findings=findings,
        worker_results=worker_results,
        limitations=limitations,
    )


def _find_cluster(clusters: list[dict[str, object]], finding: str, tokens: set[str]) -> dict[str, object] | None:
    best_cluster = None
    best_score = 0.0
    for cluster in clusters:
        cluster_tokens = cluster["tokens"]
        score = _token_similarity(tokens, cluster_tokens)
        if score > best_score:
            best_cluster = cluster
            best_score = score

    if best_cluster is None:
        return None
    if best_score >= 0.74:
        return best_cluster
    return None


def _finding_tokens(text: str) -> set[str]:
    tokens = []
    for token in TOKEN_RE.findall(text.lower()):
        if token in STOPWORDS:
            continue
        tokens.append(_normalize_token(token))
    return {token for token in tokens if token}


def _normalize_token(token: str) -> str:
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _token_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = left & right
    union = left | right
    if not union:
        return 0.0
    return len(intersection) / len(union)


def _rank_and_dedupe_evidence(evidence: list[EvidenceItem]) -> list[EvidenceItem]:
    deduped: OrderedDict[str, EvidenceItem] = OrderedDict()
    for item in evidence:
        key = _evidence_key(item)
        existing = deduped.get(key)
        if existing is None or _evidence_score(item) > _evidence_score(existing):
            deduped[key] = item
    return sorted(deduped.values(), key=_evidence_score, reverse=True)[:3]


def _evidence_key(item: EvidenceItem) -> str:
    claim_tokens = TOKEN_RE.findall(item.claim.lower())
    source_ref = ""
    if item.url:
        source_ref = str(item.url).rstrip("/").lower()
    elif item.source_title:
        source_ref = " ".join(item.source_title.lower().split())
    return f"{' '.join(claim_tokens)}|{source_ref}"


def _evidence_score(item: EvidenceItem) -> float:
    score = item.confidence
    if item.url:
        score += 0.15
    if item.source_title:
        score += 0.05
    if item.quote:
        score += 0.1
    if item.published_date:
        score += 0.05
    return score


def _build_summary(succeeded: list[WorkerResult]) -> str:
    answers = []
    for result in succeeded:
        assert result.output is not None
        answers.append(f"{result.output.angle_name or result.angle.name}: {result.output.answer}")
    return "\n".join(answers)

