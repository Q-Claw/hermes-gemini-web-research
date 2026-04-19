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
    "do",
    "does",
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

NEGATION_TOKENS = {
    "absent",
    "cannot",
    "cant",
    "disabled",
    "impossible",
    "lack",
    "lacks",
    "never",
    "no",
    "not",
    "unavailable",
    "without",
}

NEGATION_CONTRACTIONS = {
    "can't": "cannot",
    "don't": "not",
    "doesn't": "not",
    "isn't": "not",
    "won't": "not",
}

TOKEN_RE = re.compile(r"[a-z0-9]+")
NEGATION_CONTRACTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(contraction) for contraction in NEGATION_CONTRACTIONS) + r")\b",
    re.IGNORECASE,
)


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
    enrich_findings(findings, total_angles=len({result.output.angle_name or result.angle.name for result in succeeded}))
    findings.sort(
        key=lambda finding: (
            -len(finding.supporting_angles),
            -finding.consensus_score,
            -(finding.best_evidence_score or 0.0),
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


def enrich_findings(findings: list[ReconciledFinding], *, total_angles: int) -> list[ReconciledFinding]:
    """Populate deterministic scoring and contradiction metadata on findings."""

    _score_findings(findings, total_angles=total_angles)
    _mark_contradictions(findings)
    return findings


def _find_cluster(clusters: list[dict[str, object]], finding: str, tokens: set[str]) -> dict[str, object] | None:
    best_cluster = None
    best_score = 0.0
    for cluster in clusters:
        if _are_contradictory(finding, cluster["finding"]):
            continue
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
    for token in TOKEN_RE.findall(_expand_negation_contractions(text).lower()):
        if token in STOPWORDS:
            continue
        tokens.append(_normalize_token(token))
    return {token for token in tokens if token}


def _expand_negation_contractions(text: str) -> str:
    return NEGATION_CONTRACTION_RE.sub(
        lambda match: NEGATION_CONTRACTIONS[match.group(1).lower()],
        text,
    )


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
    return sorted(deduped.values(), key=_evidence_score, reverse=True)


def _score_findings(findings: list[ReconciledFinding], *, total_angles: int) -> None:
    total_angles = max(total_angles, 1)
    for finding in findings:
        evidence_scores = [min(_evidence_score(item), 1.0) for item in finding.evidence]
        unique_sources = {_source_identity(item) for item in finding.evidence}
        unique_sources.discard("")
        source_count = len(unique_sources)
        source_diversity = source_count / len(finding.evidence) if finding.evidence else 0.0
        angle_score = min(len(set(finding.supporting_angles)) / total_angles, 1.0)
        source_score = min(source_count / 3, 1.0)
        evidence_quality = sum(evidence_scores) / len(evidence_scores) if evidence_scores else 0.0
        consensus_score = (0.45 * angle_score) + (0.35 * source_score) + (0.20 * evidence_quality)

        finding.source_count = source_count
        finding.source_diversity = round(source_diversity, 2)
        finding.consensus_score = round(min(consensus_score, 1.0), 2)
        finding.best_evidence = finding.evidence[0] if finding.evidence else None
        finding.best_evidence_score = round(evidence_scores[0], 2) if evidence_scores else None
        finding.confidence = _confidence_label(finding.consensus_score)
        finding.severity = _severity_label(finding.consensus_score, has_contradiction=False)


def _mark_contradictions(findings: list[ReconciledFinding]) -> None:
    for finding in findings:
        finding.contradicts = []

    for left_index, left in enumerate(findings):
        for right in findings[left_index + 1 :]:
            if not _are_contradictory(left.finding, right.finding):
                continue
            left.contradicts.append(right.finding)
            right.contradicts.append(left.finding)

    for finding in findings:
        finding.contradicts = sorted(set(finding.contradicts), key=str.lower)
        finding.severity = _severity_label(finding.consensus_score, has_contradiction=bool(finding.contradicts))


def _are_contradictory(left: str, right: str) -> bool:
    left_has_negation = _has_negation(left)
    right_has_negation = _has_negation(right)
    if left_has_negation == right_has_negation:
        return False
    left_tokens = _finding_tokens_without_negation(left)
    right_tokens = _finding_tokens_without_negation(right)
    return _token_similarity(left_tokens, right_tokens) >= 0.72


def _has_negation(text: str) -> bool:
    return any(token in NEGATION_TOKENS for token in TOKEN_RE.findall(_expand_negation_contractions(text).lower()))


def _finding_tokens_without_negation(text: str) -> set[str]:
    return {
        token
        for token in _finding_tokens(text)
        if token not in {_normalize_token(negation) for negation in NEGATION_TOKENS}
    }


def _source_identity(item: EvidenceItem) -> str:
    if item.url:
        return f"url:{_normalize_source_url(item.url)}"
    if item.source_title:
        return f"title:{' '.join(item.source_title.lower().split())}"
    return ""


def _normalize_source_url(url: object) -> str:
    return str(url).rstrip().rstrip("/")


def _confidence_label(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def _severity_label(score: float, *, has_contradiction: bool) -> str:
    if has_contradiction and score >= 0.65:
        return "high"
    if has_contradiction or score >= 0.75:
        return "medium"
    return "low"


def _evidence_key(item: EvidenceItem) -> str:
    claim_tokens = TOKEN_RE.findall(item.claim.lower())
    source_ref = ""
    if item.url:
        source_ref = _normalize_source_url(item.url)
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
