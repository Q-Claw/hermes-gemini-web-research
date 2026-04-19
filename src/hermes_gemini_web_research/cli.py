"""Command-line entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from hermes_gemini_web_research.models import ResearchAngle, ResearchRequest
from hermes_gemini_web_research.orchestrator import ResearchOrchestrator
from hermes_gemini_web_research.report import render_markdown_report
from hermes_gemini_web_research.runner import GeminiRunner


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        angles = _load_angles(args)
        request = ResearchRequest(
            question=args.question,
            angles=angles,
            timeout_seconds=args.timeout,
            max_concurrency=args.max_concurrency,
        )
    except (OSError, ValueError, ValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    runner = GeminiRunner(command=args.gemini_command, args=args.gemini_arg or None)
    result = asyncio.run(ResearchOrchestrator(runner).run(request))

    if args.format == "json":
        print(result.model_dump_json(indent=2))
    else:
        print(render_markdown_report(result))

    return 0 if result.status in {"complete", "partial"} else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hermes-gemini-web-research",
        description="Run concurrent Gemini CLI research workers and synthesize a report.",
    )
    parser.add_argument("question", help="User research question.")
    parser.add_argument(
        "--angle",
        action="append",
        default=[],
        help="Research angle as 'name: description'. Can be passed multiple times.",
    )
    parser.add_argument(
        "--angle-file",
        type=Path,
        help="JSON file containing angles, either a list or {'angles': [...]} objects with name and description.",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-worker timeout in seconds.")
    parser.add_argument("--max-concurrency", type=int, default=4, help="Maximum concurrent Gemini workers.")
    parser.add_argument("--gemini-command", default="gemini", help="Gemini CLI executable.")
    parser.add_argument(
        "--gemini-arg",
        action="append",
        default=None,
        help=(
            "Argument(s) placed before the prompt. Defaults to "
            "--output-format json --approval-mode=yolo --prompt when omitted."
        ),
    )
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    return parser


def _load_angles(args: argparse.Namespace) -> list[ResearchAngle]:
    angles: list[ResearchAngle] = []
    if args.angle_file:
        data = json.loads(args.angle_file.read_text(encoding="utf-8"))
        raw_angles = data.get("angles", data) if isinstance(data, dict) else data
        if not isinstance(raw_angles, list):
            raise ValueError("angle file must contain a list of angle objects")
        angles.extend(ResearchAngle.model_validate(item) for item in raw_angles)

    for raw_angle in args.angle:
        if ":" not in raw_angle:
            raise ValueError("--angle must use 'name: description'")
        name, description = raw_angle.split(":", 1)
        angles.append(ResearchAngle(name=name.strip(), description=description.strip()))

    return angles


if __name__ == "__main__":
    raise SystemExit(main())
