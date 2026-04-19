"""Prompt construction and JSON extraction utilities."""

from __future__ import annotations

import json
import re
from typing import Any

from hermes_gemini_web_research.models import WorkerInput, WorkerOutput


STRICT_JSON_INSTRUCTIONS = """You are one research worker in a multi-agent research controller.
Return STRICT JSON only. Do not include Markdown fences, prose before JSON, or prose after JSON.
Your JSON must conform to the provided schema. Prefer concrete evidence and include URLs when available.
If evidence is weak or missing, say so in open_questions and lower confidence."""

ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def strip_ansi_escape_sequences(text: str) -> str:
    """Remove terminal control sequences that can wrap CLI JSON output."""

    return ANSI_ESCAPE_RE.sub("", text)


def worker_output_schema() -> dict[str, Any]:
    """Return the JSON schema sent to Gemini for WorkerOutput."""

    return WorkerOutput.model_json_schema()


def build_worker_prompt(worker_input: WorkerInput) -> str:
    """Build a headless Gemini prompt for one angle."""

    schema_json = json.dumps(worker_output_schema(), indent=2, sort_keys=True)
    return "\n\n".join(
        [
            STRICT_JSON_INSTRUCTIONS,
            f"Question: {worker_input.question}",
            f"Research angle name: {worker_input.angle.name}",
            f"Research angle instructions: {worker_input.angle.description}",
            "Required JSON schema:",
            schema_json,
        ]
    )


def extract_json_object(text: str) -> str:
    """Extract the first complete JSON object from raw model or wrapper text."""

    stripped = strip_ansi_escape_sequences(text).strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)

    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start = stripped.find("{")
    if start == -1:
        raise ValueError("no JSON object found")

    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(stripped[start:], start=start):
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : index + 1]

    raise ValueError("unterminated JSON object")


def parse_worker_output(text: str) -> WorkerOutput:
    """Parse and validate worker JSON from raw text."""

    return WorkerOutput.model_validate_json(extract_json_object(text))
