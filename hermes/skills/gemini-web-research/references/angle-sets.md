# Angle Sets

Use these angle patterns when a Hermes task needs repeatable Gemini worker coverage.

## Current Decision

Use for product, engineering, vendor, or ecosystem decisions.

```json
{
  "angles": [
    {
      "name": "Current facts",
      "description": "Find the latest verifiable facts with primary sources and dates."
    },
    {
      "name": "Evidence quality",
      "description": "Check source authority, conflicts, weak evidence, and caveats."
    },
    {
      "name": "Practical implications",
      "description": "Translate findings into decision-relevant tradeoffs and next steps."
    }
  ]
}
```

## Implementation Readiness

Use before committing engineering time to an integration.

```json
{
  "angles": [
    {
      "name": "API and tooling",
      "description": "Verify available APIs, CLIs, flags, schemas, and version constraints."
    },
    {
      "name": "Reliability",
      "description": "Look for known failure modes, quotas, retries, observability, and recovery paths."
    },
    {
      "name": "Operations",
      "description": "Assess cost, security, deployment, CI, and maintenance implications."
    }
  ]
}
```

## Report Review Checklist

- Check `status` first: `complete`, `partial`, or `failed`.
- Check `synthesis_method`: `deterministic` is the baseline; `semantic` means an optional synthesizer refined the result.
- Treat `synthesis_error` as a graceful fallback note, not as a failed research run.
- Read limitations before presenting the answer as final.
- Prefer evidence with URLs, source titles, publication dates, and higher confidence.
