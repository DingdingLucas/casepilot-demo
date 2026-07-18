#!/usr/bin/env python3
"""
Option A case-retrieval engine — v4 (argument confidence; no document grounding).

Given a matter description, ranks relevant past antitrust cases from cases_index.json
and returns the winning arguments to borrow — each with a confidence grade and the
basis for it — plus cautionary (court-rejected) points.

Safeguards enforced in code:
  * case_id validated against the index; unknown IDs dropped with a warning.
  * name / citation / folder_url / confidence / outcome joined locally from the
    index (authoritative by construction — the model never echoes them).
  * stop_reason (max_tokens / refusal) and typed API errors handled cleanly.

Only needs cases_index.json (v3). No grounding file.

Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY=...     # or run `ant auth login`

Usage:
    python search_cases_v4.py "Defending a two-sided-platform merger; DOJ alleges loss of a disruptor."
    python search_cases_v4.py --interactive --ttl 1h
    echo "monopsony merger depressing supplier prices" | python search_cases_v4.py --json
"""

import argparse
import json
import sys

import anthropic

MODEL = "claude-opus-4-8"
MAX_RESULTS = 6

ARGUMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "argument": {"type": "string", "description": "The concrete winning argument/evidence the user can borrow."},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"], "description": "Confidence this argument is sound authority for the user's matter (see rubric)."},
        "confidence_basis": {"type": "string", "description": "One sentence: court disposition (approved/rejected/mixed/pending), record confidence, and closeness of fit."},
    },
    "required": ["argument", "confidence", "confidence_basis"],
    "additionalProperties": False,
}

RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "query_summary": {"type": "string", "description": "One sentence restating the matter and the user's apparent posture (plaintiff/defendant, merger/conduct)."},
        "matches": {
            "type": "array",
            "description": f"Up to {MAX_RESULTS} most relevant cases, most relevant first.",
            "items": {
                "type": "object",
                "properties": {
                    "case_id": {"type": "string", "description": "Exact case_id copied from the index."},
                    "relevance": {"type": "integer", "description": "0-100 relevance to this matter."},
                    "why_it_fits": {"type": "string", "description": "Why this case is on point (market, theory of harm, economic models, posture)."},
                    "arguments_to_use": {"type": "array", "items": ARGUMENT_SCHEMA},
                    "cautions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Arguments the court REJECTED, the loser's failure mode, or provenance caveats (unverified/pending).",
                    },
                },
                "required": ["case_id", "relevance", "why_it_fits", "arguments_to_use", "cautions"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["query_summary", "matches"],
    "additionalProperties": False,
}

RULES = f"""You are an antitrust litigation research assistant for a legal-tech product.

Below is a curated index of past U.S. antitrust cases (JSON). Each record has the market,
theory of harm, standardized economic_models (defined in _meta.economic_models_vocabulary),
industry_taxonomy, winning_side/arguments, winning_evidence, judge_treatment, and a citation
object whose `stands_for` list says which arguments the COURT approved or rejected.

The user will describe a matter they are working on. Your job:
1. Infer the user's likely posture (arguing FOR or AGAINST liability/the merger) unless they
   state it.
2. Rank the most relevant cases (up to {MAX_RESULTS}) by fit on theory of harm,
   market/industry, economic models, and posture. Prefer fewer, genuinely on-point cases
   over padding the list.
3. For each case, give `arguments_to_use` — concrete winning arguments/evidence the user can
   borrow, favouring propositions the court APPROVED that help the user's side. Each
   argument carries a confidence grade under this rubric:
     high   = the court expressly APPROVED the proposition (see citation.stands_for), the
              record's confidence field is "high", and the fit to the user's matter is direct;
     medium = approved but analogous rather than identical fit, or a "mixed" disposition,
              or the record's confidence is "medium";
     low    = pending/unverified cases, allegations not yet adjudicated, or arguments that
              require substantial distinguishing to help the user.
   `confidence_basis` must justify the grade in one sentence and name the disposition.
   A case whose holding CUTS AGAINST the user is still relevant — include it, put the
   adverse holding in `cautions`, and say how to distinguish it.
4. For each case, give `cautions` — arguments the court REJECTED, the loser's failure mode,
   and a provenance caveat whenever the record's confidence is "low"/"medium" or the outcome
   is "pending"/"unverified".

Hard rules:
- Use ONLY cases in the index. Never invent cases, holdings, citations, or experts.
- `case_id` must be copied verbatim from the index (it is validated downstream).
- All records are provenance="public_record" (not doc-verified) — when a case is central to
  the user's argument, remind them in cautions to verify against the underlying documents.
- If nothing is a good match, return an empty matches array and say so in query_summary.
"""


def load_index(cases_path: str) -> tuple[str, dict]:
    with open(cases_path, "r", encoding="utf-8") as f:
        raw = f.read()
    data = json.loads(raw)  # fail fast on malformed/truncated index
    lookup = {c["case_id"]: c for c in data.get("cases", [])}
    if not lookup:
        sys.exit(f"error: no cases found in {cases_path}")
    return raw, lookup


def build_system(raw_index: str, ttl: str) -> list[dict]:
    # One stable block (rules + index), byte-identical across queries -> cache prefix holds.
    block = {
        "type": "text",
        "text": RULES + "\n\n=== CASE INDEX (JSON) ===\n" + raw_index,
        "cache_control": {"type": "ephemeral"},
    }
    if ttl == "1h":
        block["cache_control"]["ttl"] = "1h"  # 2x write cost, stays warm across a demo day
    return [block]


def search(client, system, lookup, matter, posture):
    user_text = matter if not posture else f"[User's posture: {posture}]\n\n{matter}"
    resp = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=system,
        output_config={"format": {"type": "json_schema", "schema": RESULT_SCHEMA}, "effort": "high"},
        messages=[{"role": "user", "content": user_text}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("The model declined this query (stop_reason=refusal). Rephrase and retry.")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("Response hit max_tokens and may be truncated. Raise max_tokens and retry.")
    text = next((b.text for b in resp.content if b.type == "text"), None)
    if text is None:
        raise RuntimeError(f"No text block in response (stop_reason={resp.stop_reason}).")
    result = json.loads(text)  # structured outputs guarantee schema-valid JSON

    # Anti-hallucination gate + local join of authoritative fields.
    verified = []
    for m in result.get("matches", []):
        rec = lookup.get(m["case_id"])
        if rec is None:
            print(f"warning: dropped unknown case_id {m['case_id']!r} returned by the model", file=sys.stderr)
            continue
        m["name"] = rec["name"]
        m["citation"] = rec["citation"]["cite"] if isinstance(rec["citation"], dict) else rec["citation"]
        m["folder_url"] = rec.get("folder_url", "")
        m["record_confidence"] = rec.get("confidence", "")
        m["outcome"] = rec.get("outcome", "")
        verified.append(m)
    verified.sort(key=lambda m: m["relevance"], reverse=True)
    result["matches"] = verified
    return result, resp.usage


def format_report(result: dict) -> str:
    lines = [f"\nQuery: {result['query_summary']}\n"]
    if not result["matches"]:
        lines.append("No sufficiently relevant cases in the index.")
        return "\n".join(lines)
    for i, m in enumerate(result["matches"], 1):
        lines.append(f"{i}. [{m['relevance']}] {m['name']}  ({m['case_id']}, record confidence: {m['record_confidence']})")
        lines.append(f"   {m['citation']}")
        lines.append(f"   Outcome: {m['outcome']}")
        lines.append(f"   Why: {m['why_it_fits']}")
        for a in m["arguments_to_use"]:
            lines.append(f"     + ({a['confidence'].upper()}) {a['argument']}")
            lines.append(f"         basis: {a['confidence_basis']}")
        for c in m["cautions"]:
            lines.append(f"     ! {c}")
        lines.append(f"   Docs: {m['folder_url']}")
        lines.append("")
    return "\n".join(lines)


def run_query(client, system, lookup, matter, posture, as_json) -> None:
    try:
        result, usage = search(client, system, lookup, matter, posture)
    except anthropic.RateLimitError:
        print("Rate limited — wait a few seconds and retry.", file=sys.stderr); return
    except anthropic.APIStatusError as e:
        print(f"API error {e.status_code}: {e.message}", file=sys.stderr); return
    except anthropic.APIConnectionError:
        print("Network error — check your connection and retry.", file=sys.stderr); return
    except (RuntimeError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr); return

    print(json.dumps(result, indent=2) if as_json else format_report(result))
    print(f"[tokens] input={usage.input_tokens} cache_read={usage.cache_read_input_tokens} "
          f"cache_write={usage.cache_creation_input_tokens} output={usage.output_tokens}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description="Rank relevant antitrust cases with per-argument confidence.")
    ap.add_argument("matter", nargs="?", help="Matter/project description (or stdin, or --interactive).")
    ap.add_argument("--cases", default="cases_index.json", help="Path to cases_index.json (v3).")
    ap.add_argument("--json", action="store_true", help="Emit raw JSON instead of a formatted report.")
    ap.add_argument("--posture", choices=["plaintiff", "defendant"], help="Bias ranking toward this side.")
    ap.add_argument("--ttl", choices=["5m", "1h"], default="5m", help="Cache TTL (1h keeps the index warm across a demo day).")
    ap.add_argument("--interactive", action="store_true", help="REPL mode: repeated queries reuse the warm cache.")
    args = ap.parse_args()

    raw_index, lookup = load_index(args.cases)
    system = build_system(raw_index, args.ttl)
    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY or an `ant auth login` profile

    if args.interactive:
        print(f"Loaded {len(lookup)} cases. Describe your matter (blank line or Ctrl-D to exit).")
        while True:
            try:
                matter = input("\nmatter> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not matter:
                break
            run_query(client, system, lookup, matter, args.posture, args.json)
        return

    matter = args.matter or sys.stdin.read().strip()
    if not matter:
        ap.error("Provide a matter description, use stdin, or --interactive.")
    run_query(client, system, lookup, matter, args.posture, args.json)


if __name__ == "__main__":
    main()
