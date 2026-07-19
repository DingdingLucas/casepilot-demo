#!/usr/bin/env python3
"""
CasePilot — demo web app for the econ-consulting case-retrieval product.

Feature 1: user enters their role + a matter description; the app returns the most
relevant past antitrust cases from cases_index.json with winning arguments
(confidence-graded) and cautions. A follow-up chat lets the user ask questions
about the retrieved cases, grounded in the same index.

Endpoints:
    GET  /            → static/index.html
    POST /api/search  → {role, query, posture?} → structured case matches (JSON)
    POST /api/chat    → {role, history: [...]}  → streamed text answer (SSE)

Setup:
    python3 -m venv venv && venv/bin/pip install anthropic flask
    export ANTHROPIC_API_KEY=sk-ant-...
    venv/bin/python app.py            # → http://localhost:5001
"""

import json
import os
import sys
import threading
import time

import anthropic
from flask import Flask, Response, jsonify, request, send_from_directory

MODEL = "claude-opus-4-8"
MAX_RESULTS = 6
CASES_PATH = "cases_index.json"
GROUNDING_PATH = "evidence_grounding.json"

# ---------------------------------------------------------------- schemas ----

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
   state it. If a role is given, tailor relevance and arguments to that role's needs.
2. Review EVERY case in the index before answering — walk through all records and assess
   each one's fit; do not stop at the first few plausible matches. Then rank the most
   relevant cases (up to {MAX_RESULTS}) by fit on theory of harm, market/industry,
   economic models, and posture. Prefer fewer, genuinely on-point cases over padding the
   list — a case with only superficial overlap should be omitted, not listed with a low
   relevance score.
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
- Use ONLY cases in the index. This database is the entire universe you may draw from:
  never bring in cases, holdings, citations, or experts from your own knowledge, even
  famous ones you are certain about, and never invent any.
- `case_id` must be copied verbatim from the index (it is validated downstream).
- All records are provenance="public_record" (not doc-verified) — when a case is central to
  the user's argument, remind them in cautions to verify against the underlying documents.
- Be truthful about coverage gaps. If, after reviewing every case, nothing is genuinely
  relevant, return an EMPTY matches array and say plainly in query_summary that the
  database contains no relevant case for this matter (name the closest-but-insufficient
  candidates there if useful). An honest empty result is always better than weak matches.
"""

DRAFT_RULES = """
=== DRAFTING MODE ===
The user has run a case search; their matter, role, and the retrieved cases (JSON) are in
the user message. Write a DRAFT ARGUMENT OUTLINE they can adapt for their matter, built
strictly and exclusively from the retrieved cases and the case index above — never from
outside authority. Format in markdown:

# Draft arguments — [one-line matter restatement]

## I., II., ... — one section per major argument, strongest first
For each section:
- State the argument crisply in the section heading.
- Develop it in 2-4 sentences tailored to the user's matter and role.
- Support it with the specific case authority from the retrieved results: name the case,
  what the court approved, and why it transfers. Mark each authority (HIGH/MEDIUM/LOW)
  using the confidence grades from the search results.

## Anticipated counterarguments
The strongest points the other side will take from these same cases (including any
retrieved case whose holding cuts against the user), each with a one-sentence response
or distinguishing point.

## Verification checklist
Bullet list: every case relied on, with its citation, and a reminder that records are
public_record provenance — verify quotes and pinpoints against the folder documents
before filing or client use.

Rules: cite ONLY the retrieved cases (you may add context from their index records).
Never invent holdings, quotes, page cites, or experts. If the retrieved cases cannot
support a coherent draft, say so truthfully instead of forcing one. This is a working
draft for a professional, not legal advice — keep the tone practical.
"""

CHAT_RULES = """
=== CHAT MODE ===
You are now answering follow-up questions in a conversation. The user has already run a
case search (its results may appear earlier in the conversation). Answer their questions
about the cases, arguments, economic models, or litigation strategy — grounded strictly
and exclusively in the case index above. The index is the entire universe you may draw
from: do not cite or discuss cases, holdings, statistics, or experts from outside it,
even ones you know well. Before saying the database lacks something, check every case
record for it. When the index genuinely does not contain what the user asks for, say so
truthfully and plainly (e.g. "Your database has no case covering X"), point to the
closest cases it does have, and suggest verifying details against the underlying case
documents (each record has a folder_url). Be concise and practical; use short paragraphs
or bullet lists. Never invent cases, holdings, citations, or experts.

When the index cannot answer but the case's underlying documents plausibly could (per the
DOCUMENT INVENTORY — e.g., an expert's methodology lives in that case's expert report):
1. Say truthfully that the curated index doesn't contain it.
2. Name the specific document(s) from the inventory that likely contain the answer, with
   their Drive links (use the file_id link pattern).
3. Offer to retrieve and read the document for them.
4. If the user accepts the offer, be honest: full-text document retrieval is the next
   product upgrade and is not yet live in this demo — so give them the direct document
   link to read now, and note that the upgraded version will read it and answer with
   page-pinpointed citations.
Never pretend to have read a PDF, and never fabricate document contents or page numbers —
you know only the document titles, not what is inside them.
"""

# ------------------------------------------------------------------- app ----

app = Flask(__name__, static_folder="static")
client = anthropic.Anthropic()  # ANTHROPIC_API_KEY from env

# Optional shared password for hosted deployments: set DEMO_PASSWORD to enable.
DEMO_PASSWORD = os.environ.get("DEMO_PASSWORD", "")


@app.before_request
def require_password():
    if not DEMO_PASSWORD:
        return None  # auth disabled (local dev)
    auth = request.authorization
    if auth and auth.password == DEMO_PASSWORD:
        return None
    return Response(
        "Access restricted — enter any username and the demo password.",
        401,
        {"WWW-Authenticate": 'Basic realm="CasePilot demo"'},
    )


def load_index(path: str) -> tuple[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    data = json.loads(raw)  # fail fast on malformed/truncated index
    lookup = {c["case_id"]: c for c in data.get("cases", [])}
    if not lookup:
        sys.exit(f"error: no cases found in {path}")
    return raw, lookup


RAW_INDEX, LOOKUP = load_index(CASES_PATH)
with open(GROUNDING_PATH, "r", encoding="utf-8") as f:
    RAW_GROUNDING = f.read()  # real per-case document inventory (titles + Drive file IDs)

# One stable block (rules + index + document inventory), byte-identical across every
# request on all endpoints, with the cache breakpoint on it — so /api/search,
# /api/chat, and /api/draft share the same warm prefix. Mode-specific text goes
# AFTER the breakpoint.
CACHED_BLOCK = {
    "type": "text",
    "text": (RULES
             + "\n\n=== CASE INDEX (JSON) ===\n" + RAW_INDEX
             + "\n\n=== DOCUMENT INVENTORY (JSON) ===\n"
             + "Actual documents in each case's Drive folder (titles and file IDs are real "
             + "and verified; per-page contents are NOT loaded — you have not read these "
             + "PDFs). Link pattern: https://drive.google.com/file/d/{file_id}/view\n"
             + RAW_GROUNDING),
    "cache_control": {"type": "ephemeral", "ttl": "1h"},  # stays warm across a demo day
}
SEARCH_SYSTEM = [CACHED_BLOCK]
CHAT_SYSTEM = [CACHED_BLOCK, {"type": "text", "text": CHAT_RULES}]
DRAFT_SYSTEM = [CACHED_BLOCK, {"type": "text", "text": DRAFT_RULES}]


def sse_stream(system, messages):
    """Stream a Claude response as Server-Sent Events (shared by /api/chat and /api/draft)."""
    def generate():
        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=16000,
                thinking={"type": "adaptive"},
                system=system,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    yield f"data: {json.dumps({'text': text})}\n\n"
                final = stream.get_final_message()
                if final.stop_reason == "refusal":
                    yield f"data: {json.dumps({'error': 'The model declined this request.'})}\n\n"
            yield "data: [DONE]\n\n"
        except anthropic.RateLimitError:
            yield f"data: {json.dumps({'error': 'Rate limited — retry in a few seconds.'})}\n\n"
        except anthropic.APIStatusError as e:
            yield f"data: {json.dumps({'error': f'API error {e.status_code}'})}\n\n"
        except anthropic.APIConnectionError:
            yield f"data: {json.dumps({'error': 'Network error reaching the Claude API.'})}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/")
def home():
    return send_from_directory("static", "index.html")


@app.post("/api/search")
def api_search():
    body = request.get_json(silent=True) or {}
    query = (body.get("query") or "").strip()
    role = (body.get("role") or "").strip()
    posture = (body.get("posture") or "").strip()
    if not query:
        return jsonify({"error": "Please describe your matter."}), 400

    parts = []
    if role:
        parts.append(f"[User's role: {role}]")
    if posture and posture != "auto":
        parts.append(f"[User's posture: {posture}]")
    parts.append(query)
    user_text = "\n\n".join(parts)

    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=SEARCH_SYSTEM,
            output_config={"format": {"type": "json_schema", "schema": RESULT_SCHEMA}, "effort": "high"},
            messages=[{"role": "user", "content": user_text}],
        )
    except anthropic.RateLimitError:
        return jsonify({"error": "Rate limited — wait a few seconds and retry."}), 429
    except anthropic.APIStatusError as e:
        return jsonify({"error": f"API error {e.status_code}: {e.message}"}), 502
    except anthropic.APIConnectionError:
        return jsonify({"error": "Network error reaching the Claude API."}), 502

    if resp.stop_reason == "refusal":
        return jsonify({"error": "The model declined this query. Rephrase and retry."}), 422
    if resp.stop_reason == "max_tokens":
        return jsonify({"error": "Response truncated (max_tokens). Retry with a narrower query."}), 502

    text = next((b.text for b in resp.content if b.type == "text"), None)
    if text is None:
        return jsonify({"error": f"No text in response (stop_reason={resp.stop_reason})."}), 502
    result = json.loads(text)  # structured outputs guarantee schema-valid JSON

    # Anti-hallucination gate + local join of authoritative fields from the index.
    verified = []
    for m in result.get("matches", []):
        rec = LOOKUP.get(m["case_id"])
        if rec is None:
            print(f"warning: dropped unknown case_id {m['case_id']!r}", file=sys.stderr)
            continue
        m["name"] = rec["name"]
        m["citation"] = rec["citation"]["cite"] if isinstance(rec["citation"], dict) else rec["citation"]
        m["folder_url"] = rec.get("folder_url", "")
        m["record_confidence"] = rec.get("confidence", "")
        m["outcome"] = rec.get("outcome", "")
        verified.append(m)
    verified.sort(key=lambda m: m["relevance"], reverse=True)
    result["matches"] = verified
    result["usage"] = {
        "input_tokens": resp.usage.input_tokens,
        "cache_read_input_tokens": resp.usage.cache_read_input_tokens,
        "cache_creation_input_tokens": resp.usage.cache_creation_input_tokens,
        "output_tokens": resp.usage.output_tokens,
    }
    return jsonify(result)


@app.post("/api/chat")
def api_chat():
    body = request.get_json(silent=True) or {}
    history = body.get("history") or []
    role = (body.get("role") or "").strip()
    if not history or history[-1].get("role") != "user":
        return jsonify({"error": "history must end with a user message"}), 400

    # Server is stateless: the client sends the whole conversation each turn,
    # including a first message that embeds the search results as context.
    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    if role:
        messages[-1] = {
            "role": "user",
            "content": f"[User's role: {role}]\n\n{messages[-1]['content']}",
        }

    return sse_stream(CHAT_SYSTEM, messages)


@app.post("/api/draft")
def api_draft():
    body = request.get_json(silent=True) or {}
    query = (body.get("query") or "").strip()
    role = (body.get("role") or "").strip()
    matches = body.get("matches") or []
    if not query or not matches:
        return jsonify({"error": "Run a search with at least one match first."}), 400

    parts = []
    if role:
        parts.append(f"[User's role: {role}]")
    parts.append(f"My matter:\n{query}")
    parts.append("Retrieved cases from my search (JSON):\n" + json.dumps(matches))
    parts.append("Please write the draft argument outline per DRAFTING MODE.")
    messages = [{"role": "user", "content": "\n\n".join(parts)}]

    return sse_stream(DRAFT_SYSTEM, messages)


def warm_cache_forever(interval_s: int = 50 * 60):
    """Re-warm the prompt cache before its 1h TTL lapses, so no visitor ever pays
    the ~45s cold start. max_tokens=0 runs prefill only — no output tokens billed;
    each warm ping re-reads the index at cache-read rates (~$0.02). Enable with
    WARM_CACHE=1 (costs roughly $0.50/day while running)."""
    while True:
        try:
            client.messages.create(
                model=MODEL,
                max_tokens=0,
                system=SEARCH_SYSTEM,  # breakpoint on the shared block — warms all endpoints
                messages=[{"role": "user", "content": "warmup"}],
            )
            print("cache warm ping ok", file=sys.stderr)
        except Exception as e:  # never let the warmer kill the app
            print(f"cache warm ping failed: {e}", file=sys.stderr)
        time.sleep(interval_s)


if os.environ.get("WARM_CACHE") == "1":
    threading.Thread(target=warm_cache_forever, daemon=True).start()


if __name__ == "__main__":
    print(f"Loaded {len(LOOKUP)} cases from {CASES_PATH}")
    port = int(os.environ.get("PORT", "5001"))
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=port, debug=False, threaded=True)
