# CasePilot — Antitrust Case Intelligence (Demo)

AI research assistant for economic consulting and antitrust litigation. A user
enters their **role** and describes their **matter** in plain language; CasePilot
searches a curated database of 28 landmark U.S. antitrust cases and returns:

- **Ranked precedents** — by theory of harm, market/industry, economic models,
  and litigation posture (including adverse authority, with distinguishing advice)
- **Confidence-graded arguments** (high/medium/low) — based on whether the court
  expressly approved each proposition and how closely the facts fit
- **Cautions** — arguments the courts rejected and how the losing side failed
- **Draft argument outlines** — one click turns the retrieved cases into a
  structured working draft with anticipated counterarguments and a verification
  checklist
- **Grounded Q&A chat** — follow-up questions about the retrieved cases

Built on Anthropic's Claude API (`claude-opus-4-8`).

> **Demo status:** the live deployment is password-protected. Case records are
> `public_record` provenance — verify against the underlying court documents
> (linked from every result) before professional use. Not legal advice.

## Architecture

```
Browser (static/index.html)          Flask (app.py)                Claude API
────────────────────────────         ─────────────────────         ─────────────────
role + posture + query   ──POST──►   /api/search                   claude-opus-4-8
                                     system = rules + full case    structured outputs
                                     index (prompt-cached, 1h)     (JSON schema)
        case cards       ◄────────   anti-hallucination gate:
                                     unknown case_ids dropped;
                                     names/citations/links joined
                                     locally from the index

follow-up question       ──POST──►   /api/chat      ──streaming──► same cached prefix
"draft arguments" click  ──POST──►   /api/draft     ──streaming──► same cached prefix
```

Key design decisions:

1. **Whole-database retrieval, no vector store.** The full case index rides in
   the system prompt on every call and is **prompt-cached** (one shared,
   byte-identical block across all three endpoints, 1-hour TTL). First query
   writes the cache; every query after reads it at ~10% of input price. The
   model is instructed to review *every* case before ranking.
2. **Anti-hallucination by construction, not just prompting.** The model returns
   only `case_id`s plus analysis. The server validates every ID against the
   index, drops unknown ones, and joins authoritative fields (case name,
   citation, outcome, Drive folder URL) locally — the model never echoes them.
3. **Database-only grounding.** The system prompt forbids citing any authority
   outside the index, in all three modes. When nothing in the database fits,
   the product says so plainly instead of forcing weak matches.
4. **Structured outputs** (`output_config.format` with a JSON schema) guarantee
   schema-valid search results — no free-text parsing.
5. **Stateless server.** Chat history lives in the browser and is re-sent each
   turn; the server holds no sessions.

## Repository layout

| Path | Purpose |
|---|---|
| `app.py` | Flask backend: 3 endpoints, system prompts, validation gate, password gate, cache warmer |
| `static/index.html` | Entire frontend (vanilla JS, no build step) |
| `cases_index.json` | The case database (v3, 28 cases) |
| `evidence_grounding.json` | Case → court-document mapping (Drive file IDs); foundation for a future doc-grounded research agent |
| `search_cases_v4.py` | Original CLI retrieval engine the web app grew from |
| `render.yaml` | Render deployment blueprint |
| `DEPLOY.md` | Deployment + custom-domain instructions |

## Running locally

```sh
python3 -m venv venv && venv/bin/pip install -r requirements.txt
ANTHROPIC_API_KEY=sk-ant-... venv/bin/python app.py   # → http://localhost:5001
```

Optional env vars: `DEMO_PASSWORD` (enables HTTP Basic auth), `WARM_CACHE=1`
(background thread re-warms the prompt cache every 50 min), `PORT`, `HOST`.

## Deployment

Deployed on Render from this repo — see `DEPLOY.md`. Pushing to `main`
auto-redeploys. The API key and demo password live only in Render's environment
settings, never in git.

## Roadmap

- **Document-grounded answers:** an agent layer over the actual opinions and
  expert reports (already mapped in `evidence_grounding.json`) returning
  page-pinpointed, verified citations.
- Case-selection checkboxes before drafting; exportable drafts.
- Database growth beyond antitrust; per-user accounts and usage limits.
