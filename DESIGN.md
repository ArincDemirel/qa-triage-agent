# QA Triage Agent: Design Notes

A prototype assistant for hardware validation engineers. When a CI build fails, it pulls
the failing measurements, compares them with the previous build, reads what changed,
searches past incidents, and drafts a triage note. An engineer approves it.

Built as a weekend project to explore agentic triage. Data is synthetic.

## User and questions

**User:** a validation engineer on a touch sensor team. Every morning CI has run
overnight, and she needs to know what broke and why before standup.

| Branch | Example question | Answered by |
|---|---|---|
| Compute | "What failed on build 33? How much did it move vs build 32?" | SQL tools over the results tables |
| Find | "Has the offset failed in the cold before?" | Hybrid search over past Radars and wiki pages |
| Act | "Triage build 33" | LangGraph agent that combines both |

## Data model

Two stores for two jobs: **filters fetch numbers, search finds documents.**

**Numbers (SQLite, `data/qa.db`):**
- `tests`: one row per test definition: ID (`TS-4471`), name, unit, lower/upper limits.
- `builds`: one row per CI build: build ID, date, changelog (JSON list of changes).
- `test_results`: one row per measurement: build, test, unit (`S01`–`S08`), temperature
  (-20/25/60 °C), value, passed.

Decisions:
- `test_id` is the human ID (`'TS-4471'`), the same ID used in Radars and reports. It's
  what links the numbers path to the text path.
- `passed` is stored for fast filtering and as the historical verdict. Calculations always
  **include** failures; excluding them would hide the regression (survivorship bias).
- Index on `(build_id, test_id)` because almost every tool query filters on both.

**Generator (`qa/generate.py`)** stands in for the lab and the ingest pipeline: 10 builds ×
3 tests × 8 units × 3 temperatures = 720 rows, fixed seed for reproducibility.

Planted cases (the answer key, `data/ground_truth.json`, never visible to the agent):

| Case | Build | Pattern | Cause in the changelog |
|---|---|---|---|
| R1 | 33 | all units, only -20 °C | IIR filter coefficient change |
| R2 | 35 | all units, all temperatures | scan rate 240 → 120 Hz |
| R3 | 37 | only S05, S06 | sensor lot B (second supplier) |
| F1 | 31 | one measurement | nothing (flaky) |

Hardening so triage isn't trivial:
- every build has 2 harmless distractor changes, shuffled so the culprit isn't always first;
- build 33 also contains a **plausible distractor** ("update temperature compensation
  table") that sounds related to a cold failure;
- the generator asserts that every failure in the table comes from a planted case.

## Corpus

8 markdown documents with front-matter metadata (doc ID, type, test IDs): a test plan,
a triage guide, a lot-tracking page, and 5 past Radars.

- Wording deliberately differs from the changelogs: the changelog says "IIR coefficient",
  the Radar says "increased smoothing"; "scan rate" vs "refresh frequency halved".
  Keyword search alone can't link them.
- RDAR-120455 is counter-evidence: thermal table updates only affect >50 °C, which rules
  out the distractor for a cold failure.

## Architecture

```
NUMBERS: generate.py → SQLite → tools.py (get_failures, compare_builds)
TEXT:    corpus/*.md → chunking.py → BM25 (in memory) + Chroma (vectors) → search.py
AGENT:   agent.py (LangGraph) → calls the tools → draft → human review → END
```

### Chunking (`qa/chunking.py`)
Structure-aware:
- one chunk per `##` section;
- one chunk per table row, with the header line repeated so values keep their meaning;
- a context prefix on every chunk: `doc_id | title > section`. This keeps identifiers
  that only appear in the title (like the Radar number) and makes each chunk readable
  on its own.

Result: 26 chunks. A fixed-size baseline (60 words, 15 overlap) is kept for comparison.

### Search (`qa/search.py`, `qa/vectorstore.py`)
- **BM25** with a tokenizer that keeps identifiers whole and adds their parts
  (`TS-4471` → `ts-4471`, `ts`, `4471`).
- **Vectors:** Ollama `nomic-embed-text`, stored in Chroma (HNSW, cosine). Uses the
  model's task prefixes (`search_document:` / `search_query:`). One metadata flag per
  test ID enables filters like "only TS-6002 Radars".
- **Fusion:** Reciprocal Rank Fusion, `1/(60 + rank)` summed over both lists. Merges by
  rank because BM25 scores and cosine similarities are on different scales.
- No reranker: with 26 chunks there is nothing to shortlist.

### Tools (`qa/tools.py`)
- `get_failures(build)`: what failed, and the pattern (which units, which temperatures),
  plus the changelog. The tool computes the pattern, not the model.
- `compare_builds(test, a, b, temp_c?, sensor_id?)`: stats per build from all matching
  rows, mean change, `rows_used`, and every change between the builds.
- `search_docs(query, test_id?, doc_type?)`: hybrid search.

Rules: read-only connection, parameterized SQL, errors returned as data so the model can
retry, and every number reports how many rows it came from.

### Agent (`qa/agent.py`)
LangGraph, single tool-calling agent:
```
START → agent ⇄ tools          (cycle, max 8 tool steps)
          ├→ give_up            (step limit reached)
          └→ review ⏸ → END     (interrupt: engineer approves, rejects or edits)
```
- `route()` decides the next node: no tool calls → review; tool calls and steps left →
  tools; otherwise → give up. The model proposes; the code enforces the limit.
- Human review uses LangGraph `interrupt()` with a checkpointer and a `thread_id`, so the
  run pauses and resumes from saved state.
- Every tool call is traced (tool, arguments, result preview, timing) to `runs/`.
- Model: `qwen2.5:7b` via Ollama, swappable behind `make_llm()`.

The model computes nothing. Its job is judgment: choosing which slice to query, mapping
the pattern to a likely cause using the triage guide, weighing evidence between plausible
changes, and saying "nothing caused this" when that's true.

## Findings

**1. The first real run picked the distractor (build 33).**
- It chose "update temperature compensation table".
- The trace showed it searched only `"TS-4471"` (the test ID), not the symptom, so it
  never retrieved RDAR-118204, the incident pointing to the filter.
- It did retrieve RDAR-120455, which rules the thermal change out, and concluded the
  opposite anyway. So there were two problems: a weak query and weak reasoning.
- I approved the wrong note at review. Human review only helps if the reviewer checks
  the evidence, so the review step should show citations prominently.

**2. One prompt change fixed it (in that run).**
- Changed the procedure to: search by the symptom in words, and look for evidence for
  and against each candidate change. Required a chunk_id citation for every claim about
  a past incident.
- Next run: retrieved RDAR-118204, linked "IIR coefficient" to "increased smoothing",
  cited `[RDAR-118204#s1]`, picked the correct cause.
- Caveats: one run only; it didn't explicitly rule out the thermal change; "high"
  confidence after one search looks overconfident.

**3. Engineering lesson: build the index safely.**
An embedding failure left an empty Chroma collection behind. Fixed by embedding first
and deleting the old index only after embeddings succeed.

## Evaluation (to do)

- Run every planted case (31, 33, 35, 37, plus fix build 34) several times; score the
  `LIKELY CAUSE` line against the answer key.
- Check that every number in the note appears in a tool result.
- Compare against a fixed workflow (same tools, no LLM) to see when the agent earns its cost.
- Retrieval: recall@k by query type (identifier vs natural language), BM25 vs vector vs
  hybrid, structured vs fixed-size chunking.

With 5 cases this is a smoke test, not a benchmark.

## Next steps
- Repeat runs to measure consistency; track confidence vs accuracy.
- Show evidence and citations prominently at the review step.
- Cross-encoder reranker if the corpus grows.
- Expose the tools as an MCP server so other agents and clients can share them.
- Persistent checkpointer (SQLite/Postgres) so approvals can happen later.
- Triggered automatically on CI completion.