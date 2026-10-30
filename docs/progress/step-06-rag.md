# Step 06: Policy retrieval (RAG)

Retrieval over `docs/policies/`:
- An ingestion script, which chunks by heading and embeds with a local model into pgvector.
- Hybrid search: PostgreSQL full-text search plus vector similarity, merged with reciprocal rank fusion.
- A `search_policies` tool whose results cite their section.
- A config flag that switches to putting the full policy text in the system prompt.
- A 30-question test set, and a script that reports recall@1/3/5, MRR and latency for vector, full-text and hybrid search.

**Measured baseline** (`evals/retrieval/results.md`, bge-small-en-v1.5, 30 questions, 21 sections):

| Method | Recall@1 | Recall@3 | Recall@5 | MRR | Mean latency |
|---|---|---|---|---|---|
| Vector only | 0.600 | 0.900 | 1.000 | 0.764 | 11.7 ms |
| Full-text only | 0.400 | 0.633 | 0.700 | 0.552 | 1.3 ms |
| Hybrid (RRF) | 0.633 | 0.767 | 0.900 | 0.741 | 13.1 ms |

**On this corpus and test set, hybrid is not better than vector-only** except at rank 1. See Trade-offs and Open items.

## Changes made

### Retrieval package (`backend/app/retrieval/`, new)

| File | Why |
|---|---|
| `__init__.py` | Package marker. |
| `chunking.py` | <ul><li>Parses each `docs/policies/*.md` and drops the YAML front matter, which holds machine-readable values already checked against policy.yaml.</li><li>Each `##` section becomes one `Chunk` with document, title (the H1), section heading, `section_id` (a slug unique within the document; duplicates get `-2`), position, content and `version` (a sha256 prefix of the whole document).</li><li>Text before the first `##` becomes an "Overview" chunk.</li><li>`embedding_text()` puts the document title before the heading, because "What can I get?" appears in three documents.</li><li>`policy_documents_text()` builds the text for full-text mode.</li></ul> |
| `embedding.py` | <ul><li>An `Embedder` protocol.</li><li>`FastEmbedder` loads the model lazily, under a lock; `warm()` loads it off the event loop.</li><li>It adds the configurable query prefix to queries only.</li><li>It raises a clear error if a model's dimension doesn't match the 384-dimension column.</li><li>`embedder_from_settings()` builds one from settings.</li></ul> |
| `search.py` | <ul><li>`PolicyRetriever` has `vector_ranking` (cosine distance via pgvector) and `text_ranking` (`ts_rank_cd` over a generated `tsvector`, matching **any** query term).</li><li>`ranked()` runs only what a mode needs: text mode never embeds the query.</li><li>`reciprocal_rank_fusion()` with k = 60 and deterministic tie-breaking.</li><li>`search()` returns `Passage`s with a citation.</li><li>`IndexEmptyError` when nothing is indexed for the current model.</li></ul> |
| `ingest.py` | `python -m app.retrieval.ingest [--force]`. Idempotent per document, in one transaction: <ul><li>an unchanged document (same version and model) is skipped;</li><li>a changed one is deleted and re-embedded;</li><li>a document no longer on disk is removed;</li><li>a new embedding model re-embeds everything.</li></ul> |
| `evaluate.py` | `python -m app.retrieval.evaluate [--output] [--query-prefix] [--questions]`. <ul><li>Ranks every chunk in each mode and computes recall@1/3/5, MRR, and mean and p95 latency.</li><li>Writes a Markdown report with each question's rank in each mode.</li><li>Refuses a test set that refers to sections not in the index.</li></ul> |

### Storage

| File | Why |
|---|---|
| `backend/app/db/models.py` | New `PolicyChunk` model with fields document, title, section, section_id, position, content, version, embedding_model, `embedding vector(384)`, and a stored, generated `tsv`. The `tsv` gives the heading weight A, the text B and the document title C. Unique on (document, section_id). It has a GIN index on `tsv` and an HNSW index (`vector_cosine_ops`) on `embedding`. `EMBEDDING_DIM = 384`. |
| `backend/migrations/versions/20261029_0005_policy_chunks.py` | Creates the table and both indexes. The dimension and `tsvector` expression are copied rather than imported, so the migration can't change later. `alembic check` reports no drift. |

### Agent and tool

| File | Why |
|---|---|
| `backend/app/tools/policy_tools.py` (new) | <ul><li>`search_policies(query)` is a read tool that needs no verification: it's general information, which the brief allows for unverified callers.</li><li>It returns `passages: [{source, section_id, text}]`, where `source` reads like "Refunds policy: Goodwill refunds" so answers can cite it.</li><li>`search_unavailable` when retrieval is off or the index is empty.</li></ul> |
| `backend/app/tools/registry.py` | Imports the new module so the tool is registered. |
| `backend/app/tools/context.py` | `ToolContext.retriever` (optional). |
| `backend/app/tools/errors.py` | New `SEARCH_UNAVAILABLE` error code. |
| `backend/app/agent/loop.py` | <ul><li>`Agent(retriever=..., policy_documents=...)`.</li><li>`search_policies` is offered before verification too, and only when a retriever exists; full-text mode doesn't offer it.</li><li>The retriever is passed into each turn's `ToolContext`.</li></ul> |
| `backend/app/agent/prompts.py` | <ul><li>Loads the optional `policy_context/<version>.yaml`, which is included in the prompt fingerprint.</li><li>`render_system_prompt(..., retrieval=, policy_documents=)` fills `$policy_context`.</li><li>Asking for a policy mode with a prompt version that has no policy section raises an error rather than silently leaving out the policy text.</li></ul> |
| `backend/app/main.py` | `build_agent` builds the retriever from `POLICY_CONTEXT_MODE` (in retrieval mode) or loads the documents (in full-text mode). The lifespan warms the embedding model, so the first caller doesn't wait for it to load. |
| `backend/app/agent/cli.py` | Warms the model at startup. |
| `backend/app/config.py` | <ul><li>`policy_context_mode` (`retrieval` or `full_text`) and `policy_docs_dir`.</li><li>`embedding_model`, `embedding_cache_dir` and `embedding_query_prefix`.</li><li>`retrieval_search_mode` (`hybrid` by default, or `vector` or `text`).</li><li>`retrieval_top_k=3`, `retrieval_candidates=20` and `retrieval_rrf_k=60`.</li><li>The default prompt is now v3.</li></ul> |

### Prompts (`backend/prompts/`)

| File | Why |
|---|---|
| `system/v3.md` (new) | v2 with its fixed policy summary (return window, refund timeline, missing threshold) replaced by `$policy_context`. That way each mode's policy knowledge comes only from its own mechanism, and the benchmark compares the two mechanisms rather than each one plus a cheat sheet. |
| `policy_context/v3.yaml` (new) | <ul><li>Two texts, `retrieval` and `full_text`, with the same instructions: answer only from policy text, name the policy, and quote numbers exactly or say you'll check.</li><li>Retrieval mode adds "one search per question".</li><li>Full-text mode wraps the documents in `<policy_documents>`.</li></ul> |
| `turn_context/v3.md`, `notices/v3.yaml` (new) | The same as v2 apart from the version: each version is a complete set of files. |

### Evaluation (`evals/retrieval/`, new)

| File | Why |
|---|---|
| `questions.yaml` | 30 questions in a customer's words, each with its correct `document#section`. All 21 sections are covered. Written before any evaluation ran. |
| `results.md`, `results-no-prefix.md` | Baseline reports with and without BGE's query instruction. |
| `README.md` | How to run it, what the metrics mean, and the rules for the test set. |

### Build, configuration and CI

| File | Why |
|---|---|
| `backend/pyproject.toml`, `uv.lock` | `fastembed` (0.9.1, ONNX Runtime with no PyTorch) and `pgvector` (0.5.1, the SQLAlchemy `Vector` type). |
| `Makefile` | `ingest-policies` and `eval-retrieval` targets. |
| `.env.example` | `AGENT_PROMPT_VERSION=v3`, `POLICY_CONTEXT_MODE`, `EMBEDDING_MODEL`, `RETRIEVAL_SEARCH_MODE` and `RETRIEVAL_TOP_K`, plus commented-out path settings. |
| `.env` (yours) | Only `AGENT_PROMPT_VERSION` changed, from v2 to v3, because v2 has no policy section and the agent now refuses to start with it in either mode. No other line was touched. |
| `docker-compose.yml` | The backend mounts `docs/policies` read-only and a named volume for the model cache, and sets their container paths. |
| `backend/Dockerfile` | Creates the cache directory owned by the `app` user; otherwise the named volume would be created root-owned and unwritable. The image builds. |
| `.github/workflows/ci.yml` | Caches the model directory, and fetches the model before the tests so the real-model quality test runs in CI. |
| `docs/tool-schemas.json` | Regenerated with `search_policies`. |

### Tests (481 pass; 38 new, plus updated expectations)

| File | What |
|---|---|
| `tests/retrieval_helpers.py` (new) | `HashingEmbedder`: deterministic hashed bag-of-words vectors at 384 dimensions. The plumbing tests need no model download. |
| `test_retrieval_chunking.py` (new, 7) | Sections and slugs, duplicate headings, empty sections dropped, front matter excluded, version tracks the text, title in the embedding text, the real documents (21 sections), full-text assembly. |
| `test_retrieval_search.py` (new, 16) | <ul><li>RRF maths and ties.</li><li>Ingestion is idempotent; `--force`; a changed document is replaced and a removed one deleted; a new model re-embeds.</li><li>The generated `tsv`.</li><li>Hybrid results with citations; each mode uses its own ranking, and text mode doesn't embed.</li><li>Stop-word-only queries fall back to vector.</li><li>Chunks from another model are ignored.</li><li>The tool needs no verification; `search_unavailable`; argument validation; the configured mode.</li><li>The query prefix is applied to queries only; a wrong dimension fails clearly.</li></ul> |
| `test_retrieval_eval.py` (new, 6) | <ul><li>The test set has 30 questions covering every section.</li><li>`score` and `percentile`.</li><li>The evaluation runs every mode.</li><li>It refuses sections missing from the index.</li><li>**A quality floor with the real model:** vector R@5 ≥ 0.9, hybrid R@5 ≥ 0.85, hybrid R@3 ≥ 0.75, hybrid MRR ≥ 0.65 and text R@5 ≥ 0.6. Skipped if the model isn't cached.</li></ul> |
| `test_build_agent.py` (new, 3) | Both modes are wired from settings; v2 with a policy mode fails at startup. |
| `test_migrations.py` (new, 1) | The models match the migrations. Mutation-checked: changing a column length makes it fail. |
| `test_agent_loop.py` | Two new tests: retrieval mode (search offered before verification, passages with sources) and full-text mode (documents in the system prompt, no search tool). The tools-offered expectation now excludes `search_policies` when there's no retriever. |
| `test_agent_prompts.py` | Three new tests: v3's policy section per mode; a policy mode with v2 fails; `policy_context` is part of the fingerprint. |
| `test_tool_registry.py`, `test_chat_ws.py`, `agent_helpers.py` | Tool lists include `search_policies`; `FakeAgent.retriever`; `make_agent(retriever=, policy_documents=)`. |

## Trade-offs

- **Local bge-small-en-v1.5 through fastembed (your choice).**
  - **For:** no API key, no per-call cost, deterministic, works offline and in CI; a query embeds in about 10 ms on CPU; ONNX Runtime rather than PyTorch keeps the image smaller.
  - **Against:** a 65 MB download on first use, and a 384-dimension column. A model of another size needs a migration; the embedder checks the dimension and says so.
  - **Rejected:** hosted embedding APIs, because Groq has none and each search would add a network round trip and a key.
- **One chunk per `##` section, with no size-based splitting.** Sections are 60 to 460 characters and each answers one question. Splitting further would separate a rule from its exceptions (for example "Above either limit, a specialist reviews…"). Overlapping windows would only add near-duplicates on a 21-section corpus.
- **The document title is prepended to the embedded text,** because "What can I get?" appears in three documents and is meaningless without it. Full-text search gets the same effect by giving the title weight C.
- **The version is a content hash per document,** not a counter or the policy.yaml version. Re-ingesting is then a no-op unless the text changed, and every passage says exactly which text it came from. Old versions are replaced, not kept: the agent must never cite superseded policy, and git holds the history.
- **The embedding model is stored per row, and searches filter on it.** After changing the model without re-ingesting, search reports "index empty" rather than comparing vectors from two models.
- **Full-text search matches any query term.**
  - `plainto_tsquery` requires all terms, and natural questions rarely contain every term of the answer.
  - `websearch_to_tsquery` also defaults to AND.
  - With OR, `ts_rank_cd` still favours sections matching more terms, closer together, and in the heading.
- **Fusion in Python rather than one SQL statement.**
  - Two small queries cost about 1 ms more.
  - The evaluation can score vector-only, text-only and hybrid from exactly the same rankings.
  - RRF uses ranks only, so cosine distance and `ts_rank_cd`, which are on unrelated scales, never need normalising.
  - k = 60 is the usual constant from the RRF paper.
- **Hybrid stays the default although it measured worse at k = 3,** because the step asked for hybrid. The evaluation exists to make that kind of call from data. `RETRIEVAL_SEARCH_MODE=vector` switches it with no code change.
- **`search_policies` needs no verification,** because the brief allows general information before verification. It returns passage text and sources only: no scores (noise for the model) and no customer data.
- **Prompt v3 rather than editing v2.** v2 is what produced the step-05 transcripts, so it stays reproducible. v3 removes the fixed policy summary: with it, retrieval mode would rarely search, and the benchmark would measure the summary.
- **Both modes get identical instructions** apart from where the text comes from and retrieval's "one search per question". A benchmark then compares the mechanisms, not two prompts.
- **Hashing embedder in the tests, plus one real-model test.**
  - The plumbing tests need neither a 65 MB download nor ONNX.
  - Quality is only meaningful with the real model, so one floor test uses it and skips if the model isn't cached.
  - CI caches and fetches the model, so the test runs there.
- **Prompt cost:**
  - Full-text mode adds about 5.3k characters (around 1.3k tokens) to *every* model call, which matters on Groq's 8K TPM free tier.
  - Retrieval adds about 570 characters of tool schema, plus about 900 characters of passages, only on calls that search.

## Challenges

- **Hybrid underperformed vector-only on recall.**
  - Full-text search misses paraphrases: "send something back" against "return", "compensation as an apology" against "goodwill".
  - When it misses or ranks a section low, equal-weight RRF pushes a correct vector hit down, for example t6: vector rank 4, full-text 15, hybrid 9.
  - I report this rather than tuning the test set or the fusion to the 30 questions; the per-question ranks are in `results.md`.
- **A live run gave a vague answer to a two-part question** ("how long to return, and when do I get my money").
  - The model searched once; hybrid's top 3 left out the refund-timing section; and it filled the gap with "a few business days".
  - **Fix (prompt):** search once per question, and give numbers exactly as the passages state them or say you'll check.
  - **Re-run:** two searches and the exact figures, "30 days" and "5 to 7 business days".
- **A bind parameter inside `literal_column` isn't a bind parameter.** The first version of the any-term query would have sent `:q` to PostgreSQL literally. I rebuilt it from SQL functions (`plainto_tsquery`, `replace`, then a cast to `tsquery`) and gave `'english'::regconfig` a fixed type.
- **fastembed doesn't add BGE's query instruction.** It's now an explicit setting, and the evaluation measured both ways:
  - with the instruction, vector MRR is 0.764 against 0.747;
  - without it, hybrid MRR is 0.752 against 0.741.
  - The difference is within noise for 30 questions. The default follows BGE's model card.
- **The named volume would have been root-owned** in the container, which runs as `app`. The Dockerfile now creates the directory first; the image builds.
- **My file-writing tool turned `’` escapes into literal curly quotes,** which ruff rejects (RUF001). I restored the escapes with sed.
- **There was no test that models match migrations.** I added one with a new migration in this step, and checked that it fails on a deliberate column change.

## Open items

- **Choose the search mode from data.** Vector-only is currently better at the agent's k = 3 (0.900 against 0.767). Options:
  - make `vector` the default;
  - weight RRF towards vector;
  - add query expansion or synonyms to full-text search.

  The test set should grow first, to about 100 questions including keyword-style ones such as "₹499" or "RF-", where full-text search should help. 30 questions over 21 sections can't separate methods reliably.
- **Benchmark retrieval against full-text mode** end to end with the agent (answer accuracy, tokens and latency), as the step defers. The token difference is already clear.
- **Citations in the UI:** passages carry `source` and `section_id`, but no event sends them to the WebSocket client yet. Add one when the frontend exists.
- **Re-ingest automatically:** run ingestion at startup or in CI when `docs/policies` changes. For now, `make ingest-policies` is manual, and search reports an empty index clearly.
- **Brief target:** the brief asks for Recall@5 ≥ 90%. Hybrid is at exactly 0.900 and vector-only at 1.000, on this small set.
