# Retrieval evaluation

Measures how well policy search finds the section that answers a question.

| File | What it is |
|---|---|
| `questions.yaml` | 30 policy questions, each with its one correct section (`<document>#<heading slug>`). Covers all 21 sections in `docs/policies/`. |
| `results.md` | Latest report with BGE's query instruction (the default `EMBEDDING_QUERY_PREFIX`). |
| `results-no-prefix.md` | The same, with the query embedded as is (`--query-prefix ""`). |

## Running it

```bash
make migrate            # creates policy_chunks (migration 0005)
make ingest-policies    # embeds docs/policies/*.md; the first run downloads the model
make eval-retrieval     # prints the report and writes results.md
```

The script is `backend/app/retrieval/evaluate.py`. It runs against `POSTGRES_DB`, ranks
every chunk for each question in each mode, and reports:

- **Recall@k:** the share of questions whose correct section is in the top k. Recall@3
  matters most, because `search_policies` gives the agent 3 passages.
- **MRR:** the mean of 1/rank, where 0 means the method didn't return the section.
  Full-text only returns sections that contain a query term.
- **Latency:** in process, per query: query embedding (vector and hybrid) and SQL.

## Rules for the test set

- Questions are phrased as a customer would ask them, and mostly don't reuse the heading's words.
- Don't change questions to improve the numbers. Add new ones instead, and keep the old.
- `tests/test_retrieval_eval.py` fails if a question points to a section that no longer
  exists, or if a section has no question.
