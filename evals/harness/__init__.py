"""End-to-end evaluation harness: simulated customers talk to the real agent over /ws/chat,
each run against its own freshly seeded database, scored by deterministic checks first and
an LLM judge only for what code can't check. See evals/README.md."""
