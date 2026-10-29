"""Splitting docs/policies/*.md into heading sections. No database needed."""

from app.config import POLICY_DOCS_DIR
from app.retrieval.chunking import (
    chunk_document,
    document_version,
    load_documents,
    policy_documents_text,
    slugify,
)

DOC = """---
policy: sample
values:
  refunds.per_order_cap_inr: 10000
---

# Sample policy

Read this first.

## What can I get?

A refund.

## What can I get?

Still a refund.

## Empty section

## Isn't it "simple"?

Yes: **really**.
"""


def test_sections_become_chunks_with_title_and_stable_ids():
    chunks = chunk_document("sample", DOC)
    assert [(c.section, c.section_id) for c in chunks] == [
        ("Overview", "overview"),
        ("What can I get?", "what-can-i-get"),
        ("What can I get?", "what-can-i-get-2"),  # duplicate headings stay addressable
        ('Isn\'t it "simple"?', "isnt-it-simple"),  # empty sections are dropped
    ]
    assert {c.title for c in chunks} == {"Sample policy"}
    assert [c.position for c in chunks] == [0, 1, 2, 3]
    assert chunks[1].key == "sample#what-can-i-get"
    assert chunks[3].content == "Yes: **really**."


def test_front_matter_is_not_indexed():
    text = " ".join(c.content for c in chunk_document("sample", DOC))
    assert "per_order_cap" not in text and "policy: sample" not in text


def test_version_tracks_the_document_text():
    before = chunk_document("sample", DOC)[0].version
    assert before == document_version(DOC)
    assert chunk_document("sample", DOC + "\nMore.\n")[0].version != before


def test_embedding_text_carries_the_title_for_generic_headings():
    chunk = chunk_document("sample", DOC)[1]
    assert chunk.embedding_text().startswith("Sample policy: What can I get?\n\n")


def test_slugify():
    assert slugify("My parcel is late. When does it count as missing?") == (
        "my-parcel-is-late-when-does-it-count-as-missing"
    )
    assert slugify("The carrier says it was delivered, but I didn\u2019t get it") == (
        "the-carrier-says-it-was-delivered-but-i-didnt-get-it"
    )


def test_real_policy_documents():
    chunks = load_documents(POLICY_DOCS_DIR)
    assert {c.document for c in chunks} == {"damaged-items", "refunds", "returns", "shipping"}
    assert len({c.key for c in chunks}) == len(chunks) == 21
    assert all(c.content and not c.content.startswith("#") for c in chunks)


def test_full_text_mode_gets_every_document_without_front_matter():
    text = policy_documents_text(POLICY_DOCS_DIR)
    assert text.count("\n# ") + text.startswith("# ") == 4
    assert "source: backend/app/policy/policy.yaml" not in text
    for chunk in load_documents(POLICY_DOCS_DIR):
        assert chunk.content in text
