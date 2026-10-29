"""docs/policies/*.md must agree with policy.yaml (they feed RAG; a stale number would be
read to customers as policy)."""

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from app.domain.formatting import format_inr
from tests.policy_helpers import policy_dict

DOCS_DIR = Path(__file__).resolve().parents[2] / "docs" / "policies"
DOCS = sorted(DOCS_DIR.glob("*.md"))
# Operational settings, not customer policy: no document needs to state them.
NOT_CUSTOMER_FACING = {"version", "verification", "confirmation"}

INR = re.compile(r"₹[\d,]+")
DAYS = re.compile(r"(?<!to )\b(\d+) (?:business )?days?\b")
DAY_RANGE = re.compile(r"\b(\d+) to (\d+) business days\b")


def _parse(path: Path) -> tuple[dict[str, Any], str]:
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", path.read_text(), re.DOTALL)
    assert match, f"{path.name} has no front matter"
    return yaml.safe_load(match.group(1)), match.group(2)


def _lookup(raw: dict[str, Any], dotted: str) -> Any:
    node: Any = raw
    for part in dotted.split("."):
        node = node[part]
    return node


def _leaves(raw: dict[str, Any], prefix: str = "") -> set[str]:
    out = set()
    for key, value in raw.items():
        path = f"{prefix}{key}"
        out |= _leaves(value, f"{path}.") if isinstance(value, dict) else {path}
    return out


def _is_days(key: str) -> bool:
    return "days" in key.rsplit(".", 1)[-1]


def test_expected_documents_exist():
    assert {p.stem for p in DOCS} == {"returns", "refunds", "shipping", "damaged-items"}


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.stem)
def test_declared_values_match_policy_yaml(doc):
    meta, _ = _parse(doc)
    raw = policy_dict()
    assert meta["source"] == "backend/app/policy/policy.yaml"
    for key, value in meta["values"].items():
        assert _lookup(raw, key) == value, f"{doc.name}: {key} says {value}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.stem)
def test_every_amount_and_duration_in_the_text_is_a_declared_value(doc):
    meta, body = _parse(doc)
    values = meta["values"]
    amounts = {format_inr(v * 100) for k, v in values.items() if k.endswith("_inr")}
    days = {v for k, v in values.items() if _is_days(k)}
    for amount in INR.findall(body):
        assert amount in amounts, f"{doc.name}: {amount} isn't a declared policy value"
    for n in DAYS.findall(body):
        assert int(n) in days, f"{doc.name}: '{n} days' isn't a declared policy value"
    for low, high in DAY_RANGE.findall(body):
        assert (int(low), int(high)) == (
            values["refunds.processing_business_days_min"],
            values["refunds.processing_business_days_max"],
        )


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.stem)
def test_declared_values_are_actually_stated(doc):
    meta, body = _parse(doc)
    for key, value in meta["values"].items():
        if key.endswith("_inr"):
            assert format_inr(value * 100) in body, f"{doc.name}: {key} not stated"
        elif _is_days(key):
            assert re.search(rf"\b{value} (?:business )?days?\b|\b{value} to ", body), key
        elif key.startswith("escalate_when.") and value:
            assert "specialist" in body, f"{doc.name}: {key} escalates but no specialist"


def test_customer_facing_policy_is_fully_documented():
    raw = policy_dict()
    customer_facing = {k for k in _leaves(raw) if k.split(".")[0] not in NOT_CUSTOMER_FACING}
    documented = set().union(*(_parse(d)[0]["values"] for d in DOCS))
    assert customer_facing - documented == set()
