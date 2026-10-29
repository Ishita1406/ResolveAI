"""Registry, validation and schema export. No database needed."""

import json
from pathlib import Path

import pytest

from app.tools.base import READ_TOOLS, WRITE_TOOLS, ToolKind, WriteInput
from app.tools.context import ToolContext
from app.tools.errors import ErrorCode
from app.tools.registry import ALL_TOOLS, call_tool
from app.tools.schema import llm_tool_definitions

pytestmark = pytest.mark.anyio

SCHEMA_SNAPSHOT = Path(__file__).resolve().parents[2] / "docs" / "tool-schemas.json"
EXPECTED_READ = {
    "verify_customer",
    "list_recent_orders",
    "get_order",
    "get_shipment_status",
    "check_return_eligibility",
    "find_order_candidates",
}
EXPECTED_WRITE = {
    "create_return",
    "issue_refund",
    "reship_order",
    "create_ticket",
    "escalate_to_human",
}


def test_read_and_write_tools_are_separate():
    assert set(READ_TOOLS) == EXPECTED_READ
    assert set(WRITE_TOOLS) == EXPECTED_WRITE
    assert not set(READ_TOOLS) & set(WRITE_TOOLS)
    assert all(s.kind is ToolKind.READ for s in READ_TOOLS.values())
    assert all(s.kind is ToolKind.WRITE for s in WRITE_TOOLS.values())


def test_write_inputs_require_idempotency_key_and_read_inputs_have_none():
    for spec in WRITE_TOOLS.values():
        assert issubclass(spec.input_model, WriteInput)
        assert "idempotency_key" in spec.input_model.model_json_schema()["required"]
    for spec in READ_TOOLS.values():
        assert "idempotency_key" not in spec.input_model.model_fields


def test_no_tool_accepts_customer_identity_or_amounts():
    for spec in ALL_TOOLS.values():
        fields = set(spec.input_model.model_fields)
        assert not fields & {"customer_id", "amount", "amount_paise"}, spec.name


def test_only_verify_and_escalate_skip_verification():
    open_tools = {s.name for s in ALL_TOOLS.values() if not s.requires_verification}
    assert open_tools == {"verify_customer", "escalate_to_human"}


def test_llm_definitions_are_self_contained_and_strict():
    defs = llm_tool_definitions()
    assert {d["function"]["name"] for d in defs} == EXPECTED_READ | EXPECTED_WRITE
    assert "$ref" not in json.dumps(defs)
    for d in defs:
        params = d["function"]["parameters"]
        assert params["type"] == "object"
        assert params["additionalProperties"] is False
        assert "idempotency_key" not in params["properties"]
        assert d["function"]["description"].startswith(("[read-only]", "[changes data"))


def test_internal_definitions_include_idempotency_key():
    defs = {d["function"]["name"]: d for d in llm_tool_definitions(include_idempotency_key=True)}
    for name in EXPECTED_WRITE:
        assert "idempotency_key" in defs[name]["function"]["parameters"]["required"]


def test_schema_snapshot_is_current():
    """Regenerate with `make tool-schemas` and review the diff when tools change."""
    assert json.loads(SCHEMA_SNAPSHOT.read_text()) == llm_tool_definitions()


# Validation happens before any database access, so no session is needed.
@pytest.fixture
def no_db_ctx() -> ToolContext:
    return ToolContext(db=None)  # type: ignore[arg-type]


async def test_unknown_tool(no_db_ctx):
    result = await call_tool(no_db_ctx, "delete_all_orders", {})
    assert not result.ok
    assert result.error.code is ErrorCode.UNKNOWN_TOOL


@pytest.mark.parametrize(
    ("tool", "args", "bad_field"),
    [
        ("get_order", {"order_number": "VX-1508", "customer_id": 7}, "customer_id"),
        ("get_order", {"order_number": "fifteen oh eight"}, "order_number"),
        ("get_order", {}, "order_number"),
        ("list_recent_orders", {"limit": 500}, "limit"),
        ("verify_customer", {"phone": "9876543210", "email": "a@b.c", "postal_code": "400001"}, ""),
        ("verify_customer", {"email": "a@example.com", "postal_code": "40001"}, "postal_code"),
        (
            "issue_refund",
            {"order_number": "VX-1508", "reason": "missing_delivery"},
            "idempotency_key",
        ),
        (
            "issue_refund",
            {"order_number": "VX-1508", "reason": "return", "idempotency_key": "key-12345"},
            "",
        ),
        (
            "issue_refund",
            {"order_number": "VX-1508", "reason": "cancellation", "idempotency_key": "key-12345"},
            "reason",
        ),
        (
            "issue_refund",
            {"order_number": "VX-1508", "reason": "goodwill", "idempotency_key": "x"},
            "idempotency_key",
        ),
        (
            "issue_refund",
            {
                "order_number": "VX-1508",
                "reason": "goodwill",
                "idempotency_key": "key-12345",
                "amount_paise": 1,
            },
            "amount_paise",
        ),
        (
            "create_ticket",
            {"category": "other", "summary": "short", "idempotency_key": "key-12345"},
            "summary",
        ),
    ],
)
async def test_invalid_arguments_are_rejected_before_running(no_db_ctx, tool, args, bad_field):
    result = await call_tool(no_db_ctx, tool, args)
    assert not result.ok
    assert result.error.code is ErrorCode.INVALID_ARGUMENTS
    fields = [e["field"] for e in result.error.details["errors"]]
    assert any(f.startswith(bad_field) for f in fields), fields
    assert "Traceback" not in result.model_dump_json()


async def test_spoken_style_order_number_is_normalized(no_db_ctx):
    # Normalization succeeds, so the call proceeds to the verification gate.
    result = await call_tool(no_db_ctx, "get_order", {"order_number": " vx 1508 "})
    assert result.error.code is ErrorCode.NOT_VERIFIED
