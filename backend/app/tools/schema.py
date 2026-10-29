"""Tool definitions for LLM tool calling (OpenAI-compatible format, as used by Groq).

    python -m app.tools.schema > ../docs/tool-schemas.json

idempotency_key is left out of the LLM-facing schema by default: the agent loop derives it
from the confirmed action and adds it before calling call_tool, so a model retry can never
mint a fresh key for the same action.
"""

import copy
import json
from typing import Any

from app.tools.base import ToolKind, ToolSpec
from app.tools.registry import ALL_TOOLS

_KIND_NOTE = {
    ToolKind.READ: "[read-only]",
    ToolKind.WRITE: "[changes data: only call after the customer has explicitly confirmed]",
}


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Replace $ref with the referenced definition; some tool-calling APIs reject $defs."""
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                target = copy.deepcopy(defs[node["$ref"].split("/")[-1]])
                extra = {k: v for k, v in node.items() if k != "$ref"}
                return resolve({**target, **extra})
            return {k: resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


def _compact(node: Any) -> Any:
    """Fewer tokens per request: drop generated titles, and write an optional field as its
    plain type (anyOf [X, null] with default null) since optional just means not required."""
    if isinstance(node, list):
        return [_compact(v) for v in node]
    if not isinstance(node, dict):
        return node
    node = {k: _compact(v) for k, v in node.items() if k != "title"}
    variants = node.get("anyOf")
    if (
        isinstance(variants, list)
        and len(variants) == 2
        and {"type": "null"} in variants
        and node.get("default", None) is None
    ):
        (inner,) = [v for v in variants if v != {"type": "null"}]
        rest = {k: v for k, v in node.items() if k not in ("anyOf", "default")}
        node = {**inner, **rest}
    return node


def parameters_schema(spec: ToolSpec, *, include_idempotency_key: bool = False) -> dict[str, Any]:
    schema = _compact(_inline_refs(spec.input_model.model_json_schema()))
    if not include_idempotency_key:
        schema.get("properties", {}).pop("idempotency_key", None)
        required = [r for r in schema.get("required", []) if r != "idempotency_key"]
        if required:
            schema["required"] = required
        else:
            schema.pop("required", None)
    return schema


def llm_tool_definitions(*, include_idempotency_key: bool = False) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": f"{_KIND_NOTE[spec.kind]} {spec.description}",
                "parameters": parameters_schema(
                    spec, include_idempotency_key=include_idempotency_key
                ),
            },
        }
        for spec in sorted(ALL_TOOLS.values(), key=lambda s: (s.kind != ToolKind.READ, s.name))
    ]


if __name__ == "__main__":
    print(json.dumps(llm_tool_definitions(), indent=2, ensure_ascii=False))
