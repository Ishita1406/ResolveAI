"""The only entry point the agent uses: call_tool(ctx, name, arguments) -> ToolResult."""

import hashlib
import json
import logging
from typing import Any

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.db.models import ToolInvocation
from app.tools import read_tools, write_tools  # noqa: F401 - registers the tools
from app.tools.base import READ_TOOLS, WRITE_TOOLS, ToolKind, ToolSpec, WriteInput
from app.tools.context import ToolCallRecord, ToolContext
from app.tools.errors import ErrorCode, ToolError, ToolFailure, ToolResult

log = logging.getLogger(__name__)

ALL_TOOLS: dict[str, ToolSpec] = {**READ_TOOLS, **WRITE_TOOLS}


def _validation_error(exc: ValidationError) -> ToolError:
    problems = [
        {"field": ".".join(str(p) for p in e["loc"]) or "(arguments)", "problem": e["msg"]}
        for e in exc.errors()
    ]
    return ToolError(
        code=ErrorCode.INVALID_ARGUMENTS,
        message="Sorry, I need to double-check a detail before I can do that.",
        hint="Fix the arguments: " + "; ".join(f"{p['field']}: {p['problem']}" for p in problems),
        details={"errors": problems},
    )


def _request_hash(args: WriteInput) -> str:
    payload = args.model_dump(mode="json", exclude={"idempotency_key"})
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


async def _run_write(ctx: ToolContext, spec: ToolSpec, args: WriteInput) -> ToolResult:
    """Claim the key, run, store the result, commit: all in one transaction.

    A concurrent call with the same key blocks on the unique index until this one commits,
    then replays its stored result. If this call fails, the claim rolls back with it.
    """
    request_hash = _request_hash(args)
    claimed = await ctx.db.scalar(
        insert(ToolInvocation)
        .values(
            idempotency_key=args.idempotency_key,
            tool_name=spec.name,
            customer_id=ctx.state.customer_id,
            conversation_id=ctx.conversation_id,
            request_hash=request_hash,
        )
        .on_conflict_do_nothing(index_elements=["idempotency_key"])
        .returning(ToolInvocation.id)
    )
    if claimed is None:
        prior = await ctx.db.scalar(
            select(ToolInvocation).where(ToolInvocation.idempotency_key == args.idempotency_key)
        )
        same = (
            prior is not None
            and prior.tool_name == spec.name
            and prior.request_hash == request_hash
            and prior.customer_id == ctx.state.customer_id
        )
        response = prior.response if same else None  # read before rollback expires `prior`
        await ctx.db.rollback()
        if not same:
            raise ToolFailure(
                ErrorCode.IDEMPOTENCY_KEY_REUSED,
                "Sorry, something went wrong on my side with that request.",
                hint="This idempotency key was already used for a different action. "
                "Use a new key for a new action.",
            )
        return ToolResult(tool=spec.name, ok=True, data=response, replayed=True)

    output = await spec.handler(ctx, args)
    data = output.model_dump(mode="json")
    stored = await ctx.db.get(ToolInvocation, claimed)
    stored.response = data
    await ctx.db.commit()
    return ToolResult(tool=spec.name, ok=True, data=data)


async def call_tool(ctx: ToolContext, name: str, arguments: dict[str, Any]) -> ToolResult:
    spec = ALL_TOOLS.get(name)
    if spec is None:
        error = ToolError(
            code=ErrorCode.UNKNOWN_TOOL,
            message="Sorry, I can't do that.",
            hint=f"No tool named {name!r}. Available: {', '.join(sorted(ALL_TOOLS))}.",
        )
        return ToolResult(tool=name, ok=False, error=error)

    result = await _dispatch(ctx, spec, arguments)
    ctx.state.history.append(
        ToolCallRecord(
            tool=name,
            ok=result.ok,
            error_code=result.error.code if result.error else None,
            at=ctx.now(),
        )
    )
    return result


async def _dispatch(ctx: ToolContext, spec: ToolSpec, arguments: dict[str, Any]) -> ToolResult:
    try:
        args = spec.input_model.model_validate(arguments)
    except ValidationError as exc:
        return ToolResult(tool=spec.name, ok=False, error=_validation_error(exc))
    if spec.requires_verification:
        # Gate before any database access: an unverified call never touches data.
        try:
            ctx.require_customer_id()
        except ToolFailure as failure:
            return ToolResult(tool=spec.name, ok=False, error=failure.error)

    try:
        if spec.kind is ToolKind.WRITE:
            return await _run_write(ctx, spec, args)
        output = await spec.handler(ctx, args)
        data = output.model_dump(mode="json")
        await ctx.db.rollback()  # end the read transaction; don't sit idle in one between turns
        return ToolResult(tool=spec.name, ok=True, data=data)
    except ToolFailure as failure:
        await ctx.db.rollback()
        return ToolResult(tool=spec.name, ok=False, error=failure.error)
    except Exception:
        # Full detail goes to the server log only; the agent gets a generic, speakable error.
        log.exception("tool %s failed", spec.name)
        await ctx.db.rollback()
        error = ToolError(
            code=ErrorCode.INTERNAL_ERROR,
            message="Sorry, something went wrong on our side.",
            hint="Retry once. If it fails again, call escalate_to_human with reason tool_failure.",
            retryable=True,
        )
        return ToolResult(tool=spec.name, ok=False, error=error)
