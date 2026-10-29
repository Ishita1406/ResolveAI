import inspect
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any, get_type_hints

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from app.tools.context import ToolContext


class ToolKind(StrEnum):
    READ = "read"  # no database writes
    WRITE = "write"  # changes data; idempotency key required


def _normalize_order_number(value: Any) -> Any:
    if isinstance(value, str):
        # Models often write U+2010-2014 dashes ("VX\u20118015") instead of an ASCII hyphen.
        m = re.fullmatch(r"\s*(?:vx)?[\s\-\u2010-\u2014]*(\d{4})\s*", value, re.IGNORECASE)
        if m:
            return f"VX-{m.group(1)}"
    return value


# Field before BeforeValidator: the pattern is then both enforced and visible in the schema.
OrderNumber = Annotated[
    str,
    Field(
        pattern=r"^VX-[0-9]{4}$",
        description="Order number, e.g. VX-1508. If unsure what the caller said, "
        "use find_order_candidates first.",
    ),
    BeforeValidator(_normalize_order_number),
]
Sku = Annotated[
    str,
    Field(pattern=r"^[A-Z0-9-]{3,32}$", description="Item SKU from get_order."),
    BeforeValidator(lambda v: v.strip().upper() if isinstance(v, str) else v),
]
IdempotencyKey = Annotated[
    str,
    Field(
        min_length=8,
        max_length=64,
        pattern=r"^[A-Za-z0-9_.:-]+$",
        description="Unique per intended action; repeating it repeats nothing.",
    ),
]


class ToolInput(BaseModel):
    # Unknown fields (e.g. a hallucinated customer_id) are rejected, not ignored.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class WriteInput(ToolInput):
    idempotency_key: IdempotencyKey


class ToolOutput(BaseModel):
    pass


Handler = Callable[[ToolContext, Any], Awaitable[ToolOutput]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    kind: ToolKind
    description: str
    input_model: type[ToolInput]
    output_model: type[ToolOutput]
    handler: Handler
    requires_verification: bool


READ_TOOLS: dict[str, ToolSpec] = {}
WRITE_TOOLS: dict[str, ToolSpec] = {}


def tool(kind: ToolKind, *, requires_verification: bool = True) -> Callable[[Handler], Handler]:
    """Register `async def name(ctx, args: Input) -> Output`; the docstring is the description."""

    def register(fn: Handler) -> Handler:
        hints = get_type_hints(fn)
        input_model, output_model = hints["args"], hints["return"]
        if (kind is ToolKind.WRITE) != issubclass(input_model, WriteInput):
            raise TypeError(f"{fn.__name__}: write tools (and only they) take a WriteInput")
        registry = WRITE_TOOLS if kind is ToolKind.WRITE else READ_TOOLS
        if fn.__name__ in READ_TOOLS or fn.__name__ in WRITE_TOOLS:
            raise ValueError(f"duplicate tool name {fn.__name__}")
        registry[fn.__name__] = ToolSpec(
            name=fn.__name__,
            kind=kind,
            description=inspect.cleandoc(fn.__doc__ or ""),
            input_model=input_model,
            output_model=output_model,
            handler=fn,
            requires_verification=requires_verification,
        )
        return fn

    return register
