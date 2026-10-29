"""search_policies: hybrid retrieval over the policy documents. General information, so it
needs no verification. The passages explain policy; they never decide a case (the policy
engine does that when an action is proposed)."""

from pydantic import BaseModel, Field

from app.retrieval.search import IndexEmptyError
from app.tools.base import ToolInput, ToolKind, ToolOutput, tool
from app.tools.context import ToolContext
from app.tools.errors import ErrorCode, ToolFailure


class SearchPoliciesInput(ToolInput):
    query: str = Field(
        min_length=3, max_length=300, description="The policy question, in plain words."
    )


class PolicyPassage(BaseModel):
    source: str = Field(description="Document and section, to cite.")
    section_id: str
    text: str


class SearchPoliciesOutput(ToolOutput):
    passages: list[PolicyPassage]


def _unavailable(hint: str) -> ToolFailure:
    return ToolFailure(
        ErrorCode.SEARCH_UNAVAILABLE,
        "I can't look up our policies right now.",
        hint=hint,
    )


@tool(ToolKind.READ, requires_verification=False)
async def search_policies(ctx: ToolContext, args: SearchPoliciesInput) -> SearchPoliciesOutput:
    """Search the store's policies (returns, refunds, shipping and delivery, damaged or wrong
    items) for a general policy question. Returns the most relevant sections with their
    source. Whether a specific order qualifies is decided by the order and action tools."""
    if ctx.retriever is None:
        raise _unavailable("Policy search is turned off. Offer escalate_to_human if needed.")
    try:
        passages = await ctx.retriever.search(ctx.db, args.query)
    except IndexEmptyError as exc:
        raise _unavailable(
            "The policy index is empty (run the ingestion script). Offer escalate_to_human."
        ) from exc
    return SearchPoliciesOutput(
        passages=[
            PolicyPassage(source=p.citation, section_id=p.key, text=p.content) for p in passages
        ]
    )
