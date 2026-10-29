import random
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.seed import ANCHOR
from app.repositories.customers import get_customer_by_email
from app.tools.context import ToolContext
from app.tools.registry import call_tool


def make_ctx(db: AsyncSession) -> ToolContext:
    return ToolContext(db=db, clock=lambda: ANCHOR, rng=random.Random(7))


async def verify_as(ctx: ToolContext, email: str) -> int:
    """Verify `ctx` as the customer with this email via the real tool; returns customer id."""
    customer = await get_customer_by_email(ctx.db, email)
    assert customer is not None
    postal_code = customer.postal_code
    result = await call_tool(ctx, "verify_customer", {"email": email, "postal_code": postal_code})
    assert result.ok, result.error
    return ctx.state.customer_id


def new_key() -> str:
    return f"test-{uuid.uuid4().hex}"
