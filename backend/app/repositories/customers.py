import re

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Customer


def normalize_phone(raw: str) -> str | None:
    """Indian mobile number to E.164 (+91XXXXXXXXXX), or None if it isn't one."""
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    if len(digits) != 10 or digits[0] not in "6789":
        return None
    return f"+91{digits}"


async def get_customer(session: AsyncSession, customer_id: int) -> Customer | None:
    return await session.get(Customer, customer_id)


async def get_customer_by_email(session: AsyncSession, email: str) -> Customer | None:
    stmt = select(Customer).where(Customer.email == email.strip().lower())
    return await session.scalar(stmt)


async def get_customer_by_phone(session: AsyncSession, phone: str) -> Customer | None:
    normalized = normalize_phone(phone)
    if normalized is None:
        return None
    return await session.scalar(select(Customer).where(Customer.phone == normalized))


async def find_customers_by_name(session: AsyncSession, full_name: str) -> list[Customer]:
    """Case-insensitive exact match. Names are not unique; never verify on name alone."""
    stmt = (
        select(Customer)
        .where(func.lower(Customer.full_name) == " ".join(full_name.lower().split()))
        .order_by(Customer.id)
    )
    return list(await session.scalars(stmt))
