"""Static checks of scenario files against the seed data and the tool registry, without a
database or an LLM: a typo in an order number or a tool name fails here, not after an hour
of runs."""

from app.db.seed import build_dataset
from app.tools.registry import ALL_TOOLS
from harness.scenario import RowCount, Scenario


def problems(scenarios: list[Scenario]) -> list[str]:
    data = build_dataset()
    emails = {c.email for c in data.customers}
    phones = {c.phone for c in data.customers}
    orders = {o.order_number: o for o in data.orders}
    owner = {o.order_number: o.customer.email for o in data.orders}
    found: list[str] = []
    for s in scenarios:
        f = s.facts
        if f.email and f.email not in emails:
            found.append(f"{s.id}: unknown email {f.email}")
        if f.phone and f.phone not in phones:
            found.append(f"{s.id}: unknown phone {f.phone}")
        if f.order_number and f.order_number not in orders:
            found.append(f"{s.id}: unknown order {f.order_number}")
        if f.email and f.order_number in owner and owner[f.order_number] != f.email:
            if "other_customers_order" not in s.tags:
                found.append(
                    f"{s.id}: {f.order_number} isn't {f.email}'s (tag other_customers_order)"
                )
        verified = s.expect.verified_as
        if verified not in (None, "any") and verified not in emails:
            found.append(f"{s.id}: verified_as {verified} isn't a customer")
        calls = s.expect.tool_calls
        for want in calls.required + calls.forbidden:
            for tool in want.tool if isinstance(want.tool, list) else [want.tool]:
                if tool not in ALL_TOOLS:
                    found.append(f"{s.id}: unknown tool {tool}")
            number = want.args.get("order_number")
            if isinstance(number, str) and number not in orders:
                found.append(f"{s.id}: tool expectation names unknown order {number}")
        for kind in ("refunds", "returns", "replacements", "tickets"):
            want = getattr(s.expect.end_state, kind)
            rows = (
                want
                if isinstance(want, list)
                else [want.match]
                if isinstance(want, RowCount)
                else []
            )
            for row in rows:
                for key in ("order_number", "original_order_number"):
                    if key in row and row[key] not in orders:
                        found.append(f"{s.id}: {kind} expects unknown order {row[key]}")
                if "sku" in row and "order_number" in row and row["order_number"] in orders:
                    skus = {i.sku for i in orders[row["order_number"]].items}
                    if row["sku"] not in skus:
                        found.append(f"{s.id}: {row['sku']} isn't on {row['order_number']}")
    return found
