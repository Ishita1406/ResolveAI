"""Guards against the model speaking or acting for the customer.

- CustomerTagGuard: the model must never write a customer turn. Streamed text that could be
  the start of a <customer_message> tag is held back; if the tag appears, the reply is cut
  there and the rest of that generation (including its tool calls) is discarded.
- ungrounded_identity: verification details must come from the customer's own words in this
  conversation, never from the model's guess.
"""

import re
from collections.abc import Iterable
from typing import Any

from app.domain.order_ids import parse_spoken_order_number

MARKERS = ("<customer_message", "</customer_message")


class CustomerTagGuard:
    def __init__(self) -> None:
        self._held = ""
        self.tripped = False

    def feed(self, text: str) -> str:
        """Text that is safe to show now. After the guard trips, always ''."""
        if self.tripped:
            return ""
        self._held += text
        lowered = self._held.lower()
        hits = [i for i in (lowered.find(m) for m in MARKERS) if i >= 0]
        if hits:
            safe, self._held, self.tripped = self._held[: min(hits)], "", True
            return safe
        keep = 0
        for marker in MARKERS:
            for k in range(min(len(marker) - 1, len(lowered)), 0, -1):
                if marker.startswith(lowered[-k:]):
                    keep = max(keep, k)
                    break
        safe = self._held[: len(self._held) - keep]
        self._held = self._held[len(self._held) - keep :]
        return safe

    def flush(self) -> str:
        """End of stream: release held text (it never became a tag)."""
        held, self._held = ("" if self.tripped else self._held), ""
        return held


def _digit_strings(text: str) -> set[str]:
    out = {re.sub(r"\D", "", text)}
    spoken = parse_spoken_order_number(text)  # "three zero two six four seven" -> 302647
    if spoken:
        out.add(spoken)
    return {d for d in out if d}


def _email_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+at\s+", "@", text)
    text = re.sub(r"\s+dot\s+", ".", text)
    return re.sub(r"\s*([@.])\s*", r"\1", text)


def ungrounded_identity(args: dict[str, Any], customer_texts: Iterable[str]) -> str | None:
    """The first verification detail the customer never said, or None if all were said."""
    texts = list(customer_texts)
    digits = set().union(*(_digit_strings(t) for t in texts)) if texts else set()

    def said_digits(value: str, last: int | None = None) -> bool:
        wanted = re.sub(r"\D", "", value)
        wanted = wanted[-last:] if last else wanted
        return bool(wanted) and any(wanted in d for d in digits)

    postal = args.get("postal_code")
    if isinstance(postal, str) and not said_digits(postal):
        return "PIN code"
    phone = args.get("phone")
    if isinstance(phone, str) and not said_digits(phone, last=10):
        return "phone number"
    email = args.get("email")
    if isinstance(email, str):
        wanted = email.strip().lower()
        if not any(wanted in _email_text(t) for t in texts):
            return "email address"
    return None
