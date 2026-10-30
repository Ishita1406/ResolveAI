"""Semantic end of turn: how much silence ends the customer's turn, given what they've said.

VAD alone cuts people off when they pause mid-thought ("my order number is... VX...").
After speech stops we look at the transcript so far: if it sounds finished we wait a short
silence, otherwise a longer one. Deterministic and instant (no model call), so the extra
wait is only paid when the words suggest more is coming.
"""

import re

from app.domain.order_ids import parse_spoken_order_number

# A turn ending on one of these words is almost never finished.
CONTINUATIONS = frozenset(
    """and or but so because then also plus the a an my your our their this that these is
    are was were be been am to of for with from in on at by as if um uh er erm like order
    number pin code phone email it's its i'm i i'd id me wait hmm""".split()
)
# Nouns that often come right before details ("my order... VX 8015", "my phone... 98...") but
# also end finished requests ("where is my order"). They end a turn only when it's clearly a
# finished request: STT closed it with punctuation, or it asks or asks for something.
DETAIL_NOUNS = frozenset("order orders number pin code phone email".split())
REQUEST_WORDS = frozenset(
    """where when what why how which who status track tracking help cancel return refund
    replace change check late delayed arrived missing damaged wrong""".split()
)
_DIGITS_NEEDED = (  # an identifier being dictated: done only when it has enough digits
    (re.compile(r"\b(order|vx|v x)\b"), 4),
    (re.compile(r"\b(pin|postal|post code|zip)\b"), 6),
    (re.compile(r"\b(phone|mobile|number is)\b"), 10),
)


def looks_complete(text: str) -> bool:
    words = re.findall(r"[a-z0-9']+", text.lower())
    if not words:
        return False
    if words[-1] in CONTINUATIONS and not _finished_request(text, words):
        return False
    tail = " ".join(words[-8:])
    for pattern, needed in _DIGITS_NEEDED:
        matches = list(pattern.finditer(tail))
        if not matches:
            continue
        digits = parse_spoken_order_number(tail[matches[-1].end() :]) or ""
        if 0 < len(digits) < needed:
            return False  # "PIN is three zero two" -> still dictating
    return True


def _finished_request(text: str, words: list[str]) -> bool:
    """True for "where is my order" or "help with my order.", not for "it's order"."""
    if words[-1] not in DETAIL_NOUNS:
        return False
    return text.rstrip()[-1:] in "?.!" or any(w in REQUEST_WORDS for w in words[:-1])


def silence_needed_ms(text: str, *, complete_ms: int, incomplete_ms: int) -> int:
    return complete_ms if looks_complete(text) else incomplete_ms
