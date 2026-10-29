"""Order numbers (VX-NNNN) that are easy to confuse when spoken.

A number is read as two halves ("fifteen oh-eight"). Within a half, these sound alike:
- teens and tens: "fifteen" / "fifty"  (15 <-> 50)
- "oh-N" and "N-ty": "oh-eight" / "eighty"  (08 <-> 80)
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass

ORDER_NUMBER_RE = re.compile(r"^VX-[0-9]{4}$")


def _confusable_halves(half: int) -> set[int]:
    alts: set[int] = set()
    if 13 <= half <= 19:
        alts.add((half - 10) * 10)
    if half % 10 == 0 and 30 <= half <= 90:
        alts.add(half // 10 + 10)
    if 1 <= half <= 9:
        alts.add(half * 10)
    if half % 10 == 0 and 10 <= half <= 90:
        alts.add(half // 10)
    return alts


def spoken_confusables(order_number: str) -> set[str]:
    """Other VX-NNNN numbers that a listener could mishear this one as. Symmetric."""
    if not ORDER_NUMBER_RE.match(order_number):
        raise ValueError(f"not an order number: {order_number!r}")
    hi, lo = int(order_number[3:5]), int(order_number[5:7])
    out = {f"VX-{alt:02d}{lo:02d}" for alt in _confusable_halves(hi)}
    out |= {f"VX-{hi:02d}{alt:02d}" for alt in _confusable_halves(lo)}
    return out


_UNITS = {
    "zero": 0, "oh": 0, "o": 0, "nought": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
}  # fmt: skip
_TEENS = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}  # fmt: skip
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}  # fmt: skip
_REPEAT = {"double": 2, "triple": 3}


def parse_spoken_order_number(heard: str) -> str | None:
    """Digits from a typed or transcribed order number, or None if there are none.

    "VX-1508", "vx 1508", "fifteen oh eight", "one five oh eight", "double five" -> "55".
    Words that aren't numbers ("order", "um", "vx") are ignored.
    """
    tokens = re.findall(r"[a-z]+|\d+", heard.lower())
    digits: list[str] = []
    repeat = 1
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.isdigit():
            digits.append(tok)
        elif tok in _REPEAT:
            repeat = _REPEAT[tok]
            i += 1
            continue
        elif tok in _UNITS:
            digits.append(str(_UNITS[tok]) * repeat)
        elif tok in _TEENS:
            digits.append(str(_TEENS[tok]))
        elif tok in _TENS:
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            if nxt in _UNITS and _UNITS[nxt] > 0:
                digits.append(str(_TENS[tok] + _UNITS[nxt]))
                i += 1
            else:
                digits.append(str(_TENS[tok]))
        elif tok == "hundred" and digits:
            digits.append("00")
        repeat = 1
        i += 1
    return "".join(digits) or None


def _edit_distance(a: str, b: str) -> int:
    """Optimal string alignment distance (insert, delete, substitute, adjacent transpose)."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


@dataclass(frozen=True)
class OrderMatch:
    order_number: str
    score: float
    reason: str  # exact | sounds_alike | transposed | one_digit_off | partial


def match_order_numbers(heard: str, candidates: Iterable[str]) -> list[OrderMatch]:
    """Rank `candidates` (the caller's own order numbers) against a possibly misheard ID.

    Only plausible matches are returned, best first. Never searches beyond `candidates`,
    so callers must pass only orders the verified customer owns.
    """
    digits = parse_spoken_order_number(heard)
    if digits is None:
        return []
    heard_number = f"VX-{digits}" if len(digits) == 4 else None
    matches: list[OrderMatch] = []
    for candidate in set(candidates):
        target = candidate[3:]
        if digits == target:
            match = OrderMatch(candidate, 1.0, "exact")
        elif heard_number and heard_number in spoken_confusables(candidate):
            match = OrderMatch(candidate, 0.9, "sounds_alike")
        elif 3 <= len(digits) <= 5 and _edit_distance(digits, target) == 1:
            transposed = sorted(digits) == sorted(target) and len(digits) == len(target)
            match = OrderMatch(candidate, 0.7, "transposed" if transposed else "one_digit_off")
        elif len(digits) == 3 and digits in target:
            match = OrderMatch(candidate, 0.6, "partial")
        else:
            continue
        matches.append(match)
    return sorted(matches, key=lambda m: (-m.score, m.order_number))
