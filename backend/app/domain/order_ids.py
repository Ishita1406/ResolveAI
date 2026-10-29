"""Order numbers (VX-NNNN) that are easy to confuse when spoken.

A number is read as two halves ("fifteen oh-eight"). Within a half, these sound alike:
- teens and tens: "fifteen" / "fifty"  (15 <-> 50)
- "oh-N" and "N-ty": "oh-eight" / "eighty"  (08 <-> 80)
"""

import re

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
