"""Deterministic lookups before the model is called.

Most turns that carry an email or phone, a PIN or an order number need the same tool calls
every time: verify_customer, then get_order. Each one the model makes costs a full model
call (the whole prompt again), so the loop pulls these details out of the customer's
message itself, runs the tools through the same path a model call takes (grounding, the
policy gate, the tool log), and the model answers once with the results in front of it.

Extraction is deliberately narrow: anything ambiguous (two order numbers, digits that may
not be a PIN) is left to the model, which handles it as before.
"""

import re
from dataclasses import dataclass

from app.domain.order_ids import parse_spoken_order_number

_EMAIL = re.compile(r"[a-z0-9._%+-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}")
# Speech-to-text splits digit runs at odd places ("652432-5150", "VX441-0"): separators may
# fall between any two digits.
_PHONE = re.compile(r"(?<![\d+])(?:\+?91[\s-]?)?([6-9](?:[\s-]?\d){9})(?![\s-]?\d)")
_BARE_PIN = re.compile(r"(?<![\d-])\d{6}(?![\d-])")
_ORDER = re.compile(
    r"\bvx\s*-?\s*(\d(?:[\s-]?\d){3})(?![\s-]?\d)"
    r"|\border\b(?:\s*(?:number|no|id|is|it's|its|#|:)\.?)*\s*#?\s*(\d(?:[\s-]?\d){3})(?![\s-]?\d)",
    re.IGNORECASE,
)
_PIN_WORDS = frozenset({"pin", "pincode", "postal", "zip"})
_PIN_FILLER = frozenset({"code", "number", "no", "is", "it's", "its", "my", "the", "="})
_REPEAT = frozenset({"double", "triple"})
_TOKEN = re.compile(r"[a-z0-9']+|=")

# Words a bare "here are my details" message may contain besides the details themselves.
_SMALL_TALK = frozenset(
    "hi hello hey there good morning afternoon evening my email e-mail mail address id is "
    "it's its i'm im i am this here and pin code pincode postal zip the phone mobile number "
    "no ok okay sure thanks thank you please details are it name so um uh er verify me my "
    "account registered with of a an to for on that's thats".split()
)


def _spoken_email(text: str) -> str:
    """Lowercased, with a spoken address joined: "meera dot pillai at example dot com". A
    written ". " is left alone, so "...example.com. My PIN" doesn't become "example.com.my"."""
    text = re.sub(r"\s+at\s+", "@", text.lower())
    text = re.sub(r"\s+dot\s+", ".", text)
    return re.sub(r"\s*@\s*", "@", text)


@dataclass(frozen=True)
class Found:
    email: str | None = None
    phone: str | None = None
    pin: str | None = None
    order_number: str | None = None  # only when exactly one distinct number was given


def _is_number_token(tok: str) -> bool:
    return tok.isdigit() or tok in _REPEAT or parse_spoken_order_number(tok) is not None


def _pin(text: str) -> str | None:
    """A 6-digit PIN: digits or number words right after "PIN" ("my PIN code is three zero
    two six four seven"), else one standalone 6-digit number."""
    tokens = _TOKEN.findall(text)
    for i, tok in enumerate(tokens):
        if tok not in _PIN_WORDS:
            continue
        j = i + 1
        while j < len(tokens) and tokens[j] in _PIN_FILLER:
            j += 1
        run = []
        while j < len(tokens) and _is_number_token(tokens[j]):
            run.append(tokens[j])
            j += 1
        digits = parse_spoken_order_number(" ".join(run)) if run else None
        if digits and len(digits) == 6:
            return digits
    bare = set(_BARE_PIN.findall(text))
    return bare.pop() if len(bare) == 1 else None


def extract(text: str) -> Found:
    lowered = _spoken_email(text)
    email_match = _EMAIL.search(lowered)
    email = email_match.group(0) if email_match else None
    rest = _EMAIL.sub(" ", lowered)
    phone_match = _PHONE.search(rest)
    phone = re.sub(r"\D", "", phone_match.group(1)) if phone_match else None
    rest = _PHONE.sub(" ", rest)
    orders = {"VX-" + re.sub(r"\D", "", a or b) for a, b in _ORDER.findall(rest)}
    pin = _pin(_ORDER.sub(" ", rest))
    return Found(
        email=email,
        phone=phone,
        pin=pin,
        order_number=orders.pop() if len(orders) == 1 else None,
    )


def only_identification(text: str, found: Found, name: str | None = None) -> bool:
    """True if the message is just the verification details and small talk around them, so
    there is nothing for the model to answer beyond "you're verified"."""
    lowered = _spoken_email(text)
    rest = _PHONE.sub(" ", _EMAIL.sub(" ", lowered))
    allowed = _SMALL_TALK | set((name or "").lower().split())
    return all(t in allowed or _is_number_token(t) for t in _TOKEN.findall(rest)) and (
        found.order_number is None
    )
