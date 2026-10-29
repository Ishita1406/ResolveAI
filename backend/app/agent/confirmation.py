"""Deterministic yes/no classification of the customer's reply to a proposal.

Only the customer's own words can confirm an action, and only clear ones. The model never
decides that the customer said yes. Anything not clearly yes or no is AMBIGUOUS, and the
model is asked to re-ask, so the action keeps waiting (and expires if never confirmed).
"""

import re
from enum import StrEnum


class Reply(StrEnum):
    YES = "yes"
    NO = "no"
    AMBIGUOUS = "ambiguous"


_YES = {
    "yes", "yeah", "yep", "yup", "yes please", "sure", "ok", "okay", "go ahead",
    "please go ahead", "do it", "please do", "confirm", "confirmed", "i confirm",
    "that's right", "thats right", "correct", "right", "proceed", "absolutely", "definitely",
    "alright", "all right", "sounds good", "fine", "yes go ahead", "yes do it",
    "yes that's right", "yes thats right", "yes confirm", "ok go ahead", "okay go ahead",
    "yeah go ahead", "yep go ahead", "sure go ahead", "yeah sure", "yes sure", "yeah do it",
    "yep do it", "go for it", "yes go for it", "please proceed", "yes proceed",
    "haan", "haan ji", "ji haan", "ha",
}  # fmt: skip
_NO = {
    "no", "nope", "nah", "no thanks", "no thank you", "don't", "dont", "do not",
    "cancel", "cancel it", "stop", "wait", "not now", "never mind", "nevermind",
    "don't do it", "dont do it", "no don't", "no dont", "no cancel", "hold on",
    "nahi", "nahin", "mat karo",
}  # fmt: skip
_MAX_WORDS = 6


def _normalize(text: str) -> str:
    text = text.lower().replace("\u2019", "'")  # curly apostrophe from phone keyboards
    text = re.sub(r"[^\w\s']", " ", text)
    return " ".join(text.split())


def classify_reply(text: str) -> Reply:
    """YES or NO only for a short reply that is exactly a known phrase (fillers removed)."""
    normalized = _normalize(text)
    for filler in ("please", "thanks", "thank you", "ji", "um", "uh"):
        if normalized.endswith(f" {filler}") and normalized not in _YES | _NO:
            normalized = normalized[: -len(filler) - 1]
    if not normalized or len(normalized.split()) > _MAX_WORDS:
        return Reply.AMBIGUOUS
    if normalized in _YES:
        return Reply.YES
    if normalized in _NO:
        return Reply.NO
    return Reply.AMBIGUOUS
