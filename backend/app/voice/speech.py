"""Text shaping between the agent and speech.

- `SentenceChunker`: streamed LLM tokens in, speakable sentences out, so TTS starts on the
  first sentence instead of waiting for the whole reply.
- `for_speech`: IDs read back one character at a time ("VX-8015" -> "V X, 8, 0, 1, 5"),
  which is how an order number must be confirmed on a call.
- `normalize_transcript`: a spoken order ID ("v x eight oh one five") becomes VX-8015 for
  the agent, using the Step 3 parser. Only next to "order"/"VX", so a spoken PIN or phone
  number is never turned into an order number. Mishearings ("fifteen eighty" for 1508) are
  then caught by the agent's own Step 3 fuzzy matcher (order not found -> "did you mean").
"""

import re

from app.domain.order_ids import parse_spoken_order_number

_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")
_SOFT_BREAK = re.compile(r"[,;:](?=\s)")


class SentenceChunker:
    def __init__(self, *, soft_after: int = 80, hard_after: int = 180) -> None:
        self.soft_after = soft_after
        self.hard_after = hard_after
        self._buffer = ""

    def feed(self, text: str) -> list[str]:
        self._buffer += text
        out = []
        while (end := self._boundary()) is not None:
            sentence, self._buffer = self._buffer[:end].strip(), self._buffer[end:]
            if sentence:
                out.append(sentence)
        return out

    def _boundary(self) -> int | None:
        """Where the next chunk ends: a sentence end; else, once the text is long, a comma;
        else, once it's very long, the last space (a run-on reply still starts speaking)."""
        if match := _SENTENCE_END.search(self._buffer):
            return match.end()
        if len(self._buffer) >= self.soft_after:
            soft = list(_SOFT_BREAK.finditer(self._buffer))
            if soft:
                return soft[-1].end()
        if len(self._buffer) >= self.hard_after:
            cut = self._buffer.rfind(" ", 0, self.hard_after)
            return cut if cut > 0 else self.hard_after
        return None

    def flush(self) -> list[str]:
        rest, self._buffer = self._buffer.strip(), ""
        return [rest] if rest else []


_ID = re.compile(r"\b([A-Z]{2,3})[-\u2010-\u2014 ]?(\d{4,6})\b")


def _spell_id(match: re.Match) -> str:
    letters = " ".join(match.group(1))
    digits = ", ".join(match.group(2))
    return f"{letters}, {digits}"


def for_speech(text: str) -> str:
    """Text as it should be spoken: IDs (VX-8015, TCK-000012, RF-000041) spelled out."""
    return _ID.sub(_spell_id, text)


_NUMBER_WORD = (  # word-bounded, or "o" would match the start of "one"
    r"\b(?:\d+|zero|oh|o|nought|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|"
    r"forty|fifty|sixty|seventy|eighty|ninety|hundred|double|triple)\b"
)
_SPOKEN_ORDER = re.compile(
    rf"\b(?P<anchor>v\.?\s*x\b[\s.,-]*|order(?:\s+(?:number|id|no\.?))?(?:\s+is)?[\s,:-]*)"
    rf"(?P<digits>{_NUMBER_WORD}(?:[\s,.-]+{_NUMBER_WORD}){{0,7}})",
    re.IGNORECASE,
)


def normalize_transcript(text: str) -> str:
    """Spoken order IDs next to "order"/"VX" become VX-NNNN; everything else is untouched."""

    def replace(match: re.Match) -> str:
        digits = parse_spoken_order_number(match.group("digits"))
        if not digits or len(digits) != 4:
            return match.group(0)
        anchor = match.group("anchor")
        keep = "" if re.match(r"v\.?\s*x", anchor, re.IGNORECASE) else anchor
        return f"{keep}VX-{digits}"

    return _SPOKEN_ORDER.sub(replace, text)
