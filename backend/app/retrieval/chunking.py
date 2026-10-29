"""Split policy documents into heading sections.

A document is YAML front matter (machine-readable values, checked against policy.yaml by the
tests and left out of the chunks), one "# Title", then "## Section" headings. Each section
becomes one chunk: the sections are short and each answers one question, so splitting them
further would only separate a rule from its exceptions.
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

_FRONT_MATTER = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
_HEADING = re.compile(r"^(#{1,2}) +(.+?) *$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    document: str  # file stem, e.g. "refunds"
    title: str  # the document's H1, e.g. "Refunds policy"
    section: str  # the H2, e.g. "Are there limits on refunds?"
    section_id: str  # slug of the H2, unique within the document
    position: int
    content: str
    version: str  # hash of the whole document text

    @property
    def key(self) -> str:
        """Stable reference used for citations and the retrieval test set."""
        return f"{self.document}#{self.section_id}"

    def embedding_text(self) -> str:
        # Headings like "What can I get?" repeat across documents; the title disambiguates.
        return f"{self.title}: {self.section}\n\n{self.content}"


def slugify(text: str) -> str:
    text = text.lower().replace("\u2019", "").replace("'", "")
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


def document_version(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def chunk_document(name: str, text: str) -> list[Chunk]:
    version = document_version(text)
    body = _FRONT_MATTER.sub("", text, count=1)
    headings = list(_HEADING.finditer(body))
    title = next((m.group(2) for m in headings if m.group(1) == "#"), name)
    sections = [m for m in headings if m.group(1) == "##"]

    parts: list[tuple[str, str]] = []
    first = sections[0].start() if sections else len(body)
    intro = re.sub(r"^# .*$", "", body[:first], count=1, flags=re.MULTILINE).strip()
    if intro:
        parts.append(("Overview", intro))
    for i, match in enumerate(sections):
        end = sections[i + 1].start() if i + 1 < len(sections) else len(body)
        parts.append((match.group(2), body[match.end() : end].strip()))

    chunks: list[Chunk] = []
    seen: dict[str, int] = {}
    for heading, content in parts:
        if not content:
            continue
        slug = slugify(heading)
        seen[slug] = seen.get(slug, 0) + 1
        if seen[slug] > 1:
            slug = f"{slug}-{seen[slug]}"
        chunks.append(
            Chunk(
                document=name,
                title=title,
                section=heading,
                section_id=slug,
                position=len(chunks),
                content=content,
                version=version,
            )
        )
    return chunks


def load_documents(directory: Path) -> list[Chunk]:
    """Every section of every *.md in `directory`, in a stable order."""
    paths = sorted(directory.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"no policy documents (*.md) in {directory}")
    return [chunk for path in paths for chunk in chunk_document(path.stem, path.read_text())]


def policy_documents_text(directory: Path) -> str:
    """Every policy document without its front matter, for full-text mode. The same text
    that retrieval chunks, so the two modes see identical policy wording."""
    paths = sorted(directory.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"no policy documents (*.md) in {directory}")
    return "\n\n".join(_FRONT_MATTER.sub("", p.read_text(), count=1).strip() for p in paths)
