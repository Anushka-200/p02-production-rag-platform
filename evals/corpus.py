"""Loads the pinned corpus as passages (document, locator, text), the same way ingestion sees it."""

import json
import re
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from app.ingest.models import ParsedDocument
from app.ingest.parsers import parse, parse_markdown
from app.ingest.preprocess import preprocess

RAW = Path("data/raw")
GITLAB_SLICE = [
    "people-group", "people-policies", "total-rewards", "hiring",
    "values", "communication", "leadership", "finance",
]  # fmt: skip
PDFS = {
    "irs-p15b": "irs-p15b-fringe-benefits.pdf",
    "irs-p969": "irs-p969-health-accounts.pdf",
    "opm": "opm-health-insurance-overview.pdf",
}
MIN_WORDS, MAX_WORDS = 60, 450


@dataclass
class Passage:
    corpus: str
    tenant: str
    doc: str  # source_uri
    locator: str
    title: str
    path: str  # heading path as text
    text: str
    askable: bool  # enough prose to base a question on (see eligible())


_PUNCT = str.maketrans(
    {"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "‑": "-", "\u00ad": None}
)


def _clean_for_match(text: str) -> str:
    """Make two versions of the same sentence comparable: no markup, one kind of quote and dash."""
    text = unicodedata.normalize("NFKC", text).translate(_PUNCT)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)  # [label](url) -> label
    text = re.sub(r"[*_`#>|]", " ", text)
    return " ".join(text.lower().split())


def contains_verbatim(passage_text: str, quote: str) -> bool:
    """True if the quote really appears in the passage (ignoring markup and spacing)."""
    wanted = _clean_for_match(quote).rstrip(" .;:,")  # a different final punctuation mark is fine
    return bool(wanted) and wanted in _clean_for_match(passage_text)


def eligible(text: str) -> bool:
    """A section worth asking about: enough prose, not a list of links or leftover markup."""
    words = text.split()
    if not MIN_WORDS <= len(words) <= MAX_WORDS or "{{" in text:
        return False
    alpha = sum(w.strip(".,;:()-*").isalpha() for w in words)
    links = len(re.findall(r"\]\(", text))
    return alpha / len(words) > 0.6 and links < len(words) / 8


def _passages(doc: ParsedDocument, corpus: str, tenant: str, uri: str) -> list[Passage]:
    out = []
    for s in doc.sections:
        locator = f"page:{s.page}" if s.page else s.anchor
        out.append(
            Passage(
                corpus,
                tenant,
                uri,
                locator,
                doc.title,
                " > ".join(s.heading_path),
                s.text,
                eligible(s.text),
            )
        )
    return out


def load_markdown_corpus(
    root: Path, prefix: str, corpus: str, tenant: str, dirs: list[str] | None = None
) -> dict[str, list[Passage]]:
    files = sorted(
        f for d in (dirs or [""]) for f in (root / d).rglob("*") if f.suffix in {".md", ".mdx"}
    )
    result: dict[str, list[Passage]] = {}
    for f in files:
        uri = f"{prefix}/{f.relative_to(root).as_posix()}"
        doc = preprocess(parse_markdown(f.read_bytes(), f.name))
        if ps := _passages(doc, corpus, tenant, uri):
            result[uri] = ps
    return result


def load_gitlab() -> dict[str, list[Passage]]:
    return load_markdown_corpus(
        RAW / "gitlab-handbook/content/handbook", "gitlab", "gitlab", "A", GITLAB_SLICE
    )


def load_posthog() -> dict[str, list[Passage]]:
    return load_markdown_corpus(
        RAW / "posthog-handbook/contents/handbook", "posthog", "posthog", "B"
    )


def load_pdfs() -> dict[str, list[Passage]]:
    result = {}
    for corpus, name in PDFS.items():
        path = RAW / "hr-pdfs" / name
        doc = preprocess(parse("pdf", path.read_bytes(), name))
        # one passage per page is enough: PDF sections are already page-bound
        result[name] = _passages(doc, corpus, "A", name)
    return result


def write_manifest(path: Path) -> dict[str, object]:
    """Pin exactly which version of each corpus the eval was written against."""

    def sha(repo: str) -> str:
        return subprocess.run(
            ["git", "-C", str(RAW / repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    import hashlib

    manifest = {
        "gitlab_handbook_commit": sha("gitlab-handbook"),
        "gitlab_slice_dirs": GITLAB_SLICE,
        "posthog_handbook_commit": sha("posthog-handbook"),
        "pdf_sha256": {
            n: hashlib.sha256((RAW / "hr-pdfs" / n).read_bytes()).hexdigest() for n in PDFS.values()
        },
        "note": "Gold answers are only valid against these exact versions. Wiki-sync demos replay LATER commits separately.",
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def askable(corpus: dict[str, list[Passage]]) -> dict[str, list[Passage]]:
    """Only the sections worth asking about, and only the pages that have any."""
    out = {uri: [p for p in ps if p.askable] for uri, ps in corpus.items()}
    return {uri: ps for uri, ps in out.items() if ps}
