"""Helpers shared by all parsers."""

import re


def slugify(text: str) -> str:
    """Heading text -> URL-style tag: "Parental Leave & Return!" -> "parental-leave-return"."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"


def table_to_markdown(rows: list[list[str]]) -> str:
    """Rows of cells -> a pipe table. Every format (HTML, PDF, DOCX) uses this one style.

    The first row is treated as the header. Newlines inside a cell become spaces and a literal
    pipe is escaped, so one table row is always exactly one line.
    """
    clean = [
        [re.sub(r"\s+", " ", (c or "")).strip().replace("|", "\\|") for c in row] for row in rows
    ]
    clean = [r for r in clean if any(r)]
    if not clean:
        return ""
    width = max(len(r) for r in clean)
    clean = [r + [""] * (width - len(r)) for r in clean]
    lines = ["| " + " | ".join(clean[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in clean[1:]]
    return "\n".join(lines)


def join_wrapped_lines(lines: list[str]) -> str:
    """Join the lines of one PDF paragraph, repairing words hyphenated at the line end.

    "em-" + "ployee" -> "employee". Only when the next line starts lowercase, so "Pub. 15-" +
    "B" stays apart. Limitation: a real compound split at the line end ("self-" + "employed")
    becomes "selfemployed".
    """
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if out.endswith("-") and line[:1].islower():
            out = out[:-1] + line
        elif out:
            out += " " + line
        else:
            out = line
    return out
