# ruff: noqa: E501
"""Draws the README result charts as plain SVG (no plotting library). Numbers are copied from docs/benchmarks.md.

Run: uv run python docs/make_charts.py
One measure per chart, one hue, bars start at zero, every bar has its value written on it, the chosen setting is
the strong blue and the alternatives are the light blue. A fixed light surface keeps them readable on any theme.
"""

from pathlib import Path

OUT = Path(__file__).parent / "img"
SURFACE, INK, MUTED, GRID, BLUE, PALE = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#e4e3de",
    "#2a78d6",
    "#a9c8ee",
)


def bar_chart(
    name: str,
    title: str,
    subtitle: str,
    rows: list[tuple[str, float, bool]],
    unit: str = "",
    top: float | None = None,
    fmt: str = "{:.2f}",
) -> None:  # noqa: E501
    """rows = (label, value, highlighted). Horizontal bars, labels left, value at the bar end."""
    top = top or max(v for _, v, _ in rows) * 1.15
    left, right, row_h, head = 250, 70, 34, 70
    width, height = 760, head + row_h * len(rows) + 28
    plot = width - left - right
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="{title}. {subtitle}" font-family="system-ui, sans-serif">',
        f"<title>{title}</title><desc>{subtitle}</desc>",
        f'<rect width="{width}" height="{height}" rx="8" fill="{SURFACE}"/>',
        f'<text x="20" y="28" font-size="16" font-weight="600" fill="{INK}">{title}</text>',
        f'<text x="20" y="48" font-size="12" fill="{MUTED}">{subtitle}</text>',
    ]
    for i in range(0, 5):  # quiet grid
        x = left + plot * i / 4
        parts.append(
            f'<line x1="{x:.0f}" y1="{head - 8}" x2="{x:.0f}" y2="{height - 22}" stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.0f}" y="{height - 6}" font-size="10" text-anchor="middle" fill="{MUTED}">{top * i / 4:.2g}</text>'
        )
    for n, (label, value, strong) in enumerate(rows):
        y = head + n * row_h
        w = max(plot * value / top, 1)
        parts.append(
            f'<text x="{left - 10}" y="{y + 17}" font-size="12" text-anchor="end" fill="{INK}">{label}</text>'
        )
        parts.append(
            f'<rect x="{left}" y="{y + 5}" width="{w:.0f}" height="18" rx="4" fill="{BLUE if strong else PALE}"/>'
        )
        parts.append(
            f'<text x="{left + w + 8:.0f}" y="{y + 18}" font-size="12" font-weight="600" fill="{INK}">{fmt.format(value)}{unit}</text>'
        )
    parts.append("</svg>")
    (OUT / name).write_text("\n".join(parts))


bar_chart(
    "retrieval-modes.svg",
    "Reranking is the one clear retrieval win",
    "Recall@5 on 69 questions (unreviewed gold), B-026",
    [
        ("dense (meaning)", 0.76, False),
        ("keyword", 0.33, False),
        ("hybrid (RRF)", 0.76, False),
        ("hybrid + rerank (default)", 0.83, True),
    ],
    top=1.0,
)
bar_chart(
    "chunking.svg",
    "Heading-aware chunks beat plain splitting; size barely matters",
    "Recall@5, dense only, all of Tenant A, 87 questions, B-027",
    [
        ("recursive 400/50", 0.67, False),
        ("structured 200/25", 0.69, False),
        ("structured 300/40", 0.73, False),
        ("structured 400/50 (default)", 0.75, True),
        ("structured 400/100", 0.77, False),
        ("structured 500/50", 0.79, False),
    ],
    top=1.0,
)
bar_chart(
    "embeddings.svg",
    "Arctic-m is the best embedding model on our documents",
    "Recall@5, dense only, all of Tenant A, 87 questions, B-033 (not switched yet)",
    [
        ("nomic-embed-v1.5", 0.68, False),
        ("bge-small (current)", 0.75, True),
        ("bge-base", 0.79, False),
        ("snowflake-arctic-embed-m", 0.81, False),
    ],
    top=1.0,
)
bar_chart(
    "cost-per-answer.svg",
    "RAG costs 18x less per answer than long context",
    "US dollars per answer, finance folder (150k tokens), 17 questions, B-030",
    [("RAG (8 chunks)", 0.0033, True), ("long context (whole folder)", 0.0597, False)],
    fmt="${:.4f}",
)
bar_chart(
    "latency.svg",
    "...and answers 11x faster",
    "Median seconds per answer, same 17 questions, B-030",
    [("RAG (8 chunks)", 4.2, True), ("long context (whole folder)", 46.8, False)],
    unit=" s",
    fmt="{:.1f}",
)
print("wrote", sorted(p.name for p in OUT.glob("*.svg")))
