"""Show the journey of ONE document from raw file to rows in Postgres, stage by stage.

Run: uv run python -m scripts.trace_document <file> <source_uri> "<a question about it>"
Uses the real parsers, chunker and embedding model; reads (never writes) the database.
"""

import sys
from pathlib import Path

import psycopg

from app.config import get_settings
from app.embeddings import get_embedder, get_token_counter, vector_literal
from app.ingest.chunkers import chunk_document
from app.ingest.contextual import embedding_text
from app.ingest.parsers import parse
from app.ingest.preprocess import preprocess
from app.ingest.upload import validate_upload


def show(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def clip(text: str, n: int = 150) -> str:
    return repr(text[:n] + ("..." if len(text) > n else ""))


def main(path: str, uri: str, question: str) -> None:
    s = get_settings()
    raw = Path(path).read_bytes()
    kind = validate_upload(Path(path).name, raw, s.max_upload_mb * 1_000_000)

    show("STAGE 0  RAW FILE (bytes on disk)")
    print(f"{path}  |  {len(raw):,} bytes  |  type {kind}")
    for line in raw.decode(errors="replace").splitlines()[:9]:
        print("   ", line[:110])

    show("STAGE 1  PARSE: raw text -> title + sections (heading path, anchor, text)")
    parsed = parse(kind, raw, uri)
    print(
        f"title={parsed.title!r}  metadata keys={list(parsed.metadata)}  sections={len(parsed.sections)}"
    )
    for sec in parsed.sections[:3]:
        print(
            f"   path={' > '.join(sec.heading_path)!r}  anchor={sec.anchor!r}\n      text={clip(sec.text, 110)}"
        )

    show("STAGE 2  PREPROCESS: remove footers, duplicates, invisible characters")
    cleaned = preprocess(parsed)
    print(
        "   ",
        cleaned.metadata["preprocess"],
        f"| sections {len(parsed.sections)} -> {len(cleaned.sections)}",
    )

    show(
        f"STAGE 3  CHUNK ({s.chunk_strategy}, {s.chunk_tokens} tokens, counted with the embedding model's tokenizer)"
    )
    count = get_token_counter(s)
    chunks = chunk_document(cleaned, s, count)
    for c in chunks:
        role = "PARENT" if c.level == 1 else "child "
        print(
            f"   index={c.index:2} {role} parent_index={c.parent_index!s:4} tokens={c.token_count:3} covers={list(c.anchors)[:3]}"
        )
    child = next(c for c in chunks if c.level == 0 and c.token_count > 60)
    print(
        f"\n   the text of child {child.index} (what the model will see):\n      {clip(child.text, 330)}"
    )

    show("STAGE 4  BLURB (contextual retrieval): off in this run (no spend)")
    print(
        "    when on, text = blurb + chunk; the blurb is stored in chunks.context_blurb and keyword-indexed"
    )

    show("STAGE 5  EMBED: child text -> 384 numbers (parents are not embedded)")
    embedder = get_embedder(s)
    vector = embedder.embed_documents([embedding_text(child)])[0]
    print(
        f"    model={embedder.model}  length={len(vector)}  first 6 numbers={[round(x, 3) for x in vector[:6]]}"
    )

    show("STAGE 6  POSTGRES: what is really stored")
    with psycopg.connect(s.database_url("owner")) as conn:
        doc = conn.execute(
            "SELECT id, title, status, active_version, content_hash FROM documents WHERE source_uri = %s",
            (uri,),
        ).fetchone()
        print(
            f"    documents: id={str(doc[0])[:8]} title={doc[1]!r} status={doc[2]} active_version={doc[3]} hash={doc[4][:10]}..."
        )
        print("    chunks (id | level | parent_id | section_path | embedded):")
        for r in conn.execute(
            "SELECT id, level, parent_id, section_path, embedding IS NOT NULL FROM chunks "
            "WHERE document_id = %s ORDER BY chunk_index",
            (doc[0],),
        ):
            print(
                f"      {str(r[0])[:8]} | {r[1]} | {str(r[2])[:8] if r[2] else 'NULL':8} | {r[3][:60]!r} | {r[4]}"
            )

        show(f"STAGE 7  SEARCH PROOF for: {question!r}")
        qv = vector_literal(embedder.embed_query(question))
        print("    dense (cosine distance, smaller = closer):")
        for r in conn.execute(
            "SELECT d.source_uri, c.section_path, c.embedding <=> %s::vector AS dist FROM chunks c "
            "JOIN documents d ON d.id = c.document_id WHERE c.level = 0 AND c.embedding IS NOT NULL "
            "ORDER BY dist LIMIT 3",
            (qv,),
        ):
            print(f"      {r[2]:.3f}  {r[0].split('/')[-1]}  >  {r[1][-55:]}")
        print("    keyword (ts_rank, bigger = better):")
        for r in conn.execute(
            "SELECT d.source_uri, c.section_path, ts_rank(c.tsv, q) AS rank FROM chunks c "
            "JOIN documents d ON d.id = c.document_id, websearch_to_tsquery('english', %s) q "
            "WHERE c.level = 0 AND c.tsv @@ q ORDER BY rank DESC LIMIT 3",
            (" OR ".join(w.strip("?.,") for w in question.split() if len(w) > 3),),
        ):
            print(f"      {r[2]:.3f}  {r[0].split('/')[-1]}  >  {r[1][-55:]}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
