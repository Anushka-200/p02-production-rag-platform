"""Admin console: retrieval inspector, ingestion status, chunk previews, upload and sharing.

Only organisation admins get in (the API refuses everyone else; the page also hides itself).
"""

import time

import pandas as pd
import streamlit as st

from ui.api import ApiError
from ui.common import require_login

st.set_page_config(page_title="Admin", page_icon="🛠", layout="wide")


def table(rows: list[dict], columns: list[str]) -> None:  # type: ignore[type-arg]
    st.dataframe(pd.DataFrame(rows, columns=columns), width="stretch", hide_index=True)


def inspector(api) -> None:  # type: ignore[no-untyped-def]
    st.caption(
        "Type a question: see every candidate with its dense, keyword, fused and rerank scores."
    )
    question = st.text_input("Question", key="inspect-q")
    mode = st.selectbox("Retrieval mode", ["hybrid", "dense", "keyword"])
    if st.button("Inspect") and question:
        try:
            result = api.inspect(question, mode)
        except ApiError as exc:
            st.error(exc.detail)
            return
        t = result["timings_ms"]
        st.write(
            f"mode **{result['mode']}** · keywords `{result['keyword_query']}` · "
            f"context {result['context_sources']} sources / {result['context_tokens']} tokens · "
            f"timings {t}"
        )
        rows = [
            {
                "used": c["context_label"] or "",
                "fused": c["fused_rank"],
                "dense rank": c["dense_rank"],
                "dense dist": c["dense_distance"],
                "kw rank": c["keyword_rank"],
                "kw score": c["keyword_score"],
                "rrf": c["rrf_score"],
                "rerank rank": c["rerank_rank"],
                "rerank": c["rerank_score"],
                "document": c["document"],
                "section": c["section"],
                "pages": c["pages"],
            }
            for c in result["candidates"]
        ]
        table(rows, list(rows[0]) if rows else [])
        for c in result["candidates"][:10]:
            with st.expander(f"#{c['fused_rank']} {c['document']} — {c['section']}"):
                st.write(c["text_preview"])


def ingestion(api) -> None:  # type: ignore[no-untyped-def]
    with st.expander("Upload a document", expanded=False):
        file = st.file_uploader("File", type=["pdf", "docx", "md", "mdx", "txt", "html"])
        visibility = st.selectbox("Who can read it", ["private", "org", "public"], index=1)
        if st.button("Upload") and file is not None:
            try:
                job = api.upload(file.name, file.getvalue(), visibility)
            except ApiError as exc:
                st.error(exc.detail)
            else:
                st.success(f"queued (job {job['job_id'][:8]}…); refresh to see progress")
    st.subheader("Documents")
    docs = api.admin_documents()
    table(
        docs,
        [
            "title",
            "source_uri",
            "visibility",
            "status",
            "chunks",
            "active_version",
            "owner",
            "error",
            "updated_at",
        ],
    )
    st.subheader("My ingestion jobs")
    table(
        api.admin_jobs(),
        [
            "source_uri",
            "visibility",
            "status",
            "stage",
            "attempts",
            "flagged_by_scan",
            "error",
            "finished_at",
        ],
    )
    st.caption(
        "flagged_by_scan = the ingestion scan found text aimed at the AI; review before trusting."
    )


def chunk_previews(api) -> None:  # type: ignore[no-untyped-def]
    docs = api.admin_documents()
    if not docs:
        st.info("No documents yet.")
        return
    labels = {f"{d['source_uri']} ({d['chunks']} chunks)": d["id"] for d in docs}
    choice = st.selectbox("Document", list(labels))
    level = st.radio("Level", ["all", "children (searched)", "parents (context)"], horizontal=True)
    lv = {"all": None, "children (searched)": 0, "parents (context)": 1}[level]
    for c in api.admin_chunks(str(labels[choice]), level=lv, limit=50):
        pages = f" · p.{c['page_start']}" if c["page_start"] else ""
        with st.expander(
            f"[{c['level']}.{c['chunk_index']}] {c['section_path']}{pages} · {c['chars']} chars"
        ):
            st.write(c["preview"])
            if c["blurb"]:
                st.caption(f"context blurb: {c['blurb']}")
            st.caption(f"anchors {c['anchors']} · embedded: {c['embedded']}")


def main() -> None:
    api = require_login()
    if api is None:
        return
    if st.session_state.me["role"] != "admin":
        st.warning("The admin console is for organisation admins.")
        return
    st.title("Admin console")
    tab1, tab2, tab3 = st.tabs(["Retrieval inspector", "Ingestion status", "Chunk previews"])
    with tab1:
        inspector(api)
    with tab2:
        ingestion(api)
    with tab3:
        chunk_previews(api)
    if st.button("Refresh"):
        time.sleep(0.1)
        st.rerun()


main()
