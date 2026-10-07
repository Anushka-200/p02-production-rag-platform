"""Chat page: ask questions, see sources, keep conversations, view and delete what is remembered.

Run: uv run streamlit run ui/Chat.py --server.port 8502   (API must be running on API_URL)
"""

import streamlit as st

from ui.api import ApiError
from ui.common import AI_NOTICE, describe_steps, require_login, sign_out

st.set_page_config(page_title="Document assistant", page_icon="💬", layout="wide")


def show_sources(sources: list[dict]) -> None:  # type: ignore[type-arg]
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            pages = f", pages {s['pages']}" if s.get("pages") else ""
            fresh = f" · last updated {s['updated']}" if s.get("updated") else ""
            st.markdown(f"**[{s['id']}]** `{s['document']}` — {s['section']}{pages}{fresh}")


def render_history(api, conversation_id: str) -> None:  # type: ignore[no-untyped-def]
    st.session_state.messages = [
        {"role": m["role"], "content": m["content"], "sources": m.get("citations") or []}
        for m in api.messages(conversation_id)
    ]


def main() -> None:
    api = require_login()
    if api is None:
        return
    me = st.session_state.me

    with st.sidebar:
        st.caption(f"Signed in as {me['email']} ({me['role']})")
        if st.button("New chat", width="stretch"):
            st.session_state.conversation_id = None
            st.session_state.messages = []
            st.rerun()
        st.subheader("Conversations")
        for c in api.conversations()[:15]:
            if st.button(c["title"] or "(untitled)", key=f"conv-{c['id']}", width="stretch"):
                st.session_state.conversation_id = str(c["id"])
                render_history(api, str(c["id"]))
                st.rerun()
        if st.session_state.get("conversation_id") and st.button("Delete this conversation"):
            api.delete_conversation(st.session_state.conversation_id)
            st.session_state.conversation_id, st.session_state.messages = None, []
            st.rerun()
        with st.expander("What the assistant remembers about you"):
            facts = api.memories()
            if not facts:
                st.caption(
                    "Nothing yet. Lasting facts you mention (role, preferences) appear here."
                )
            for f in facts:
                left, right = st.columns([5, 1])
                left.write(f["fact"])
                if right.button("✕", key=f"mem-{f['id']}", help="forget this"):
                    api.delete_memory(str(f["id"]))
                    st.rerun()
            if facts and st.button("Forget everything"):
                api.delete_all_memories()
                st.rerun()
        if st.button("Sign out"):
            sign_out()

    st.title("Document assistant")
    st.info(AI_NOTICE, icon="🤖")
    st.session_state.setdefault("messages", [])
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                show_sources(m["sources"])

    question = st.chat_input("Ask about your documents")
    if not question:
        return
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        sources: list[dict] = []  # type: ignore[type-arg]
        done: dict = {}  # type: ignore[type-arg]
        placeholder = st.empty()
        shown = ""
        try:
            for event, data in api.chat(question, st.session_state.get("conversation_id")):
                if event == "sources":
                    sources = data["sources"]
                elif event == "token":
                    shown += data["text"]
                    placeholder.markdown(shown + "▌")
                elif event == "done":
                    done = data
                elif event == "error":
                    st.error(data["message"])
        except ApiError as exc:
            st.error(exc.detail)
            return
        final = done.get("text", shown)
        placeholder.markdown(final)  # the validated answer replaces whatever streamed
        cited = [s for s in sources if s["id"] in {c["id"] for c in done.get("citations", [])}]
        if cited:
            show_sources(cited)
        if done.get("agent"):
            with st.expander("How I searched"):
                for line in describe_steps(done["agent"]):
                    st.markdown(f"- {line}")
        if done:
            t = done["timings_ms"]
            refused = "Refused: not in the documents. " if done.get("refused") else ""
            st.caption(
                f"{refused}first token {t.get('first_token_ms', 0) / 1000:.1f}s · "
                f"total {t.get('total_ms', 0) / 1000:.1f}s · ${done['cost_usd']:.4f}"
            )
            st.session_state.conversation_id = done.get("conversation_id")
    st.session_state.messages.append({"role": "assistant", "content": final, "sources": cited})


main()
