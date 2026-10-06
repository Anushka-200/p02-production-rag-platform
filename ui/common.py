"""Pieces shared by the Streamlit pages: the client in session state and the login form."""

import os

import streamlit as st

from ui.api import ApiError, PlatformClient

API_URL = os.environ.get("API_URL", "http://localhost:8001")
AI_NOTICE = (
    "You are chatting with an AI assistant. It answers only from your organisation's documents "
    "and cites its sources; please check them before relying on an answer."
)


def client() -> PlatformClient:
    """One client per browser session (the token stays in server-side session state)."""
    if "client" not in st.session_state:
        st.session_state.client = PlatformClient(API_URL)
    return st.session_state.client  # type: ignore[no-any-return]


def require_login() -> PlatformClient | None:
    """Show the login form until the user is in; return the client once they are."""
    api = client()
    if api.token and "me" in st.session_state:
        return api
    st.title("Sign in")
    with st.form("login"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in"):
            try:
                api.login(email, password)
                st.session_state.me = api.me()
                st.rerun()
            except ApiError as exc:
                api.token = None
                st.error(exc.detail)
    return None


def sign_out() -> None:
    for key in ("client", "me", "conversation_id", "messages"):
        st.session_state.pop(key, None)
    st.rerun()
