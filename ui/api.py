"""A small client for the platform's HTTP API, used by the Streamlit pages.

Kept free of Streamlit on purpose: the UI is thin, and everything it does is a call here, so the
client can be tested on its own. The login token lives only in the caller (Streamlit session state).
"""

import json
from collections.abc import Iterator
from typing import Any

import httpx


class ApiError(Exception):
    """The API answered with an error. `detail` is safe to show the user."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"{status}: {detail}")
        self.status, self.detail = status, detail


def parse_sse(lines: Iterator[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Server-Sent Events -> (event name, data) pairs.

    event: token            ->   ("token", {"text": "Twenty "})
    data: {"text": "Twenty "}
    """
    name = None
    for line in lines:
        if line.startswith("event: "):
            name = line[7:].strip()
        elif line.startswith("data: ") and name:
            yield name, json.loads(line[6:])
            name = None


class PlatformClient:
    def __init__(
        self, base_url: str, token: str | None = None, *, http: httpx.Client | None = None
    ):
        self.http = http or httpx.Client(base_url=base_url, timeout=120)
        self.token = token

    # -- plumbing ------------------------------------------------------------------------------
    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _check(self, response: httpx.Response) -> httpx.Response:
        if response.is_error:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ApiError(response.status_code, str(detail))
        return response

    def _get(self, path: str, **params: Any) -> Any:
        return self._check(self.http.get(path, headers=self._headers, params=params)).json()

    # -- login ---------------------------------------------------------------------------------
    def login(self, email: str, password: str) -> str:
        r = self._check(self.http.post("/auth/login", json={"email": email, "password": password}))
        self.token = r.json()["access_token"]
        return self.token

    def me(self) -> dict[str, str]:
        return self._get("/me")  # type: ignore[no-any-return]

    # -- chat ----------------------------------------------------------------------------------
    def chat(
        self, question: str, conversation_id: str | None = None
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """Yield ("sources" | "token" | "done" | "error", data) as the answer streams in."""
        body: dict[str, Any] = {"question": question}
        if conversation_id:
            body["conversation_id"] = conversation_id
        with self.http.stream("POST", "/chat", json=body, headers=self._headers) as r:
            if r.is_error:
                r.read()
                self._check(r)
            yield from parse_sse(r.iter_lines())

    def conversations(self) -> list[dict[str, Any]]:
        return self._get("/conversations")  # type: ignore[no-any-return]

    def messages(self, conversation_id: str) -> list[dict[str, Any]]:
        return self._get(f"/conversations/{conversation_id}")  # type: ignore[no-any-return]

    def delete_conversation(self, conversation_id: str) -> None:
        self._check(self.http.delete(f"/conversations/{conversation_id}", headers=self._headers))

    # -- memory --------------------------------------------------------------------------------
    def memories(self) -> list[dict[str, Any]]:
        return self._get("/memories")  # type: ignore[no-any-return]

    def delete_memory(self, memory_id: str) -> None:
        self._check(self.http.delete(f"/memories/{memory_id}", headers=self._headers))

    def delete_all_memories(self) -> int:
        r = self._check(self.http.delete("/memories", headers=self._headers))
        return int(r.json()["deleted"])

    # -- documents and admin -------------------------------------------------------------------
    def documents(self) -> list[dict[str, Any]]:
        return self._get("/documents")  # type: ignore[no-any-return]

    def upload(self, filename: str, content: bytes, visibility: str = "private") -> dict[str, Any]:
        r = self._check(
            self.http.post(
                "/documents",
                files={"file": (filename, content)},
                data={"visibility": visibility},
                headers=self._headers,
            )  # fmt: skip
        )
        return r.json()  # type: ignore[no-any-return]

    def set_visibility(self, document_id: str, visibility: str) -> None:
        self._check(
            self.http.patch(
                f"/documents/{document_id}/visibility",
                json={"visibility": visibility},
                headers=self._headers,
            )  # fmt: skip
        )

    def job(self, job_id: str) -> dict[str, Any]:
        return self._get(f"/jobs/{job_id}")  # type: ignore[no-any-return]

    def admin_documents(self) -> list[dict[str, Any]]:
        return self._get("/admin/documents")  # type: ignore[no-any-return]

    def admin_jobs(self) -> list[dict[str, Any]]:
        return self._get("/admin/jobs")  # type: ignore[no-any-return]

    def admin_chunks(
        self, document_id: str, level: int | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit}
        if level is not None:
            params["level"] = level
        return self._get(f"/admin/documents/{document_id}/chunks", **params)  # type: ignore[no-any-return]

    def admin_leads(self) -> list[dict[str, Any]]:
        return self._get("/admin/leads")  # type: ignore[no-any-return]

    def admin_handoffs(self) -> list[dict[str, Any]]:
        return self._get("/admin/handoffs")  # type: ignore[no-any-return]

    def handoff_transcript(self, session_id: str) -> list[dict[str, Any]]:
        return self._get(f"/admin/handoffs/{session_id}")  # type: ignore[no-any-return]

    def resolve_handoff(self, session_id: str) -> None:
        self._check(self.http.post(f"/admin/handoffs/{session_id}/resolve", headers=self._headers))

    def inspect(self, question: str, mode: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"question": question, **({"mode": mode} if mode else {})}
        r = self._check(self.http.post("/admin/inspect", json=body, headers=self._headers))
        return r.json()  # type: ignore[no-any-return]
