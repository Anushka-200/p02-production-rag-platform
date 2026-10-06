"""The widget in a REAL browser (headless Chromium) on the demo page, against a real API server.

Everything here is real except the model: the API runs under uvicorn on a free port, the demo page
is served by a second web server (a different ORIGIN, as on a customer's site), the browser
enforces CORS and Shadow DOM like any visitor's browser would. Skipped when Playwright's Chromium
is not installed. (Needs libnss3/libnspr4/libasound2; on a machine without root they can be
unpacked into ~/.cache/chromium-libs, which this module adds to LD_LIBRARY_PATH.)
"""

import functools
import http.server
import os
import socket
import threading
from pathlib import Path

import pytest
import uvicorn

from app.api.main import create_app
from app.config import Settings
from app.ratelimit import MemoryRateLimiter
from scripts.create_site import create_site
from tests.test_api import FakeEmbedder, FakeStream, SharedConnection
from tests.test_search import World, vec

LOCAL_LIBS = Path.home() / ".cache/chromium-libs/usr/lib/x86_64-linux-gnu"
if LOCAL_LIBS.exists():
    os.environ["LD_LIBRARY_PATH"] = f"{LOCAL_LIBS}:{os.environ.get('LD_LIBRARY_PATH', '')}"

DEMO_DIR = Path(__file__).resolve().parents[1] / "widget" / "demo"
PUBLIC = "Acme opens at 8:30 every weekday."


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return int(s.getsockname()[1])


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:  # type: ignore[no-untyped-def]
        pass


class Stack:
    """API server + two page servers (the customer's site and a stranger's site)."""

    def __init__(self, db_schema) -> None:  # type: ignore[no-untyped-def]
        self.pieces = ["We open at 8:30 [S1]."]
        w = World(db_schema)
        ann = w.person("acme", "ann")
        w.chunk(ann, "acme", PUBLIC, vec(30), visibility="public", uri="hours.md")
        self.conn = db_schema
        self.api_port, self.site_port, self.other_port = free_port(), free_port(), free_port()
        self.key = create_site(
            db_schema, "acme", "Acme website", [f"http://localhost:{self.site_port}"]
        )
        settings = Settings(  # type: ignore[call-arg]
            _env_file=None, environment="dev", reranker_provider="none", retrieval_candidates=10
        )  # fmt: skip
        app = create_app(
            settings, connect=lambda: SharedConnection(db_schema), embedder=FakeEmbedder(),
            reranker=None, limiter=MemoryRateLimiter(),
            stream_fn=lambda messages, **kw: FakeStream(list(self.pieces)),
            complete_fn=lambda *a, **k: pytest.fail("no JSON fallback expected"),
        )  # fmt: skip
        self.server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=self.api_port, log_level="error")
        )
        threading.Thread(target=self.server.run, daemon=True).start()
        handler = functools.partial(QuietHandler, directory=str(DEMO_DIR))
        self.pages = [
            http.server.ThreadingHTTPServer(("127.0.0.1", p), handler)
            for p in (self.site_port, self.other_port)
        ]
        for page in self.pages:
            threading.Thread(target=page.serve_forever, daemon=True).start()
        for _ in range(100):
            if self.server.started:
                break
            threading.Event().wait(0.1)

    def url(self, port: int | None = None) -> str:
        return f"http://localhost:{port or self.site_port}/?key={self.key}&api=http://localhost:{self.api_port}"

    def close(self) -> None:
        self.server.should_exit = True
        for page in self.pages:
            page.shutdown()


@pytest.fixture
def stack(db_schema):  # type: ignore[no-untyped-def]
    s = Stack(db_schema)
    yield s
    s.close()


@pytest.fixture
def browser():  # type: ignore[no-untyped-def]
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # browser or its system libraries missing
            pytest.skip(f"Chromium cannot start: {str(exc)[:80]}")
        yield b
        b.close()


def open_panel(browser, stack, port=None, viewport=None):  # type: ignore[no-untyped-def]
    page = browser.new_page(viewport=viewport or {"width": 1100, "height": 800})
    page.goto(stack.url(port))
    page.click(".btn")  # the floating bubble (inside the widget's Shadow DOM)
    return page


def ask(page, text: str) -> None:  # type: ignore[no-untyped-def]
    page.fill("input[aria-label='Your question']", text)
    page.press("input[aria-label='Your question']", "Enter")


def test_a_visitor_asks_and_gets_a_cited_answer_and_can_rate_it(browser, stack) -> None:  # type: ignore[no-untyped-def]
    page = open_panel(browser, stack)
    assert "AI assistant" in page.inner_text(".note")  # the disclosure line, from the server
    assert "Acme website" in page.inner_text(".head")  # the site's name, from the server
    ask(page, "When do you open?")
    page.wait_for_selector(".m.bot >> text=We open at 8:30", timeout=15000)
    page.wait_for_selector(
        "button[aria-label='Helpful']", timeout=15000
    )  # appears with the final event
    assert "Sources: " in page.inner_text(".msgs")
    if shot := os.environ.get("WIDGET_SCREENSHOT_DIR"):
        page.screenshot(path=str(Path(shot) / "widget-answer.png"))
    page.click("button[aria-label='Helpful']")
    page.wait_for_selector("text=Thanks for the feedback.", timeout=5000)
    assert stack.conn.execute("SELECT rating FROM feedback").fetchall() == [(1,)]
    assert [
        r[0] for r in stack.conn.execute("SELECT role FROM widget_messages ORDER BY created_at, id")
    ] == ["user", "assistant"]


def test_a_second_question_continues_the_same_session_after_a_page_reload(browser, stack) -> None:  # type: ignore[no-untyped-def]
    page = open_panel(browser, stack)
    ask(page, "When do you open?")
    page.wait_for_selector(".m.bot >> text=We open at 8:30", timeout=15000)
    page.wait_for_selector(
        "button[aria-label='Helpful']", timeout=15000
    )  # final event: session saved
    page.reload()
    page.click(".btn")
    ask(page, "And on Friday?")
    page.wait_for_selector(".m.user >> text=And on Friday?")
    page.wait_for_function(
        "document.querySelector('div').shadowRoot.querySelectorAll('.m.bot').length >= 2",
        timeout=15000,
    )
    assert stack.conn.execute("SELECT count(*) FROM widget_sessions").fetchone()[0] == 1


def test_asking_for_a_person_and_leaving_details_needs_consent(browser, stack) -> None:  # type: ignore[no-untyped-def]
    page = open_panel(browser, stack)
    ask(page, "When do you open?")
    page.wait_for_selector(".m.bot >> text=We open at 8:30", timeout=15000)
    page.wait_for_selector(
        "button[aria-label='Helpful']", timeout=15000
    )  # session id is stored now
    page.click(".foot >> text=Talk to a person")
    page.wait_for_selector(".lead")
    page.fill(".lead input[aria-label='Your name']", "Jo Visitor")
    page.fill(".lead input[aria-label='Email']", "jo@example.com")
    page.click(".lead .go")  # no consent tick yet
    page.wait_for_selector(".lead .src >> text=consent", timeout=5000)
    assert stack.conn.execute("SELECT count(*) FROM leads").fetchone()[0] == 0
    page.check("#consent")
    page.click(".lead .go")
    page.wait_for_selector("text=Thank you. We have your details", timeout=5000)
    assert stack.conn.execute("SELECT name, email, consent FROM leads").fetchone() == (
        "Jo Visitor",
        "jo@example.com",
        True,
    )
    assert stack.conn.execute("SELECT handoff_requested FROM widget_sessions").fetchone() == (True,)


def test_markup_in_an_answer_is_shown_as_text_and_never_runs(browser, stack) -> None:  # type: ignore[no-untyped-def]
    stack.pieces = ["<b>bold</b><script>window.__pwned=1</script> [S1]"]
    page = open_panel(browser, stack)
    ask(page, "When do you open?")
    page.wait_for_selector(
        ".m.bot >> text=<b>bold</b>", timeout=15000
    )  # literal angle brackets are visible
    assert page.evaluate("window.__pwned") is None
    assert (
        page.evaluate(
            "document.querySelector('div').shadowRoot.querySelectorAll('.m.bot b, .m.bot script').length"
        )
        == 0
    )


def test_another_website_cannot_use_the_site_key_the_browser_blocks_it(browser, stack) -> None:  # type: ignore[no-untyped-def]
    page = open_panel(browser, stack, port=stack.other_port)  # same key, but NOT the allowed origin
    ask(page, "When do you open?")
    page.wait_for_selector(".m.bot >> text=Could not reach the server", timeout=15000)
    assert stack.conn.execute("SELECT count(*) FROM widget_sessions").fetchone()[0] == 0
    assert "AI assistant" in page.inner_text(
        ".note"
    )  # still the built-in default, no server data came through


def test_the_panel_fits_a_phone_screen(browser, stack) -> None:  # type: ignore[no-untyped-def]
    page = open_panel(browser, stack, viewport={"width": 375, "height": 667})
    box = page.locator(".panel").bounding_box()
    assert box is not None and box["x"] >= 0 and box["x"] + box["width"] <= 375 and box["y"] >= 0
    if shot := os.environ.get("WIDGET_SCREENSHOT_DIR"):
        page.screenshot(path=str(Path(shot) / "widget-phone.png"))
