"""One real, tiny model call, traced in Langfuse (a fraction of a cent).

Run: uv run python -m scripts.llm_smoke
"""

from app.llm import complete
from app.log import setup_logging
from app.tracing import flush, get_client

setup_logging()
client = get_client()
print("Langfuse:", "keys valid" if client and client.auth_check() else "not configured (no traces)")
result = complete(
    [{"role": "user", "content": "Reply with the single word: pong"}],
    name="smoke-test",
    max_tokens=10,
)
print(result)
flush()  # a short script exits before the background sender does, so send explicitly
