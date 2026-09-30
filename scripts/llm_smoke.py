"""One real, tiny model call (a fraction of a cent). Run: uv run python scripts/llm_smoke.py"""

from app.llm import complete
from app.log import setup_logging

setup_logging()
result = complete([{"role": "user", "content": "Reply with the single word: pong"}], max_tokens=10)
print(result)
