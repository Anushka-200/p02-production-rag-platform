# One image for the API, the ingestion worker and the Streamlit UI (the compose file picks the command).
FROM python:3.12-slim

# uv installs exactly the versions in uv.lock
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf FASTEMBED_CACHE_PATH=/models/fastembed

# dependencies first, so a code change does not reinstall them
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY app ./app
COPY ui ./ui
COPY scripts ./scripts
COPY migrations ./migrations
COPY widget ./widget
COPY evals ./evals

# the app never runs as root
RUN useradd --create-home --uid 10001 rag && mkdir -p /app/data/uploads /models && chown -R rag /app/data /models
USER rag

EXPOSE 8001 8502
CMD ["uvicorn", "app.api.main:_default_app", "--factory", "--host", "0.0.0.0", "--port", "8001"]
