FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY . .

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /home/app/.cache/resolveai && chown -R app /home/app/.cache
USER app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"]
