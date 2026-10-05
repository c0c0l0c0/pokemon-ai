# Training image, also used for TensorBoard. No CUDA base image needed: the torch wheels
# bring their own CUDA libraries and the NVIDIA Container Toolkit mounts the host driver.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.8 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp

WORKDIR /app

# Dependencies first so code changes don't reinstall torch
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY src ./src
COPY scripts ./scripts
COPY configs ./configs

ENTRYPOINT ["python", "-m", "scripts.train_selfplay"]
CMD []
