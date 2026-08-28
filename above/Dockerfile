# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements-cpu.txt requirements.txt ./

RUN python -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --requirement requirements-cpu.txt \
    && /opt/venv/bin/python -m pip install --requirement requirements.txt \
    && /opt/venv/bin/python -m pip check \
    && /opt/venv/bin/python -m pip uninstall --yes pip

FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends antiword poppler-utils \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home --uid 10001 appuser

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser main.py ./

USER appuser

CMD [ "python", "main.py" ]
