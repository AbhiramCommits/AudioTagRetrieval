# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1
RUN python -m venv /venv

# CPU-only torch wheels keep the image small (no CUDA).
RUN /venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 torchaudio==2.11.0

COPY pyproject.toml README.md ./
COPY audiotag ./audiotag
# torch is already satisfied, so `pip install .` only adds the remaining deps.
RUN /venv/bin/pip install .

FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home appuser

COPY --from=builder /venv /venv
ENV PATH=/venv/bin:$PATH

WORKDIR /app
COPY artifacts ./artifacts
RUN chown -R appuser:appuser /app

USER appuser
EXPOSE 8000

ENV AUDIOTAG_SERVE_DEVICE=cpu \
    AUDIOTAG_MODEL=cnn \
    AUDIOTAG_INDEX_TYPE=flat \
    AUDIOTAG_BACKEND=torch \
    AUDIOTAG_TORCH_THREADS=2 \
    AUDIOTAG_LOG_LEVEL=info

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)" || exit 1

CMD ["sh", "-c", "exec uvicorn audiotag.api.main:app --host 0.0.0.0 --port 8000 --workers 2 --log-level ${AUDIOTAG_LOG_LEVEL:-info}"]
