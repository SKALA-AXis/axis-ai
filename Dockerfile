# axis-ai API 서버 — multi-stage (builder 에서 compile-only deps 제거).
# Playwright + FlagEmbedding(torch ~2GB) 는 runtime 에만 유지.
# card-evaluator 등 경량 CronJob 은 Dockerfile.cron (axis-ai-cron 이미지) 사용.

# ── builder: compile deps + Python wheels ────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential gcc g++ \
        libpq-dev \
        zlib1g-dev libxml2-dev libxslt1-dev libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /usr/local/bin/

COPY pyproject.toml uv.lock ./

ENV UV_LINK_MODE=copy
# --extra-index-url: torch==+cpu 휠은 PyPI 가 아닌 PyTorch CPU 인덱스에만 존재
# (pyproject [tool.uv.sources] 참조 — CUDA 동봉 휠 ~2.5GB 제거)
# --index-strategy unsafe-best-match: PyTorch 인덱스가 certifi 등 공용 패키지 사본을
# 가져 first-match 가 버전 충돌함. 모든 패키지가 == 핀이라 결정성은 유지됨.
RUN uv export --frozen --no-emit-project --no-hashes --format requirements-txt -o /tmp/requirements.txt \
    && uv pip install --system --no-cache \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match \
        -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# ── runtime: Playwright system libs + 앱 코드만 ───────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl ca-certificates \
        libpq5 libxml2 libxslt1.1 libffi8 libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Playwright chromium-headless-shell binary + 시스템 의존성
# - SPA IR 사이트 (예: samsungsds.com/investor/ir_events) 크롤링 시 사용
# - chromium-headless-shell 만 (full chromium 보다 ~80MB 작음)
RUN python -m playwright install --with-deps chromium-headless-shell \
    && rm -rf /var/lib/apt/lists/* /root/.cache/pip

COPY src ./src
COPY scripts ./scripts
COPY data ./data

EXPOSE 8001

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8001"]
