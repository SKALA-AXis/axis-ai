FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl ca-certificates \
        build-essential gcc g++ \
        libpq-dev \
        zlib1g-dev libxml2-dev libxslt1-dev libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# uv 설치 — pyproject.toml + uv.lock 단일 진실원으로 의존성 해소
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /usr/local/bin/

# 의존성 메타데이터 먼저 복사 → 코드 변경 시 캐시 활용
COPY pyproject.toml uv.lock ./

# uv export로 lockfile을 requirements 형식으로 변환 후 시스템 파이썬에 설치
# (--system은 venv 없이 컨테이너 내장 python에 직접 설치 — Dockerfile 관용)
# FlagEmbedding(torch ~2GB) 포함이라 빌드 시간 ~5분
ENV UV_LINK_MODE=copy
RUN uv export --frozen --no-emit-project --no-hashes --format requirements-txt -o /tmp/requirements.txt \
    && uv pip install --system --no-cache -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# 애플리케이션 코드
COPY src ./src
# v3 §5.1·§5.2: FinancialLinkerAgent가 런타임에 읽는 stub 재무 데이터
# peer_financials 테이블 마이그레이션 후 DB 조회로 전환되면 이 라인 제거
COPY data ./data

EXPOSE 8001

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8001"]
