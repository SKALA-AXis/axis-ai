FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# FlagEmbedding(torch ~2GB)은 빌드 시간 문제로 분리
# 실제 임베딩 기능 사용 시 FlagEmbedding을 별도 레이어로 추가
RUN pip install --no-cache-dir \
    "fastapi>=0.115,<0.120" \
    "uvicorn[standard]>=0.27" \
    "sqlalchemy>=2.0" \
    "psycopg2-binary>=2.9" \
    "qdrant-client>=1.9" \
    "langchain>=0.2" \
    "langchain-core>=0.2" \
    "langchain-openai>=0.1" \
    "langgraph>=0.2" \
    "openai>=1.30" \
    "apscheduler>=3.10" \
    "feedparser>=6.0" \
    "beautifulsoup4>=4.12" \
    "httpx>=0.27" \
    "pydantic>=2.0" \
    "python-dotenv>=1.0"

COPY src ./src

EXPOSE 8001

CMD ["uvicorn", "src.api.router:app", "--host", "0.0.0.0", "--port", "8001"]
