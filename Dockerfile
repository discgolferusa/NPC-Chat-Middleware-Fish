# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV GAME_API_HOST=0.0.0.0 \
    GAME_API_PORT=5000 \
    VOICES_DIR=/voices \
    MEMORY_DB=/data/npc_memory.db \
    UPLOAD_DIR=/data/uploads \
    OUTPUT_DIR=/data/outputs \
    PYTHONUNBUFFERED=1

RUN mkdir -p /voices /data/uploads /data/outputs /root/.cache/huggingface

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=4)"

CMD ["python", "server.py"]
