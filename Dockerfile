FROM python:3.11-slim

WORKDIR /app

# Tinker SDK pulls native deps for pyqwest/httpx; keep build minimal.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

ENV PORT=11435
ENV HOST=0.0.0.0
EXPOSE 11435

# Unbuffered stdout so Caprover's log tail shows lines as they happen.
ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
