FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=3

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY src/ ./src/
COPY assets/ ./assets/
COPY data/ ./data/
COPY models/ ./models/

EXPOSE 8050

CMD ["gunicorn", "app:server", "-b", "0.0.0.0:8050", "--workers", "2", "--timeout", "120"]
