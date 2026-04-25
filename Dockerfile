FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    F1_DATA_DIR=/app/data

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-deps -e .

RUN useradd --create-home --uid 1000 f1 \
    && mkdir -p /app/data && chown -R f1:f1 /app
USER f1

VOLUME ["/app/data"]

ENTRYPOINT ["python", "-m", "f1_predictor"]
CMD ["weekly"]
