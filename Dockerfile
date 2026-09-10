FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libopenblas0-pthread \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir .[dev]

COPY . .

ENV PYTHONPATH="/app"

EXPOSE 7134

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "7134"]
