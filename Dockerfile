FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir .

EXPOSE 18420

CMD ["uvicorn", "ai.main:app", "--host", "0.0.0.0", "--port", "18420"]
