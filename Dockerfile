# ── Stage 1: Install dependencies ─────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# System deps for packages that need compilation (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install into a prefix dir so we can copy cleanly
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime image ─────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source code only (exclude .venv, __pycache__, tests, .git)
COPY app ./app
COPY pyproject.toml .

# Expose port
EXPOSE 3000

# Run server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3000"]
