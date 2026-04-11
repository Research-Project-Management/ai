FROM python:3.12-slim

WORKDIR /app

# Install Python deps (all packages have pre-built wheels — no compiler needed)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Expose port
EXPOSE 3000

# Run server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3000"]
